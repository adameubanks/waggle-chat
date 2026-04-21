from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from waggle.bin_dataset import BinaryWaggleDataset, CachedBinaryWaggleDataset, compute_pos_weight
from waggle.model import WaggleNet


def _dist_info() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    return int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ["WORLD_SIZE"])


def _average_precision(probs: np.ndarray, y: np.ndarray) -> float:
    yb = (y > 0.5).astype(np.int64)
    npos = int(yb.sum())
    if npos == 0:
        return 0.0
    order = np.argsort(-probs)
    y_sorted = yb[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / npos
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(prec.tolist(), rec.tolist()):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap)


def _best_f1(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    yb = (y > 0.5).astype(np.int64)
    npos = int(yb.sum())
    if npos == 0 or probs.size == 0:
        return 0.0, 0.5
    order = np.argsort(-probs)
    p_sorted = probs[order]
    y_sorted = yb[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / npos
    f1 = (2 * prec * rec) / np.maximum(1e-12, prec + rec)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(p_sorted[idx])


def train_binary(
    *,
    train_manifest: Path,
    val_manifest: Path,
    out_dir: Path,
    cache_train_dir: Path | None = None,
    cache_val_dir: Path | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 5e-3,
    seed: int = 0,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    val_every: int = 1,
    val_max_batches: int = 60,
    amp: bool = False,
    device: str | None = None,
) -> Path:
    rank, local_rank, world = _dist_info()
    distributed = world > 1
    if distributed:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        backend = "none"
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = out_dir.expanduser().resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] device={dev} distributed={distributed} world={world} backend={backend}", flush=True)
    if distributed:
        dist.barrier()

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    use_amp = bool(amp) and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    if cache_train_dir is not None:
        train_ds = CachedBinaryWaggleDataset(train_manifest, cache_train_dir)
    else:
        train_ds = BinaryWaggleDataset(train_manifest)
    if cache_val_dir is not None:
        val_ds = CachedBinaryWaggleDataset(val_manifest, cache_val_dir)
    else:
        val_ds = BinaryWaggleDataset(val_manifest)

    dl_common = {"pin_memory": dev.type == "cuda", "num_workers": int(num_workers), "persistent_workers": int(num_workers) > 0}
    if int(num_workers) > 0:
        dl_common["prefetch_factor"] = int(prefetch_factor)

    train_sampler: DistributedSampler | None = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **dl_common,
    )
    val_dl = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, drop_last=False, **dl_common)

    model = WaggleNet(pretrained=True, in_channels=2).to(dev)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    lr_eff = float(lr) * world if distributed else float(lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(lr_eff) * 0.05)

    pos_w = torch.tensor([compute_pos_weight(train_manifest)], device=dev, dtype=torch.float32)

    def _state_dict():
        m = model.module if isinstance(model, DDP) else model
        return m.state_dict()

    best_ap = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    t0 = time.perf_counter()
    try:
        for epoch in range(1, int(epochs) + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            pbar = tqdm(train_dl, desc=f"train {epoch}", leave=False, disable=rank != 0)
            for x, y in pbar:
                x = x.to(dev, non_blocking=True)
                y = y.to(dev, non_blocking=True)
                with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                    logit = model(x)
                    loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                if rank == 0:
                    pbar.set_postfix(bce=float(loss.detach()))
            sched.step()

            if val_every > 0 and epoch % int(val_every) == 0:
                if distributed:
                    dist.barrier()
                if rank == 0:
                    model.eval()
                    logits: list[np.ndarray] = []
                    ys: list[np.ndarray] = []
                    v_loss_sum = 0.0
                    v_n = 0
                    with torch.no_grad():
                        for i, (x, y) in enumerate(tqdm(val_dl, desc=f"val {epoch}", leave=False)):
                            if val_max_batches and i >= int(val_max_batches):
                                break
                            x = x.to(dev, non_blocking=True)
                            y = y.to(dev, non_blocking=True)
                            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                                logit = model(x).float()
                                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
                            bs = int(len(x))
                            v_loss_sum += float(loss) * bs
                            v_n += bs
                            logits.append(logit.detach().cpu().numpy())
                            ys.append(y.detach().cpu().numpy())
                    logits_cat = np.concatenate(logits) if logits else np.zeros((0,), dtype=np.float32)
                    y_cat = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.float32)
                    probs = 1.0 / (1.0 + np.exp(-logits_cat))
                    ap = _average_precision(probs, y_cat)
                    best_f1, best_thr = _best_f1(probs, y_cat)
                    v_loss = v_loss_sum / max(1, v_n)
                    ckpt = {
                        "epoch": epoch,
                        "model": _state_dict(),
                        "args": {
                            "train_manifest": str(train_manifest),
                            "val_manifest": str(val_manifest),
                            "epochs": int(epochs),
                            "batch_size": int(batch_size),
                            "world_size": world,
                            "lr_base": float(lr),
                            "lr_effective": float(lr_eff),
                            "weight_decay": float(weight_decay),
                        },
                        "val_bce": float(v_loss),
                        "val_ap": float(ap),
                        "val_best_f1": float(best_f1),
                        "val_best_f1_thr": float(best_thr),
                    }
                    torch.save(ckpt, last_path)
                    if ap > best_ap:
                        best_ap = ap
                        torch.save(ckpt, best_path)
                    print(
                        f"epoch={epoch}/{epochs} lr={opt.param_groups[0]['lr']:.2e} "
                        f"val_bce={v_loss:.5f} AP={ap:.3f} bestF1={best_f1:.3f}@{best_thr:.3f} "
                        f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
                        flush=True,
                    )
                if distributed:
                    dist.barrier()
    finally:
        if distributed:
            dist.destroy_process_group()

    return best_path if best_path.exists() else last_path

