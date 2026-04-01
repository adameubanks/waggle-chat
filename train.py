from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.video_reader")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from waggle.manifest import build_manifest, write_splits
from waggle.model import WaggleNet
from waggle.video_dataset import WaggleWindowDataset, compute_pos_weight


def _log1p(y: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(y, min=0.0))


def _cls_metrics(probs: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float, float]:
    pred = (probs >= thr).astype(np.int64)
    yt = (y > 0.5).astype(np.int64)
    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, acc


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


def _setup_dist() -> tuple[bool, int, int, int]:
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, local_rank, ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("."))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/train"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--duration_loss_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center_stride_frames", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--neg_per_pos", type=int, default=3)
    ap.add_argument("--cls_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--train_manifest", type=Path, default=None)
    ap.add_argument("--val_manifest", type=Path, default=None)
    ap.add_argument("--no_build_manifest", action="store_true")
    args = ap.parse_args()

    is_dist, rank, local_rank, world_size = _setup_dist()
    if is_dist:
        args.lr = args.lr * world_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()

    if args.no_build_manifest:
        if args.train_manifest is None or args.val_manifest is None:
            raise SystemExit("--no_build_manifest requires --train_manifest and --val_manifest")
        train_csv = args.train_manifest
        val_csv = args.val_manifest
    else:
        annotations_dir = args.data_dir / "annotations"
        videos_dir = args.data_dir / "raw_videos"
        csv_paths = sorted(annotations_dir.glob("filtered_waggles_*.csv"))
        if not csv_paths:
            raise SystemExit(f"No filtered_waggles_*.csv in {annotations_dir}")
        if rank == 0:
            print(f"building manifest ({len(csv_paths)} annotation files)...", flush=True)
            manifest = build_manifest(
                videos_dir=videos_dir,
                csv_paths=csv_paths,
                fps=args.fps,
                clip_frames=args.clip_frames,
                decode_stride=args.stride,
                center_stride_frames=args.center_stride_frames,
                neg_per_pos=args.neg_per_pos,
                seed=args.seed,
            )
            if args.max_samples > 0 and len(manifest) > args.max_samples:
                manifest = manifest.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
            write_splits(manifest, out_dir=args.out_dir, train_frac=0.8, seed=args.seed)
            print(f"wrote manifests: {len(manifest)} rows -> {args.out_dir}", flush=True)
        if is_dist:
            dist.barrier()
        train_csv = args.out_dir / "train_manifest.csv"
        val_csv = args.out_dir / "val_manifest.csv"

    train_ds = WaggleWindowDataset(train_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)
    val_ds = WaggleWindowDataset(val_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if is_dist else None
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=True).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if is_dist else model

    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    pos_w = torch.tensor([compute_pos_weight(train_csv)], device=device, dtype=torch.float32)
    val_pos_w = torch.tensor([compute_pos_weight(val_csv)], device=device, dtype=torch.float32)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_dl, desc=f"train {epoch}", leave=False, disable=rank != 0)
        for clip, y_cls, y_dur in pbar:
            clip = clip.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)

            out = model(clip)
            logit = out[:, 0]
            pred_log = out[:, 1]
            tgt_log = _log1p(y_dur)
            wmask = y_cls > 0.5

            loss_cls = F.binary_cross_entropy_with_logits(logit, y_cls, pos_weight=pos_w)
            if wmask.any():
                loss_dur = F.l1_loss(pred_log[wmask], tgt_log[wmask])
            else:
                loss_dur = torch.tensor(0.0, device=device)
            loss = loss_cls + args.duration_loss_weight * loss_dur

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if rank == 0:
                pbar.set_postfix(bce=float(loss_cls.detach()), dur=float(loss_dur.detach()))

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])

        v_cls = v_dur_l1 = v_dur_mae = 0.0
        v_n = vd_n = 0
        v_mae_sum = 0.0
        logits_all: list[torch.Tensor] = []
        y_all: list[torch.Tensor] = []

        if rank == 0:
            model.eval()
            with torch.no_grad():
                for clip, y_cls, y_dur in tqdm(val_dl, desc=f"val {epoch}", leave=False):
                    clip = clip.to(device, non_blocking=True)
                    y_cls = y_cls.to(device, non_blocking=True)
                    y_dur = y_dur.to(device, non_blocking=True)

                    out = model(clip)
                    logit = out[:, 0]
                    pred_log = out[:, 1]
                    tgt_log = _log1p(y_dur)
                    bs = int(len(clip))

                    v_cls += float(F.binary_cross_entropy_with_logits(logit, y_cls, pos_weight=val_pos_w)) * bs
                    v_n += bs
                    wmask = y_cls > 0.5
                    if wmask.any():
                        v_dur_l1 += float(F.l1_loss(pred_log[wmask], tgt_log[wmask], reduction="sum"))
                        pred_s = torch.expm1(torch.clamp(pred_log, max=12.0))
                        v_mae_sum += float((pred_s[wmask] - y_dur[wmask]).abs().sum().cpu())
                        vd_n += int(wmask.sum())
                    logits_all.append(logit.float().cpu())
                    y_all.append(y_cls.float().cpu())

            v_cls /= max(1, v_n)
            v_dur_l1 /= max(1, vd_n)
            v_dur_mae = v_mae_sum / max(1, vd_n)
            val_total = v_cls + args.duration_loss_weight * v_dur_l1

            logits_cat = torch.cat(logits_all)
            y_cat = torch.cat(y_all)
            probs = torch.sigmoid(logits_cat).numpy()
            prec, rec, f1, acc = _cls_metrics(probs, y_cat.numpy(), float(args.cls_threshold))
            ap = _average_precision(probs, y_cat.numpy())
            prob_mean = float(probs.mean())
            pos_rate = float((y_cat > 0.5).float().mean())

            args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            ckpt = {
                "epoch": epoch,
                "model": raw.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "args": args_dict,
                "val_total": val_total,
                "val_bce": v_cls,
                "val_dur_l1_log": v_dur_l1,
                "val_dur_mae_s": v_dur_mae,
                "val_dur_n": vd_n,
                "val_precision": prec,
                "val_recall": rec,
                "val_f1": f1,
                "val_accuracy": acc,
                "val_ap": ap,
                "lr": lr_now,
            }
            torch.save(ckpt, args.out_dir / "last.pt")
            if val_total < best_val:
                best_val = val_total
                torch.save(ckpt, args.out_dir / "best.pt")

            print(
                f"epoch={epoch}/{args.epochs} lr={lr_now:.2e} | "
                f"val_bce={v_cls:.5f} val_dur_l1(log)={v_dur_l1:.5f} val_dur_mae_s={v_dur_mae:.4f} n_dur={vd_n} | "
                f"val_total={val_total:.5f} | "
                f"P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f} acc={acc:.3f} AP={ap:.3f} thr={args.cls_threshold} | "
                f"prob_mean={prob_mean:.3f} pos_rate={pos_rate:.3f}"
            )

        if is_dist:
            dist.barrier()

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
