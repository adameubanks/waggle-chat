from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.video_reader")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from waggle.manifest import build_manifest, write_splits
from waggle.model import WaggleNet
from waggle.video_dataset import WaggleWindowDataset, compute_pos_weight


def _focal_bce_logits(
    logit: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, gamma: float
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logit)
    pt = target * prob + (1.0 - target) * (1.0 - prob)
    return (loss * (1.0 - pt).pow(gamma)).mean()


def _log1p_dur(y: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(y, min=0.0))


def _dur_smoothl1(pred_log: torch.Tensor, y_dur: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    t = _log1p_dur(y_dur)
    m = mask.float()
    per = F.smooth_l1_loss(pred_log, t, reduction="none")
    return (per * m).sum() / m.sum().clamp_min(1.0)


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


def _best_f1_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        _, _, f1, _ = _cls_metrics(probs, y, float(thr))
        if f1 > best_f:
            best_f, best_t = float(f1), float(thr)
    return best_t, best_f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("."))
    ap.add_argument("--out_dir", type=Path, default=Path("runs/r3d18"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window_seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--duration_loss_weight", type=float, default=1.0)
    ap.add_argument("--duration_warmup_epochs", type=int, default=2)
    ap.add_argument("--lr_warmup_epochs", type=int, default=3)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--cls_threshold", type=float, default=0.5)
    ap.add_argument("--threshold_sweep", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    annotations_dir = args.data_dir / "annotations"
    videos_dir = args.data_dir / "raw_videos"
    csv_paths = sorted(annotations_dir.glob("filtered_waggles_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No filtered_waggles_*.csv found in {annotations_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(videos_dir=videos_dir, csv_paths=csv_paths, window_seconds=args.window_seconds, fps=args.fps)
    train_csv, val_csv = write_splits(manifest, out_dir=args.out_dir, train_frac=0.8, seed=args.seed)

    train_ds = WaggleWindowDataset(train_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)
    val_ds = WaggleWindowDataset(val_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pos_weight = torch.tensor([compute_pos_weight(train_csv)], device=device, dtype=torch.float32)
    val_pos_weight = torch.tensor([compute_pos_weight(val_csv)], device=device, dtype=torch.float32)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        lr_scale = min(1.0, epoch / float(max(1, args.lr_warmup_epochs)))
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_scale
        dur_w = 0.0 if epoch <= args.duration_warmup_epochs else args.duration_loss_weight

        model.train()
        pbar = tqdm(train_dl, desc=f"train e{epoch}", leave=False)
        for clip, y_cls, y_dur in pbar:
            clip = clip.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)
            y_dur = y_dur.to(device, non_blocking=True)

            out = model(clip)
            logit = out[:, 0]
            pred_log = out[:, 1]

            loss_cls = _focal_bce_logits(logit, y_cls, pos_weight, args.focal_gamma)
            mask = y_cls > 0.5
            loss_dur = _dur_smoothl1(pred_log, y_dur, mask)
            loss = loss_cls + dur_w * loss_dur

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        val_cls_sum = 0.0
        val_dur_sl_sum = 0.0
        val_dur_sl_n = 0
        val_dur_mae_sum = 0.0
        val_dur_mae_n = 0
        val_n = 0
        logit_chunks: list[torch.Tensor] = []
        y_cls_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for clip, y_cls, y_dur in tqdm(val_dl, desc=f"val e{epoch}", leave=False):
                clip = clip.to(device, non_blocking=True)
                y_cls = y_cls.to(device, non_blocking=True)
                y_dur = y_dur.to(device, non_blocking=True)

                out = model(clip)
                logit = out[:, 0]
                pred_log = out[:, 1]

                bs = int(len(clip))
                val_cls_sum += float(_focal_bce_logits(logit, y_cls, val_pos_weight, args.focal_gamma).detach().cpu()) * bs
                mask = y_cls > 0.5
                if mask.any():
                    sl = F.smooth_l1_loss(pred_log, _log1p_dur(y_dur), reduction="none")
                    val_dur_sl_sum += float((sl * mask.float()).sum().detach().cpu())
                    val_dur_sl_n += int(mask.sum())
                    pred_s = torch.expm1(torch.clamp(pred_log, max=12.0))
                    err = (pred_s - y_dur).abs()[mask]
                    val_dur_mae_sum += float(err.sum().detach().cpu())
                    val_dur_mae_n += int(mask.sum())
                val_n += bs
                logit_chunks.append(logit.detach().float().cpu())
                y_cls_chunks.append(y_cls.detach().float().cpu())

        val_n = max(1, val_n)
        val_cls = val_cls_sum / val_n
        val_dur_sl = val_dur_sl_sum / max(1, val_dur_sl_n)
        val_dur_mae = val_dur_mae_sum / max(1, val_dur_mae_n)
        dur_w_eff = 0.0 if epoch <= args.duration_warmup_epochs else args.duration_loss_weight
        val_score = val_cls + dur_w_eff * val_dur_sl

        all_logits = torch.cat(logit_chunks, dim=0)
        all_y = torch.cat(y_cls_chunks, dim=0)
        probs = torch.sigmoid(all_logits).numpy()
        y_np = all_y.numpy()
        thr = float(args.cls_threshold)
        prec, rec, f1, acc = _cls_metrics(probs, y_np, thr)
        best_t, best_f1 = (None, None)
        if args.threshold_sweep:
            best_t, best_f1 = _best_f1_threshold(probs, y_np)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "val_loss": val_score,
            "val_cls_loss": val_cls,
            "val_dur_smooth_l1": val_dur_sl,
            "val_dur_mae_s": val_dur_mae,
            "val_dur_n": val_dur_mae_n,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_accuracy": acc,
            "val_cls_threshold": thr,
            "val_best_f1_threshold": best_t,
            "val_best_f1": best_f1,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_score < best_val:
            best_val = val_score
            torch.save(ckpt, args.out_dir / "best.pt")

        extra = ""
        if args.threshold_sweep and best_t is not None and best_f1 is not None:
            extra = f" val_best_f1={best_f1:.4f}@thr={best_t:.2f}"
        print(
            f"epoch={epoch} lr_scale={lr_scale:.2f} dur_w={dur_w_eff:.2f} "
            f"val_score={val_score:.6f} val_cls={val_cls:.6f} val_dur_sl1={val_dur_sl:.4f} "
            f"val_dur_mae_s={val_dur_mae:.4f}(n={val_dur_mae_n}) "
            f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} acc={acc:.4f}@cls_thr={thr:.2f}{extra}"
        )


if __name__ == "__main__":
    main()
