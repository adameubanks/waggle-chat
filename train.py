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
    ap.add_argument("--window_seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--cls_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    annotations_dir = args.data_dir / "annotations"
    videos_dir = args.data_dir / "raw_videos"
    csv_paths = sorted(annotations_dir.glob("filtered_waggles_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No filtered_waggles_*.csv in {annotations_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(videos_dir=videos_dir, csv_paths=csv_paths, window_seconds=args.window_seconds, fps=args.fps)
    train_csv, val_csv = write_splits(manifest, out_dir=args.out_dir, train_frac=0.8, seed=args.seed)

    train_ds = WaggleWindowDataset(train_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)
    val_ds = WaggleWindowDataset(val_csv, fps=args.fps, clip_frames=args.clip_frames, stride=args.stride)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    pos_w = torch.tensor([compute_pos_weight(train_csv)], device=device, dtype=torch.float32)
    val_pos_w = torch.tensor([compute_pos_weight(val_csv)], device=device, dtype=torch.float32)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train {epoch}", leave=False)
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

            pbar.set_postfix(bce=float(loss_cls.detach()), dur=float(loss_dur.detach()))

        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])

        model.eval()
        v_cls = v_dur_l1 = v_dur_mae = 0.0
        v_n = vd_n = 0
        v_mae_sum = 0.0
        logits_all: list[torch.Tensor] = []
        y_all: list[torch.Tensor] = []
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
        prob_mean = float(probs.mean())
        pos_rate = float((y_cat > 0.5).float().mean())

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "args": vars(args),
            "val_total": val_total,
            "val_bce": v_cls,
            "val_dur_l1_log": v_dur_l1,
            "val_dur_mae_s": v_dur_mae,
            "val_dur_n": vd_n,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_accuracy": acc,
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
            f"P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f} acc={acc:.3f} thr={args.cls_threshold} | "
            f"prob_mean={prob_mean:.3f} pos_rate={pos_rate:.3f}"
        )


if __name__ == "__main__":
    main()
