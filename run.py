from __future__ import annotations

import argparse
from pathlib import Path

from waggle.annotations import build_manifest_from_annotations
from waggle.cache import build_cache
from waggle.predict import predict_video
from waggle.train_bin import train_binary


def main() -> None:
    ap = argparse.ArgumentParser(prog="wagglechat")
    sub = ap.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep", help="Build train/val manifests from annotation CSVs.")
    prep.add_argument("--data_dir", type=Path, default=Path("data"))
    prep.add_argument("--out_dir", type=Path, default=Path("runs/manifests"))
    prep.add_argument("--fps", type=int, default=30)
    prep.add_argument("--clip_frames", type=int, default=16)
    prep.add_argument("--stride", type=int, default=2)
    prep.add_argument("--center_stride_frames", type=int, default=2)
    prep.add_argument("--neg_per_pos", type=int, default=3)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--train_frac", type=float, default=0.8)
    prep.add_argument("--max_samples", type=int, default=0)

    tr = sub.add_parser("train", help="Train binary waggle detector.")
    tr.add_argument("--train_manifest", type=Path, required=True)
    tr.add_argument("--val_manifest", type=Path, required=True)
    tr.add_argument("--out_dir", type=Path, default=Path("runs/train"))
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=5e-3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--prefetch_factor", type=int, default=2)
    tr.add_argument("--val_every", type=int, default=1)
    tr.add_argument("--val_max_batches", type=int, default=60)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--device", type=str, default=None)
    tr.add_argument("--cache_train_dir", type=Path, default=None)
    tr.add_argument("--cache_val_dir", type=Path, default=None)

    pr = sub.add_parser("predict", help="Run detector on a video and write segments CSV.")
    pr.add_argument("--video", type=Path, required=True)
    pr.add_argument("--ckpt", type=Path, required=True)
    pr.add_argument("--out_csv", type=Path, default=Path("runs/predicted_waggles.csv"))
    pr.add_argument("--fps", type=int, default=30)
    pr.add_argument("--clip_frames", type=int, default=16)
    pr.add_argument("--stride", type=int, default=2)
    pr.add_argument("--infer_step_frames", type=int, default=8)
    pr.add_argument("--thr_on", type=float, default=0.6)
    pr.add_argument("--thr_off", type=float, default=0.4)
    pr.add_argument("--min_len_frames", type=int, default=8)
    pr.add_argument("--gap_frames", type=int, default=8)
    pr.add_argument("--device", type=str, default=None)

    ca = sub.add_parser("cache", help="Cache decoded clips for a manifest.")
    ca.add_argument("--manifest_csv", type=Path, required=True)
    ca.add_argument("--out_dir", type=Path, required=True)
    ca.add_argument("--fps", type=int, default=30)
    ca.add_argument("--clip_frames", type=int, default=16)
    ca.add_argument("--stride", type=int, default=2)
    ca.add_argument("--shard_size", type=int, default=512)

    args = ap.parse_args()
    if args.cmd == "prep":
        build_manifest_from_annotations(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            fps=args.fps,
            clip_frames=args.clip_frames,
            stride=args.stride,
            center_stride_frames=args.center_stride_frames,
            neg_per_pos=args.neg_per_pos,
            seed=args.seed,
            train_frac=args.train_frac,
            max_samples=args.max_samples,
        )
    elif args.cmd == "train":
        train_binary(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            out_dir=args.out_dir,
            cache_train_dir=args.cache_train_dir,
            cache_val_dir=args.cache_val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            val_every=args.val_every,
            val_max_batches=args.val_max_batches,
            amp=args.amp,
            device=args.device,
        )
    elif args.cmd == "predict":
        predict_video(
            video=args.video,
            ckpt=args.ckpt,
            out_csv=args.out_csv,
            fps=args.fps,
            clip_frames=args.clip_frames,
            stride=args.stride,
            infer_step_frames=args.infer_step_frames,
            thr_on=args.thr_on,
            thr_off=args.thr_off,
            min_len_frames=args.min_len_frames,
            gap_frames=args.gap_frames,
            device=args.device,
        )
    elif args.cmd == "cache":
        build_cache(
            manifest_csv=args.manifest_csv,
            out_dir=args.out_dir,
            fps=args.fps,
            clip_frames=args.clip_frames,
            stride=args.stride,
            shard_size=args.shard_size,
        )
    else:
        raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

