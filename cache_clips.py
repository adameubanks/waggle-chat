from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.video_reader")

from waggle.video_dataset import read_clip_at_center


def _clip_key(video: str, center: int, clip_frames: int, stride: int, fps: int, resize: int) -> str:
    raw = f"{video}\n{center}\n{clip_frames}\n{stride}\n{fps}\n{resize}".encode()
    h = hashlib.sha256(raw).hexdigest()
    return f"{h[:2]}/{h[2:4]}/{h}.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode each manifest row once; save resized clip .pt + new CSV with clip_pt column.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_manifest", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--resize", type=int, default=112)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    for c in ("video", "center_frame", "is_waggle", "duration_s"):
        if c not in df.columns:
            raise SystemExit(f"manifest missing column {c}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    resize_hw = (args.resize, args.resize)
    clip_pts: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="cache_clips"):
        video = str(row["video"])
        center = int(row["center_frame"])
        rel = _clip_key(video, center, args.clip_frames, args.stride, args.fps, args.resize)
        out_pt = args.cache_dir / rel
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        if not out_pt.exists():
            clip = read_clip_at_center(
                video_path=video,
                center_frame=center,
                fps=args.fps,
                clip_frames=args.clip_frames,
                stride=args.stride,
            )
            clip = torch.nn.functional.interpolate(
                clip,
                size=resize_hw,
                mode="bilinear",
                align_corners=False,
            )
            torch.save(clip.contiguous(), out_pt)
        clip_pts.append(str(out_pt.resolve()))

    out_df = df.copy()
    out_df["clip_pt"] = clip_pts
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_manifest, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out_manifest}")
    print(f"Cache root: {args.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
