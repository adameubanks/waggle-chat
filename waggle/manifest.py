from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from waggle.video_dataset import max_valid_center, min_valid_center, probe_total_frames


@dataclass(frozen=True)
class Sample:
    video: str
    center_frame: int
    is_waggle: int
    duration_s: float


def _in_any_segment(c: int, starts: np.ndarray, ends: np.ndarray) -> bool:
    return bool(np.any((starts <= c) & (c <= ends)))


def build_manifest(
    *,
    videos_dir: Path,
    csv_paths: list[Path],
    fps: int = 30,
    clip_frames: int = 32,
    decode_stride: int = 2,
    center_stride_frames: int = 1,
    neg_per_pos: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    center_stride_frames = max(1, int(center_stride_frames))
    rng = np.random.default_rng(seed)
    rows: list[Sample] = []
    used: set[tuple[str, int]] = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        need = {"startFrame", "endFrame", "duration"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns {sorted(missing)}: {list(df.columns)}")

        stem = csv_path.name.replace("filtered_waggles_", "").replace(".csv", "")
        video_path = videos_dir / f"{stem}.mp4"
        vid = str(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for {csv_path.name}: {video_path}")

        max_end = int(df["endFrame"].max())
        total_frames = max(max_end + 1, probe_total_frames(video_path, fps))
        cmin = min_valid_center(clip_frames, decode_stride)
        cmax = max_valid_center(total_frames, clip_frames, decode_stride)
        if cmax < cmin:
            continue

        events = df[["startFrame", "endFrame", "duration"]].copy()
        s_arr = events["startFrame"].to_numpy(dtype=np.int64)
        e_arr = events["endFrame"].to_numpy(dtype=np.int64)

        for _, r in events.iterrows():
            a = int(r["startFrame"])
            b = int(r["endFrame"])
            dur = float(r["duration"])
            for c in range(a, b + 1, center_stride_frames):
                if c < cmin or c > cmax:
                    continue
                k = (vid, c)
                if k in used:
                    continue
                used.add(k)
                rows.append(Sample(video=vid, center_frame=c, is_waggle=1, duration_s=dur))

        n_pos = sum(1 for r in rows if r.video == vid and r.is_waggle == 1)
        n_neg_target = max(1, int(neg_per_pos) * n_pos) if n_pos else max(1, int(neg_per_pos))
        neg_added = 0
        tries = 0
        while neg_added < n_neg_target:
            tries += 1
            if tries > n_neg_target * 2000:
                break
            c = int(rng.integers(cmin, cmax + 1))
            if _in_any_segment(c, s_arr, e_arr):
                continue
            k = (vid, c)
            if k in used:
                continue
            used.add(k)
            rows.append(Sample(video=vid, center_frame=c, is_waggle=0, duration_s=0.0))
            neg_added += 1

    return pd.DataFrame([r.__dict__ for r in rows])


def write_splits(
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    vids = np.array(sorted(manifest["video"].unique()))
    rng.shuffle(vids)
    n_train = int(round(len(vids) * train_frac))
    train_vids = set(vids[:n_train].tolist())

    train_df = manifest[manifest["video"].isin(train_vids)].reset_index(drop=True)
    val_df = manifest[~manifest["video"].isin(train_vids)].reset_index(drop=True)

    train_path = out_dir / "train_manifest.csv"
    val_path = out_dir / "val_manifest.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return train_path, val_path
