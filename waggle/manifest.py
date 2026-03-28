from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    video: str
    start_frame: int
    end_frame: int
    is_waggle: int
    duration_s: float


def _window_targets(events: pd.DataFrame, start_f: int, end_f: int) -> tuple[int, float]:
    if events.empty:
        return 0, 0.0

    s = events["startFrame"].to_numpy(dtype=np.int64)
    e = events["endFrame"].to_numpy(dtype=np.int64)
    overlap = np.maximum(0, np.minimum(e, end_f) - np.maximum(s, start_f) + 1)
    idx = int(np.argmax(overlap))
    if int(overlap[idx]) == 0:
        return 0, 0.0

    dur = float(events.iloc[idx]["duration"])
    return 1, dur


def build_manifest(
    *,
    videos_dir: Path,
    csv_paths: list[Path],
    window_seconds: float = 3.0,
    fps: int = 30,
) -> pd.DataFrame:
    window_frames = int(round(window_seconds * fps))
    rows: list[Window] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        need = {"startFrame", "endFrame", "duration"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} missing columns {sorted(missing)}: {list(df.columns)}")

        stem = csv_path.name.replace("filtered_waggles_", "").replace(".csv", "")
        video_path = videos_dir / f"{stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for {csv_path.name}: {video_path}")

        max_end = int(df["endFrame"].max())
        total_frames_est = max_end + 1
        n_windows = int(math.ceil(total_frames_est / window_frames))

        events = df[["startFrame", "endFrame", "duration"]].copy()

        for w in range(n_windows):
            start_f = w * window_frames
            end_f = start_f + window_frames - 1
            is_waggle, dur = _window_targets(events, start_f, end_f)
            rows.append(
                Window(
                    video=str(video_path),
                    start_frame=int(start_f),
                    end_frame=int(end_f),
                    is_waggle=int(is_waggle),
                    duration_s=float(dur),
                )
            )

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
