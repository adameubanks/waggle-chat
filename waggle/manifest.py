from __future__ import annotations

import time
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

    n_csv = len(csv_paths)
    t_all = time.perf_counter()
    print(f"[manifest] building from {n_csv} annotation file(s), videos_dir={videos_dir.resolve()}", flush=True)

    for i, csv_path in enumerate(csv_paths, start=1):
        t_vid = time.perf_counter()
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

        print(f"[manifest] {i}/{n_csv} {video_path.name} (annotations {len(df)} events)", flush=True)
        max_end = int(df["endFrame"].max())
        t0 = time.perf_counter()
        probed = probe_total_frames(video_path, fps)
        t_probe = time.perf_counter() - t0
        total_frames = max(max_end + 1, probed)
        print(
            f"[manifest]   frame span: annotation max_end={max_end} probed_total={probed} -> using total_frames={total_frames} (probe {t_probe:.2f}s)",
            flush=True,
        )
        cmin = min_valid_center(clip_frames, decode_stride)
        cmax = max_valid_center(total_frames, clip_frames, decode_stride)
        if cmax < cmin:
            print(f"[manifest]   skip: no valid center window (cmin={cmin} cmax={cmax})", flush=True)
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

        print(
            f"[manifest]   samples: pos={n_pos} neg={neg_added}/{n_neg_target} center in [{cmin},{cmax}] "
            f"(video total {time.perf_counter() - t_vid:.2f}s, rows in manifest {len(rows)})",
            flush=True,
        )

    out = pd.DataFrame([r.__dict__ for r in rows])
    print(
        f"[manifest] done: {len(out)} rows, {out['video'].nunique()} video(s), elapsed {time.perf_counter() - t_all:.2f}s",
        flush=True,
    )
    return out


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
    print(
        f"[manifest] split -> {train_path.name}: {len(train_df)} rows, {train_df['video'].nunique()} videos | "
        f"{val_path.name}: {len(val_df)} rows, {val_df['video'].nunique()} videos",
        flush=True,
    )
    return train_path, val_path
