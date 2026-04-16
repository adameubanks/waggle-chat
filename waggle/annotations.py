from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from waggle.manifest import write_splits
from waggle.video_dataset import max_valid_center, min_valid_center, probe_total_frames


@dataclass(frozen=True)
class Event:
    video: str
    start_frame: int
    end_frame: int
    angle: float


def _load_events(annotations_dir: Path, videos_dir: Path) -> list[Event]:
    csv_paths = sorted(annotations_dir.glob("filtered_waggles_*.csv"))
    if not csv_paths:
        raise SystemExit(f"No filtered_waggles_*.csv in {annotations_dir}")
    out: list[Event] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        need = {"startFrame", "endFrame", "angle"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"{p} missing {sorted(missing)}")
        stem = p.name.replace("filtered_waggles_", "").replace(".csv", "")
        vid_path = videos_dir / f"{stem}.mp4"
        if not vid_path.exists():
            raise SystemExit(f"Missing video for {p.name}: {vid_path}")
        for _, r in df.iterrows():
            out.append(
                Event(
                    video=str(vid_path),
                    start_frame=int(r["startFrame"]),
                    end_frame=int(r["endFrame"]),
                    angle=float(r["angle"]),
                )
            )
    return out


def build_manifest_from_annotations(
    *,
    data_dir: Path,
    out_dir: Path,
    fps: int = 30,
    clip_frames: int = 16,
    stride: int = 2,
    center_stride_frames: int = 2,
    neg_per_pos: int = 3,
    seed: int = 0,
    train_frac: float = 0.8,
    max_samples: int = 0,
) -> tuple[Path, Path]:
    data_dir = data_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = _load_events(data_dir / "annotations", data_dir / "raw_videos")
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    used: set[tuple[str, int]] = set()
    center_stride_frames = max(1, int(center_stride_frames))

    by_video: dict[str, list[Event]] = {}
    for ev in events:
        by_video.setdefault(ev.video, []).append(ev)

    for vid, evs in sorted(by_video.items()):
        max_end = max(e.end_frame for e in evs)
        total_frames = max(max_end + 1, probe_total_frames(Path(vid), fps))
        cmin = min_valid_center(clip_frames, stride)
        cmax = max_valid_center(total_frames, clip_frames, stride)
        if cmax < cmin:
            continue

        for e in evs:
            for c in range(e.start_frame, e.end_frame + 1, center_stride_frames):
                if c < cmin or c > cmax:
                    continue
                k = (vid, c)
                if k in used:
                    continue
                used.add(k)
                rows.append(
                    {
                        "video": vid,
                        "center_frame": int(c),
                        "is_waggle": 1,
                        "angle": float(e.angle),
                        "event_start": int(e.start_frame),
                        "event_end": int(e.end_frame),
                    }
                )

        pos_centers = [r["center_frame"] for r in rows if r["video"] == vid and r["is_waggle"] == 1]
        n_pos = len(pos_centers)
        n_neg_target = max(1, int(neg_per_pos) * n_pos) if n_pos else max(1, int(neg_per_pos))

        starts = np.asarray([e.start_frame for e in evs], dtype=np.int64)
        ends = np.asarray([e.end_frame for e in evs], dtype=np.int64)

        def in_any(c: int) -> bool:
            return bool(np.any((starts <= c) & (c <= ends)))

        neg_added = 0
        tries = 0
        while neg_added < n_neg_target:
            tries += 1
            if tries > n_neg_target * 2000:
                break
            c = int(rng.integers(cmin, cmax + 1))
            if in_any(c):
                continue
            k = (vid, c)
            if k in used:
                continue
            used.add(k)
            rows.append({"video": vid, "center_frame": int(c), "is_waggle": 0})
            neg_added += 1

    df = pd.DataFrame(rows)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    train_csv, val_csv = write_splits(df, out_dir=out_dir, train_frac=train_frac, seed=seed)
    return train_csv, val_csv

