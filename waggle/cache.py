from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from waggle.video_dataset import decode_start_for_center, middle_tensor_index


@dataclass(frozen=True)
class CacheMeta:
    clip_frames: int
    stride: int
    resize_hw: tuple[int, int]


def build_cache(
    *,
    manifest_csv: Path,
    out_dir: Path,
    fps: int = 30,
    clip_frames: int = 16,
    stride: int = 2,
    shard_size: int = 512,
    resize_hw: tuple[int, int] = (112, 112),
) -> Path:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"cache requires opencv-python (cv2). Import failed: {e}")

    manifest_csv = manifest_csv.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    need = {"video", "center_frame", "is_waggle"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{manifest_csv} missing {sorted(missing)}")

    meta = CacheMeta(clip_frames=int(clip_frames), stride=int(stride), resize_hw=tuple(resize_hw))
    meta_path = out_dir / "meta.npz"
    np.savez(meta_path, clip_frames=meta.clip_frames, stride=meta.stride, h=meta.resize_hw[0], w=meta.resize_hw[1])

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    shard_id = 0
    cur_centers: list[int] = []
    cur_labels: list[float] = []
    cur_g: list[np.ndarray] = []

    def flush() -> None:
        nonlocal shard_id, cur_centers, cur_labels, cur_g
        if not cur_centers:
            return
        shard = f"shard_{shard_id:05d}"
        g = np.stack(cur_g, axis=0).astype(np.uint8)  # N,T,H,W
        centers = np.asarray(cur_centers, dtype=np.int64)
        labels = np.asarray(cur_labels, dtype=np.float32)
        np.save(shards_dir / f"{shard}_g.npy", g, allow_pickle=False)
        np.save(shards_dir / f"{shard}_centers.npy", centers, allow_pickle=False)
        np.save(shards_dir / f"{shard}_labels.npy", labels, allow_pickle=False)
        for i in range(len(cur_centers)):
            index_rows.append({"shard": shard, "offset": i})
        shard_id += 1
        cur_centers = []
        cur_labels = []
        cur_g = []

    by_video = df.groupby("video", sort=True)
    mid = middle_tensor_index(meta.clip_frames)
    for vid, part in by_video:
        centers = part["center_frame"].astype(int).to_numpy()
        labels = part["is_waggle"].astype(float).to_numpy()

        starts = np.asarray([decode_start_for_center(int(c), meta.clip_frames, meta.stride) for c in centers], dtype=np.int64)
        ends = starts + (meta.clip_frames - 1) * meta.stride
        order = np.argsort(ends)
        centers = centers[order]
        labels = labels[order]
        starts = starts[order]
        ends = ends[order]

        pending: dict[int, list[int]] = {}
        for local_i, e in enumerate(ends.tolist()):
            pending.setdefault(int(e), []).append(local_i)

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            raise SystemExit(f"Failed to open {vid}")

        ring: list[np.ndarray] = []
        ring_start = 0
        span = (meta.clip_frames - 1) * meta.stride
        max_keep = span + 1 + meta.stride * 2

        frame_idx = 0
        max_end = int(ends.max()) if ends.size else -1
        while frame_idx <= max_end:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (meta.resize_hw[1], meta.resize_hw[0]), interpolation=cv2.INTER_AREA)
            ring.append(g.astype(np.uint8))
            while len(ring) > max_keep:
                ring.pop(0)
                ring_start += 1

            if frame_idx in pending:
                for local_i in pending[int(frame_idx)]:
                    s = int(starts[local_i])
                    need_idxs = [s + k * meta.stride for k in range(meta.clip_frames)]
                    if need_idxs[0] < ring_start or need_idxs[-1] > (ring_start + len(ring) - 1):
                        continue
                    clip = np.stack([ring[ii - ring_start] for ii in need_idxs], axis=0)  # T,H,W
                    cur_centers.append(int(centers[local_i]))
                    cur_labels.append(float(labels[local_i]))
                    cur_g.append(clip)
                    if len(cur_centers) >= int(shard_size):
                        flush()
            frame_idx += 1

        cap.release()

    flush()

    index = pd.DataFrame(index_rows)
    out_index = out_dir / "index.csv"
    index.to_csv(out_index, index=False)
    return out_index

