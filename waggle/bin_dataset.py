from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np

from waggle.video_dataset import read_clip_at_center


class BinaryWaggleDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        *,
        fps: int = 30,
        clip_frames: int = 16,
        stride: int = 2,
        resize_hw: tuple[int, int] = (112, 112),
    ) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.fps = int(fps)
        self.clip_frames = int(clip_frames)
        self.stride = int(stride)
        self.resize_hw = tuple(resize_hw)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)]
        clip = read_clip_at_center(
            video_path=str(r["video"]),
            center_frame=int(r["center_frame"]),
            fps=self.fps,
            clip_frames=self.clip_frames,
            stride=self.stride,
        )  # T,H,W,C in [0,1]
        clip = clip.permute(0, 3, 1, 2).contiguous()  # T,C,H,W
        clip = torch.nn.functional.interpolate(clip, size=self.resize_hw, mode="bilinear", align_corners=False)
        g = (0.2989 * clip[:, 0] + 0.5870 * clip[:, 1] + 0.1140 * clip[:, 2]).unsqueeze(1)  # T,1,H,W
        d = g[1:] - g[:-1]
        d = torch.nn.functional.pad(d, (0, 0, 0, 0, 0, 0, 0, 1))  # T,1,H,W
        x = torch.cat([g, d], dim=1)  # T,2,H,W
        x = x.permute(1, 0, 2, 3).contiguous()  # C,T,H,W for 3D conv
        y = torch.tensor(float(r["is_waggle"]), dtype=torch.float32)
        return x, y


def compute_pos_weight(manifest_csv: Path) -> float:
    df = pd.read_csv(manifest_csv)
    pos = float(df["is_waggle"].sum())
    neg = float(len(df) - pos)
    return (neg / pos) if pos > 0 else 1.0


class CachedBinaryWaggleDataset(Dataset):
    def __init__(self, manifest_csv: Path, cache_dir: Path) -> None:
        self.df = pd.read_csv(manifest_csv)
        self.cache_dir = cache_dir.expanduser().resolve()
        self.index = pd.read_csv(self.cache_dir / "index.csv")
        if len(self.index) != len(self.df):
            raise SystemExit(f"cache size mismatch: index={len(self.index)} manifest={len(self.df)}")
        meta = np.load(self.cache_dir / "meta.npz")
        self.clip_frames = int(meta["clip_frames"])
        self.stride = int(meta["stride"])
        self.h = int(meta["h"])
        self.w = int(meta["w"])
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_shard(self, name: str) -> dict[str, np.ndarray]:
        got = self._shard_cache.get(name)
        if got is not None:
            return got
        path = self.cache_dir / "shards" / name
        z = np.load(path)
        data = {"g": z["g"], "labels": z["labels"], "centers": z["centers"]}
        if len(self._shard_cache) > 2:
            self._shard_cache.clear()
        self._shard_cache[name] = data
        return data

    def __getitem__(self, idx: int):
        row = self.index.iloc[int(idx)]
        shard = str(row["shard"])
        off = int(row["offset"])
        data = self._load_shard(shard)
        g = torch.from_numpy(data["g"][off]).to(torch.float32) / 255.0  # T,H,W
        g = g.unsqueeze(1)  # T,1,H,W
        d = g[1:] - g[:-1]
        d = torch.nn.functional.pad(d, (0, 0, 0, 0, 0, 0, 0, 1))
        x = torch.cat([g, d], dim=1).permute(1, 0, 2, 3).contiguous()  # C,T,H,W
        y = torch.tensor(float(data["labels"][off]), dtype=torch.float32)
        return x, y

