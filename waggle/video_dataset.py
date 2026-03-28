from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.video_reader")

from torchvision.io import VideoReader


def _read_clip(
    *,
    video_path: str,
    start_frame: int,
    fps: int,
    frames: int,
    stride: int,
) -> torch.Tensor:
    # VideoReader timestamps are in seconds; we seek and then step forward.
    vr = VideoReader(video_path, "video")
    vr.set_current_stream("video")
    vr.seek(start_frame / fps)

    out: list[torch.Tensor] = []
    need = frames
    step = stride
    i = 0
    for frame in vr:
        if i % step == 0:
            out.append(frame["data"])
            need -= 1
            if need == 0:
                break
        i += 1

    if not out:
        raise RuntimeError(f"Failed to decode clip from {video_path} @ frame {start_frame}")

    clip = torch.stack(out, dim=0)  # T,C,H,W uint8
    if clip.shape[0] < frames:
        pad = clip[-1:].repeat(frames - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)

    clip = clip.to(torch.float32) / 255.0
    return clip


class WaggleWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        *,
        fps: int = 30,
        clip_frames: int = 32,
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
        row = self.df.iloc[int(idx)]
        clip = _read_clip(
            video_path=str(row["video"]),
            start_frame=int(row["start_frame"]),
            fps=self.fps,
            frames=self.clip_frames,
            stride=self.stride,
        )
        clip = torch.nn.functional.interpolate(
            clip,
            size=self.resize_hw,
            mode="bilinear",
            align_corners=False,
        )

        y_cls = torch.tensor(float(row["is_waggle"]), dtype=torch.float32)
        y_dur = torch.tensor(float(row["duration_s"]), dtype=torch.float32)
        return clip, y_cls, y_dur


def compute_pos_weight(manifest_csv: Path) -> float:
    df = pd.read_csv(manifest_csv)
    pos = float(df["is_waggle"].sum())
    neg = float(len(df) - pos)
    if pos == 0:
        return 1.0
    return neg / pos

