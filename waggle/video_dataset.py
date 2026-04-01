from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\._video_deprecation_warning")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\.io\.video_reader")

from torchvision.io import VideoReader


def middle_tensor_index(clip_frames: int) -> int:
    return (clip_frames - 1) // 2


def decode_start_for_center(center_frame: int, clip_frames: int, stride: int) -> int:
    mid = middle_tensor_index(clip_frames)
    return max(0, int(center_frame) - mid * int(stride))


def min_valid_center(clip_frames: int, stride: int) -> int:
    return middle_tensor_index(clip_frames) * int(stride)


def max_valid_center(total_frames: int, clip_frames: int, stride: int) -> int:
    mid = middle_tensor_index(clip_frames)
    return int(total_frames) - 1 - (clip_frames - 1 - mid) * int(stride)


def probe_total_frames(video_path: Path, fps: int) -> int:
    vr = VideoReader(str(video_path), "video")
    md = vr.get_metadata()
    dur_s = None
    try:
        dur_s = float(md["video"]["duration"][0])
    except Exception:
        dur_s = None
    if dur_s is not None and math.isfinite(dur_s) and dur_s > 0:
        return max(1, int(round(dur_s * fps)))
    n = 0
    for _ in vr:
        n += 1
    return max(1, n)


def _read_clip(
    *,
    video_path: str,
    start_frame: int,
    fps: int,
    frames: int,
    stride: int,
) -> torch.Tensor:
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

    clip = torch.stack(out, dim=0)
    if clip.shape[0] < frames:
        pad = clip[-1:].repeat(frames - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)

    clip = clip.to(torch.float32) / 255.0
    return clip


def read_clip_at_center(
    *,
    video_path: str,
    center_frame: int,
    fps: int,
    clip_frames: int,
    stride: int,
) -> torch.Tensor:
    start = decode_start_for_center(center_frame, clip_frames, stride)
    return _read_clip(
        video_path=video_path,
        start_frame=start,
        fps=fps,
        frames=clip_frames,
        stride=stride,
    )


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
        self._from_pt = "clip_pt" in self.df.columns

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        if self._from_pt:
            clip = torch.load(str(row["clip_pt"]), map_location="cpu", weights_only=False)
        else:
            clip = read_clip_at_center(
                video_path=str(row["video"]),
                center_frame=int(row["center_frame"]),
                fps=self.fps,
                clip_frames=self.clip_frames,
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
