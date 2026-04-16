from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision\..*")


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
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"video IO requires opencv-python (cv2). Import failed: {e}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if n > 0:
        return n
    return 1


def _read_clip(
    *,
    video_path: str,
    start_frame: int,
    fps: int,
    frames: int,
    stride: int,
) -> torch.Tensor:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"video IO requires opencv-python (cv2). Import failed: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    out: list[torch.Tensor] = []
    need = int(frames)
    step = max(1, int(stride))
    i = 0
    while need > 0:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(torch.from_numpy(frame))
            need -= 1
        i += 1
    cap.release()

    if not out:
        raise RuntimeError(f"Failed to decode clip from {video_path} @ frame {start_frame}")

    clip = torch.stack(out, dim=0)  # T,H,W,C uint8
    if clip.shape[0] < frames:
        pad = clip[-1:].repeat(frames - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)
    return clip.to(torch.float32) / 255.0


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

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
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
