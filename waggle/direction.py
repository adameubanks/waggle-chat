from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DirectionResult:
    angle_rad: float
    strength: float


def estimate_direction_optflow(
    *,
    video: Path,
    start_frame: int,
    end_frame: int,
    roi: tuple[int, int, int, int] | None = None,
) -> DirectionResult:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"direction requires opencv-python (cv2). Import failed: {e}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open {video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"Failed to read frame {start_frame} from {video}")

    if roi is not None:
        x, y, w, h = roi
        prev = prev[y : y + h, x : x + w]
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    vs: list[tuple[float, float]] = []
    for _ in range(int(end_frame - start_frame)):
        ok, frame = cap.read()
        if not ok:
            break
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0].reshape(-1)
        fy = flow[..., 1].reshape(-1)
        mag = np.sqrt(fx * fx + fy * fy)
        keep = mag > np.quantile(mag, 0.90)
        if np.any(keep):
            vs.extend(list(zip(fx[keep].tolist(), fy[keep].tolist())))
        prev_g = g

    cap.release()

    if len(vs) < 20:
        return DirectionResult(angle_rad=0.0, strength=0.0)

    v = np.asarray(vs, dtype=np.float32)
    v -= v.mean(axis=0, keepdims=True)
    cov = (v.T @ v) / max(1.0, float(len(v) - 1))
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    angle = float(np.arctan2(axis[1], axis[0]))
    strength = float(np.max(vals) / (np.sum(vals) + 1e-12))
    return DirectionResult(angle_rad=angle, strength=strength)

