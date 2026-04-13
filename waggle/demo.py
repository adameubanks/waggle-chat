from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from waggle.model import WaggleNet
from waggle.video_dataset import (
    max_valid_center,
    min_valid_center,
    probe_total_frames,
    read_clip_at_center,
)


@dataclass(frozen=True)
class DemoEvent:
    id: int
    start_s: float
    end_s: float
    confidence: float
    duration_s: float
    orientation_deg: float | None
    bearing_deg: float | None
    distance_m: float | None


@dataclass(frozen=True)
class DemoRun:
    schema_version: int
    video: str
    fps: int
    clip_frames: int
    stride: int
    infer_stride_seconds: float
    thr_on: float
    thr_off: float
    meters_per_second: float
    angle_offset_deg: float
    events: list[DemoEvent]


def _to_segments(centers: np.ndarray, probs: np.ndarray, *, thr_on: float, thr_off: float) -> list[tuple[int, int, float]]:
    segs: list[tuple[int, int, float]] = []
    on = False
    cur_start: int | None = None
    prev_c: int | None = None
    cur_scores: list[float] = []
    for c, p in zip(centers.tolist(), probs.tolist()):
        c = int(c)
        p = float(p)
        if not on and p >= thr_on:
            on = True
            cur_start = c
            prev_c = c
            cur_scores = [p]
        elif on:
            if p <= thr_off:
                segs.append((int(cur_start), int(prev_c), float(np.mean(cur_scores))))  # type: ignore[arg-type]
                on = False
                cur_start = None
                prev_c = None
                cur_scores = []
            else:
                prev_c = c
                cur_scores.append(p)
    if on and cur_start is not None and prev_c is not None:
        segs.append((int(cur_start), int(prev_c), float(np.mean(cur_scores))))
    return segs


def _orientation_from_clip(clip: torch.Tensor) -> float | None:
    if clip.ndim != 4 or clip.shape[0] < 2:
        return None
    x = clip.to(torch.float32)
    if x.max() > 1.5:
        x = x / 255.0
    g = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).cpu().numpy()
    m = np.abs(np.diff(g, axis=0)).mean(axis=0)
    if not np.isfinite(m).all():
        return None
    m = m - float(m.mean())
    gy, gx = np.gradient(m)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    if not (math.isfinite(jxx) and math.isfinite(jyy) and math.isfinite(jxy)):
        return None
    if (jxx + jyy) <= 1e-12:
        return None
    theta = 0.5 * math.atan2(2.0 * jxy, (jxx - jyy))
    deg = (math.degrees(theta) + 180.0) % 180.0
    return float(deg)


def _decode_vector(*, orientation_deg: float | None, duration_s: float, meters_per_second: float, angle_offset_deg: float) -> tuple[float | None, float | None]:
    if orientation_deg is None:
        return None, None
    bearing = (float(orientation_deg) + float(angle_offset_deg)) % 360.0
    dist = max(0.0, float(duration_s)) * float(meters_per_second)
    return float(bearing), float(dist)


def build_demo_run(
    *,
    video: Path,
    ckpt: Path,
    out_json: Path,
    fps: int = 30,
    infer_stride_seconds: float = 0.5,
    clip_frames: int = 32,
    stride: int = 2,
    thr_on: float = 0.6,
    thr_off: float = 0.4,
    meters_per_second: float = 300.0,
    angle_offset_deg: float = 0.0,
    device: str | None = None,
) -> DemoRun:
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=False).to(dev)
    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    total = probe_total_frames(video, fps)
    cmin = min_valid_center(clip_frames, stride)
    cmax = max_valid_center(total, clip_frames, stride)
    step = max(1, int(round(float(infer_stride_seconds) * int(fps))))

    centers: list[int] = []
    probs: list[float] = []
    durs: list[float] = []
    with torch.no_grad():
        for c in range(int(cmin), int(cmax) + 1, int(step)):
            try:
                clip = read_clip_at_center(
                    video_path=str(video),
                    center_frame=int(c),
                    fps=int(fps),
                    clip_frames=int(clip_frames),
                    stride=int(stride),
                )
            except Exception:
                break

            clip_small = torch.nn.functional.interpolate(clip, size=(112, 112), mode="bilinear", align_corners=False)
            out = model(clip_small.unsqueeze(0).to(dev))
            logit = out[0, 0].float().cpu()
            dlog = out[0, 1].float().cpu()
            p = float(torch.sigmoid(logit))
            dur_s = float(torch.expm1(torch.clamp(dlog, max=12.0)).cpu())

            centers.append(int(c))
            probs.append(p)
            durs.append(dur_s)

    if not centers:
        raise SystemExit(f"No clips decoded from {video}")

    centers_a = np.asarray(centers, dtype=np.int64)
    probs_a = np.asarray(probs, dtype=np.float32)
    durs_a = np.asarray(durs, dtype=np.float32)
    segs = _to_segments(centers_a, probs_a, thr_on=float(thr_on), thr_off=float(thr_off))

    events: list[DemoEvent] = []
    for i, (sf, ef, score) in enumerate(segs):
        in_seg = (centers_a >= sf) & (centers_a <= ef)
        if in_seg.any():
            dur_est = float(durs_a[in_seg].mean())
            mid = int(centers_a[in_seg][len(centers_a[in_seg]) // 2])
        else:
            mid = int((sf + ef) // 2)
            dur_est = float(durs_a[np.argmin(np.abs(centers_a - mid))])

        try:
            clip_mid = read_clip_at_center(
                video_path=str(video),
                center_frame=int(mid),
                fps=int(fps),
                clip_frames=int(clip_frames),
                stride=int(stride),
            )
            orientation_deg = _orientation_from_clip(clip_mid)
        except Exception:
            orientation_deg = None

        bearing_deg, distance_m = _decode_vector(
            orientation_deg=orientation_deg,
            duration_s=dur_est,
            meters_per_second=float(meters_per_second),
            angle_offset_deg=float(angle_offset_deg),
        )

        events.append(
            DemoEvent(
                id=int(i),
                start_s=float(sf) / float(fps),
                end_s=float(ef) / float(fps),
                confidence=float(score),
                duration_s=float(dur_est),
                orientation_deg=orientation_deg,
                bearing_deg=bearing_deg,
                distance_m=distance_m,
            )
        )

    run = DemoRun(
        schema_version=1,
        video=str(video),
        fps=int(fps),
        clip_frames=int(clip_frames),
        stride=int(stride),
        infer_stride_seconds=float(infer_stride_seconds),
        thr_on=float(thr_on),
        thr_off=float(thr_off),
        meters_per_second=float(meters_per_second),
        angle_offset_deg=float(angle_offset_deg),
        events=events,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
    return run


def write_manifest(*, out_path: Path, clips: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"schema_version": 1, "clips": clips}, indent=2), encoding="utf-8")

