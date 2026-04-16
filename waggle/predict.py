from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from waggle.model import WaggleNet
from waggle.postproc import to_segments
from waggle.video_dataset import max_valid_center, min_valid_center, probe_total_frames, read_clip_at_center


def _clip_to_x(clip: torch.Tensor, *, out_hw: tuple[int, int] = (112, 112)) -> torch.Tensor:
    clip = clip.permute(0, 3, 1, 2).contiguous()  # T,C,H,W
    clip = torch.nn.functional.interpolate(clip, size=out_hw, mode="bilinear", align_corners=False)
    g = (0.2989 * clip[:, 0] + 0.5870 * clip[:, 1] + 0.1140 * clip[:, 2]).unsqueeze(1)
    d = g[1:] - g[:-1]
    d = torch.nn.functional.pad(d, (0, 0, 0, 0, 0, 0, 0, 1))
    x = torch.cat([g, d], dim=1).permute(1, 0, 2, 3).contiguous()
    return x


def predict_video(
    *,
    video: Path,
    ckpt: Path,
    out_csv: Path,
    fps: int = 30,
    clip_frames: int = 16,
    stride: int = 2,
    infer_step_frames: int = 8,
    thr_on: float = 0.6,
    thr_off: float = 0.4,
    min_len_frames: int = 8,
    gap_frames: int = 8,
    device: str | None = None,
) -> Path:
    video = video.expanduser().resolve()
    ckpt = ckpt.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=False, in_channels=2).to(dev)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    total = probe_total_frames(video, int(fps))
    cmin = min_valid_center(int(clip_frames), int(stride))
    cmax = max_valid_center(int(total), int(clip_frames), int(stride))
    step = max(1, int(infer_step_frames))

    centers: list[int] = []
    probs: list[float] = []
    with torch.no_grad():
        for c in range(int(cmin), int(cmax) + 1, step):
            clip = read_clip_at_center(
                video_path=str(video),
                center_frame=int(c),
                fps=int(fps),
                clip_frames=int(clip_frames),
                stride=int(stride),
            )
            x = _clip_to_x(clip).unsqueeze(0).to(dev)
            logit = model(x)[0].float().cpu()
            centers.append(int(c))
            probs.append(float(torch.sigmoid(logit)))

    centers_a = np.asarray(centers, dtype=np.int64)
    probs_a = np.asarray(probs, dtype=np.float32)
    segs = to_segments(
        centers_a,
        probs_a,
        thr_on=float(thr_on),
        thr_off=float(thr_off),
        min_len_frames=int(min_len_frames),
        gap_frames=int(gap_frames),
    )
    rows = []
    for i, s in enumerate(segs):
        rows.append(
            {
                "idx": i,
                "startFrame": int(s.start_frame),
                "endFrame": int(s.end_frame),
                "score": float(s.score),
                "duration_s": float((s.end_frame - s.start_frame + 1) / float(fps)),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

