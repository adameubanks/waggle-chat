from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from waggle.model import WaggleNet
from waggle.video_dataset import _read_clip


def _to_segments(starts: np.ndarray, probs: np.ndarray, *, window_frames: int, thr_on: float, thr_off: float):
    segs = []
    on = False
    cur_start = None
    cur_scores = []
    for s, p in zip(starts.tolist(), probs.tolist()):
        if not on and p >= thr_on:
            on = True
            cur_start = int(s)
            cur_scores = [float(p)]
        elif on:
            cur_scores.append(float(p))
            if p <= thr_off:
                end = int(s + window_frames - 1)
                segs.append((cur_start, end, float(np.mean(cur_scores))))
                on = False
                cur_start = None
                cur_scores = []
    if on and cur_start is not None:
        end = int(starts[-1] + window_frames - 1)
        segs.append((cur_start, end, float(np.mean(cur_scores))))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, default=Path("predicted_waggles.csv"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window_seconds", type=float, default=3.0)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--thr_on", type=float, default=0.6)
    ap.add_argument("--thr_off", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    window_frames = int(round(args.window_seconds * args.fps))
    with torch.no_grad():
        starts = []
        probs = []
        durs = []
        w = 0
        while True:
            start_f = w * window_frames
            try:
                clip = _read_clip(
                    video_path=str(args.video),
                    start_frame=start_f,
                    fps=args.fps,
                    frames=args.clip_frames,
                    stride=args.stride,
                )
            except Exception:
                break

            clip = torch.nn.functional.interpolate(clip, size=(112, 112), mode="bilinear", align_corners=False)
            out = model(clip.unsqueeze(0).to(device))
            logit = out[0, 0].float().cpu()
            dlog = out[0, 1].float().cpu()

            p = float(torch.sigmoid(logit))
            dur_s = float(torch.expm1(torch.clamp(dlog, max=12.0)).cpu())

            starts.append(start_f)
            probs.append(p)
            durs.append(dur_s)
            w += 1

    if not starts:
        raise SystemExit(f"No windows decoded from {args.video}")

    starts_a = np.asarray(starts, dtype=np.int64)
    probs_a = np.asarray(probs, dtype=np.float32)
    durs_a = np.asarray(durs, dtype=np.float32)

    segs = _to_segments(starts_a, probs_a, window_frames=window_frames, thr_on=args.thr_on, thr_off=args.thr_off)

    out_rows = []
    for i, (sf, ef, score) in enumerate(segs):
        mid = (sf + ef) // 2
        nearest = int(np.argmin(np.abs(starts_a - mid)))
        in_seg = (starts_a >= sf) & (starts_a <= ef)
        dur_est = float(durs_a[in_seg].mean()) if in_seg.any() else float(durs_a[nearest])

        out_rows.append(
            {
                "idx": i,
                "startFrame": int(sf),
                "endFrame": int(ef),
                "score": float(score),
                "duration_s": dur_est,
            }
        )

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_rows)} segments to {args.out_csv}")


if __name__ == "__main__":
    main()
