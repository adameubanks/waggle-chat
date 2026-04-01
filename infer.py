from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from waggle.model import WaggleNet
from waggle.video_dataset import max_valid_center, min_valid_center, probe_total_frames, read_clip_at_center


def _to_segments(centers: np.ndarray, probs: np.ndarray, *, thr_on: float, thr_off: float):
    segs = []
    on = False
    cur_start = None
    prev_c = None
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
                segs.append((cur_start, prev_c, float(np.mean(cur_scores))))
                on = False
                cur_start = None
                prev_c = None
                cur_scores = []
            else:
                prev_c = c
                cur_scores.append(p)
    if on and cur_start is not None and prev_c is not None:
        segs.append((cur_start, prev_c, float(np.mean(cur_scores))))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, default=Path("predicted_waggles.csv"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--infer_stride_seconds", type=float, default=0.5)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--thr_on", type=float, default=0.6)
    ap.add_argument("--thr_off", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaggleNet(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    total = probe_total_frames(args.video, args.fps)
    cmin = min_valid_center(args.clip_frames, args.stride)
    cmax = max_valid_center(total, args.clip_frames, args.stride)
    step = max(1, int(round(args.infer_stride_seconds * args.fps)))

    centers = []
    probs = []
    durs = []
    with torch.no_grad():
        for c in range(cmin, cmax + 1, step):
            try:
                clip = read_clip_at_center(
                    video_path=str(args.video),
                    center_frame=c,
                    fps=args.fps,
                    clip_frames=args.clip_frames,
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

            centers.append(c)
            probs.append(p)
            durs.append(dur_s)

    if not centers:
        raise SystemExit(f"No clips decoded from {args.video}")

    centers_a = np.asarray(centers, dtype=np.int64)
    probs_a = np.asarray(probs, dtype=np.float32)
    durs_a = np.asarray(durs, dtype=np.float32)

    segs = _to_segments(centers_a, probs_a, thr_on=args.thr_on, thr_off=args.thr_off)

    out_rows = []
    for i, (sf, ef, score) in enumerate(segs):
        in_seg = (centers_a >= sf) & (centers_a <= ef)
        dur_est = float(durs_a[in_seg].mean()) if in_seg.any() else float(durs_a[np.argmin(np.abs(centers_a - (sf + ef) // 2))])

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
