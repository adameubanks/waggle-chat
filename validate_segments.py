from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Match pred waggle segments to GT; report det + duration MAE.")
    ap.add_argument("--gt_csv", type=Path, required=True)
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--iou_min", type=float, default=0.25)
    args = ap.parse_args()

    gt = pd.read_csv(args.gt_csv)
    pr = pd.read_csv(args.pred_csv)
    for c in ("startFrame", "endFrame", "duration"):
        if c not in gt.columns:
            raise SystemExit(f"gt missing {c}")
    for c in ("startFrame", "endFrame"):
        if c not in pr.columns:
            raise SystemExit(f"pred missing {c}")
    dur_pred_col = "duration_s" if "duration_s" in pr.columns else None

    gt_rows = [
        (int(r["startFrame"]), int(r["endFrame"]), float(r["duration"]))
        for _, r in gt.iterrows()
    ]
    pred_rows = [(int(r["startFrame"]), int(r["endFrame"])) for _, r in pr.iterrows()]

    matched_gt: set[int] = set()
    tp = fp = 0
    dur_err: list[float] = []

    for pi, (ps, pe) in enumerate(pred_rows):
        best_j = -1
        best_iou = 0.0
        for j, (gs, ge, _) in enumerate(gt_rows):
            iou = _iou_1d(ps, pe, gs, ge)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= args.iou_min and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
            if dur_pred_col is not None:
                gdur = gt_rows[best_j][2]
                pdur = float(pr.iloc[pi][dur_pred_col])
                dur_err.append(abs(pdur - gdur))
        else:
            fp += 1

    fn = len(gt_rows) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print(f"tp={tp} fp={fp} fn={fn} iou_min={args.iou_min}")
    print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")
    if dur_err:
        print(f"duration_mae_s={float(np.mean(dur_err)):.4f} (n={len(dur_err)} matched)")
    else:
        print("duration_mae_s=n/a (no duration_s column or no TP)")


if __name__ == "__main__":
    main()
