from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_segments(csv_path: Path) -> list[tuple[int, int, float]]:
    df = pd.read_csv(csv_path)
    for c in ("startFrame", "endFrame"):
        if c not in df.columns:
            raise SystemExit(f"{csv_path} missing column {c}")
    score_col = "score" if "score" in df.columns else None
    segs = []
    for _, r in df.iterrows():
        s = int(r["startFrame"])
        e = int(r["endFrame"])
        sc = float(r[score_col]) if score_col is not None else 1.0
        segs.append((s, e, sc))
    segs.sort(key=lambda x: x[0])
    return segs


def _active_score(segs: list[tuple[int, int, float]], frame_idx: int) -> float | None:
    for s, e, sc in segs:
        if s <= frame_idx <= e:
            return sc
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay predicted waggle segments onto a video.")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--segments_csv", type=Path, required=True)
    ap.add_argument("--out_video", type=Path, default=Path("overlay.mp4"))
    ap.add_argument("--fps", type=float, default=None, help="Override FPS for writing (defaults to source).")
    args = ap.parse_args()

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"overlay.py requires opencv-python (cv2). Import failed: {e}")

    segs = _load_segments(args.segments_csv)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    fps = float(args.fps) if args.fps is not None else src_fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.out_video), fourcc, fps, (w, h))
    if not out.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {args.out_video}")

    bar_h = max(18, h // 40)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sc = _active_score(segs, i)
        if sc is not None:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 255), thickness=-1)
            alpha = 0.35 + 0.45 * min(1.0, max(0.0, float(sc)))
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            txt = f"WAGGLE score={sc:.2f}"
            cv2.putText(frame, txt, (10, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "no waggle", (10, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        t_s = i / fps
        cv2.putText(frame, f"t={t_s:7.2f}s frame={i}" + (f"/{n_frames}" if n_frames > 0 else ""), (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
        i += 1

    cap.release()
    out.release()
    print(f"Wrote overlay video to {args.out_video} ({i} frames)")


if __name__ == "__main__":
    main()

