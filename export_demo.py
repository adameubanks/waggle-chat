from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from waggle.demo import build_demo_run, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_id", type=str, default="demo")
    ap.add_argument("--title", type=str, default="Waggle demo")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--infer_stride_seconds", type=float, default=0.5)
    ap.add_argument("--clip_frames", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--thr_on", type=float, default=0.6)
    ap.add_argument("--thr_off", type=float, default=0.4)
    ap.add_argument("--meters_per_second", type=float, default=300.0)
    ap.add_argument("--angle_offset_deg", type=float, default=0.0)
    ap.add_argument("--copy_video", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = out_dir / "data"
    clips_dir = data_dir / "clips"
    runs_dir = data_dir / "runs"
    clips_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    video_src = args.video.expanduser().resolve()
    video_rel = Path("data/clips") / f"{args.clip_id}{video_src.suffix}"
    if args.copy_video:
        shutil.copy2(video_src, out_dir / video_rel)

    run_rel = Path("data/runs") / f"{args.clip_id}.json"
    build_demo_run(
        video=video_src,
        ckpt=args.ckpt.expanduser().resolve(),
        out_json=out_dir / run_rel,
        fps=args.fps,
        infer_stride_seconds=args.infer_stride_seconds,
        clip_frames=args.clip_frames,
        stride=args.stride,
        thr_on=args.thr_on,
        thr_off=args.thr_off,
        meters_per_second=args.meters_per_second,
        angle_offset_deg=args.angle_offset_deg,
        device=args.device,
    )

    write_manifest(
        out_path=out_dir / "data" / "manifest.json",
        clips=[
            {
                "id": args.clip_id,
                "title": args.title,
                "videoUrl": str(video_rel).replace("\\", "/"),
                "resultUrl": str(run_rel).replace("\\", "/"),
            }
        ],
    )

    print(f"Wrote {run_rel} and data/manifest.json in {out_dir}")
    if args.copy_video:
        print(f"Copied video to {video_rel}")
    else:
        print("Video not copied; place a web-playable file at the manifest's videoUrl.")


if __name__ == "__main__":
    main()

