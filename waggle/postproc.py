from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    score: float

    @property
    def length_frames(self) -> int:
        return int(self.end_frame - self.start_frame + 1)


def to_segments(
    centers: np.ndarray,
    probs: np.ndarray,
    *,
    thr_on: float = 0.6,
    thr_off: float = 0.4,
    min_len_frames: int = 8,
    gap_frames: int = 8,
) -> list[Segment]:
    centers = np.asarray(centers, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    if centers.size == 0:
        return []
    order = np.argsort(centers)
    centers = centers[order]
    probs = probs[order]

    raw: list[Segment] = []
    on = False
    s = e = None
    scores: list[float] = []
    for c, p in zip(centers.tolist(), probs.tolist()):
        c = int(c)
        p = float(p)
        if not on:
            if p >= thr_on:
                on = True
                s = e = c
                scores = [p]
        else:
            if p <= thr_off:
                raw.append(Segment(int(s), int(e), float(np.mean(scores))))
                on = False
                s = e = None
                scores = []
            else:
                e = c
                scores.append(p)
    if on and s is not None and e is not None:
        raw.append(Segment(int(s), int(e), float(np.mean(scores))))

    if not raw:
        return []

    merged: list[Segment] = []
    cur = raw[0]
    for seg in raw[1:]:
        if seg.start_frame - cur.end_frame <= int(gap_frames):
            cur = Segment(cur.start_frame, seg.end_frame, float((cur.score + seg.score) / 2.0))
        else:
            merged.append(cur)
            cur = seg
    merged.append(cur)

    out = [s for s in merged if s.length_frames >= int(min_len_frames)]
    return out

