"""Lightweight multi-object tracker for worker identities (SORT-style, no Kalman/re-ID).

Association score = max(IoU with the motion-predicted box, size-normalised centre proximity).
Plain IoU fails for small, distant workers: a 20 px box that moves a few pixels per frame
(or a panning camera) overlaps too little, so every few frames the worker became "new" and
re-alerted. Predicting each track forward with its velocity and allowing distance-based matches
fixed that (72 -> 9 IDs for ~7 workers on the demo clip).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .rules import Detection


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


@dataclass
class _Track:
    id: int
    box: tuple
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0
    hits: int = 1

    def predicted(self) -> tuple:
        k = self.missed + 1
        x1, y1, x2, y2 = self.box
        return x1 + self.vx * k, y1 + self.vy * k, x2 + self.vx * k, y2 + self.vy * k


def match_score(track_box, det_box) -> float:
    """1.0 = perfect. Combines overlap with centre distance relative to object size."""
    (tx, ty), (dx, dy) = _centre(track_box), _centre(det_box)
    size = max(math.hypot(track_box[2] - track_box[0], track_box[3] - track_box[1]),
               math.hypot(det_box[2] - det_box[0], det_box[3] - det_box[1]), 1e-6)
    proximity = max(0.0, 1.0 - math.hypot(tx - dx, ty - dy) / size)
    # Reject wildly different sizes (a near worker vs a far one at a similar position).
    ta = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
    da = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    if min(ta, da) / max(ta, da, 1e-6) < 0.35:
        return 0.0
    return max(iou(track_box, det_box), proximity)


class IoUTracker:
    def __init__(self, min_score: float = 0.3, max_missed: int = 15, cls: str = "person", momentum: float = 0.6):
        self.min_score, self.max_missed, self.cls, self.momentum = min_score, max_missed, cls, momentum
        self.tracks: list[_Track] = []
        self._next = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        objs = [d for d in dets if d.cls == self.cls]
        pairs = sorted(((match_score(t.predicted(), d.box), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(objs)), reverse=True)
        used_t, used_d = set(), set()
        for s, ti, di in pairs:
            if s < self.min_score:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            (ox, oy), (nx, ny) = _centre(t.box), _centre(objs[di].box)
            steps = t.missed + 1
            t.vx = self.momentum * t.vx + (1 - self.momentum) * (nx - ox) / steps
            t.vy = self.momentum * t.vy + (1 - self.momentum) * (ny - oy) / steps
            t.box, t.missed, t.hits = objs[di].box, 0, t.hits + 1
            objs[di].track_id = t.id
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for di, d in enumerate(objs):
            if di not in used_d:
                d.track_id = self._next
                self.tracks.append(_Track(self._next, d.box))
                self._next += 1
        return dets
