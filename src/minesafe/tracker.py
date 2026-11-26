"""Minimal IoU tracker for worker identities across video frames.

Enough for debouncing violations per worker: greedy IoU association with a short
memory so brief occlusions don't spawn new IDs. (ByteTrack/BoT-SORT would be the
next step for crowded scenes.)
"""
from __future__ import annotations

from dataclasses import dataclass

from .rules import Detection


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    id: int
    box: tuple
    missed: int = 0


class IoUTracker:
    def __init__(self, iou_thr: float = 0.3, max_missed: int = 15, cls: str = "person"):
        self.iou_thr, self.max_missed, self.cls = iou_thr, max_missed, cls
        self.tracks: list[_Track] = []
        self._next = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        objs = [d for d in dets if d.cls == self.cls]
        pairs = sorted(((iou(t.box, d.box), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(objs)), reverse=True)
        used_t, used_d = set(), set()
        for s, ti, di in pairs:
            if s < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.box, t.missed = objs[di].box, 0
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
