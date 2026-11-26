"""Safety rules: turn raw detections into per-worker compliance and violation events.

Pure Python/numpy so it is unit-testable and independent of the detector backend.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

CLASS_NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "person", "no_helmet",
               "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"]
VEHICLES = {"forklift"}

# PPE item -> (positive class, negative class, vertical band of the person box it must lie in)
PPE_RULES = {
    "helmet": ("helmet", "no_helmet", (0.0, 0.35)),
    "vest": ("vest", "no_vest", (0.15, 0.75)),
    "goggles": ("goggles", "no_goggle", (0.0, 0.35)),
    "gloves": ("gloves", "no_gloves", (0.2, 0.9)),
    "boots": ("boots", "no_boots", (0.7, 1.05)),
}


@dataclass
class Detection:
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    conf: float = 1.0
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


@dataclass
class WorkerStatus:
    person: Detection
    ppe: dict[str, str] = field(default_factory=dict)  # item -> "ok" | "missing" | "unknown"
    violations: list[str] = field(default_factory=list)
    is_operator: bool = False


@dataclass
class SiteConfig:
    required_ppe: tuple[str, ...] = ("helmet", "vest")
    vehicle_margin: float = 0.5  # exclusion zone = vehicle box expanded by this fraction of its size
    restricted_zones: dict[str, list[tuple[float, float]]] = field(default_factory=dict)  # name -> polygon
    min_conf: float = 0.35
    # Strict mode: PPE that can't be confirmed counts as missing (higher recall, more human review).
    unknown_is_violation: bool = False
    # A person whose box lies mostly inside a vehicle box is its operator: exempt from proximity.
    operator_overlap: float = 0.6


def _overlap_fraction(inner, outer) -> float:
    """Fraction of `inner` box area that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def _in_band(item: Detection, person: Detection, band: tuple[float, float]) -> bool:
    """Item centre lies inside the person box horizontally and inside a vertical band of it."""
    cx, cy = item.center
    x1, y1, x2, y2 = person.box
    h = y2 - y1
    pad = 0.1 * (x2 - x1)
    return (x1 - pad) <= cx <= (x2 + pad) and (y1 + band[0] * h) <= cy <= (y1 + band[1] * h)


def point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def expand_box(box, margin: float):
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    return x1 - dw, y1 - dh, x2 + dw, y2 + dh


def _assign(items: list[Detection], persons: list[Detection], band) -> dict[int, list[Detection]]:
    """Greedy one-to-one-ish assignment: each item goes to the smallest person box that contains
    it (the closest worker when boxes overlap in a crowd)."""
    out: dict[int, list[Detection]] = defaultdict(list)
    for it in items:
        cands = [(i, p) for i, p in enumerate(persons) if _in_band(it, p, band)]
        if cands:
            i, _ = min(cands, key=lambda ip: (ip[1].box[2] - ip[1].box[0]) * (ip[1].box[3] - ip[1].box[1]))
            out[i].append(it)
    return out


def evaluate_frame(dets: list[Detection], cfg: SiteConfig = SiteConfig()) -> list[WorkerStatus]:
    dets = [d for d in dets if d.conf >= cfg.min_conf]
    persons = [d for d in dets if d.cls == "person"]
    vehicles = [d for d in dets if d.cls in VEHICLES]
    workers = [WorkerStatus(p) for p in persons]

    for item in cfg.required_ppe:
        pos, neg, band = PPE_RULES[item]
        has = _assign([d for d in dets if d.cls == pos], persons, band)
        lacks = _assign([d for d in dets if d.cls == neg], persons, band)
        for i, w in enumerate(workers):
            p = max((d.conf for d in has.get(i, [])), default=0.0)
            n = max((d.conf for d in lacks.get(i, [])), default=0.0)
            if n > p:
                w.ppe[item] = "missing"
                w.violations.append(f"no_{item}")
            elif p > 0:
                w.ppe[item] = "ok"
            else:
                w.ppe[item] = "unknown"  # occluded / too small: don't accuse, don't clear
                if cfg.unknown_is_violation:
                    w.violations.append(f"no_{item}")

    for w in workers:
        fx, fy = w.person.foot
        for v in vehicles:
            if _overlap_fraction(w.person.box, v.box) >= cfg.operator_overlap:
                w.is_operator = True  # driving the vehicle, not a pedestrian at risk
                continue
            x1, y1, x2, y2 = expand_box(v.box, cfg.vehicle_margin)
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                w.violations.append("vehicle_proximity")
                break
        for name, poly in cfg.restricted_zones.items():
            if point_in_polygon((fx, fy), poly):
                w.violations.append(f"restricted_zone:{name}")
    return workers


@dataclass
class Event:
    frame: int
    track_id: int
    violation: str
    box: tuple[float, float, float, float]


class ViolationDebouncer:
    """Emit an event only when a tracked worker shows the same violation in >= `min_hits` of the
    last `window` frames, then stay silent for `cooldown` frames. Single-frame detector flicker
    never reaches a supervisor's phone."""

    def __init__(self, window: int = 15, min_hits: int = 10, cooldown: int = 300):
        self.window, self.min_hits, self.cooldown = window, min_hits, cooldown
        self.history: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_fired: dict[tuple[int, str], int] = {}

    def update(self, frame: int, workers: list[WorkerStatus]) -> list[Event]:
        events = []
        seen: set[tuple[int, str]] = set()
        for w in workers:
            if w.person.track_id is None:
                continue
            for v in w.violations:
                key = (w.person.track_id, v)
                seen.add(key)
                self.history[key].append(1)
                hits = sum(self.history[key])
                last = self.last_fired.get(key)
                if hits >= self.min_hits and (last is None or frame - last >= self.cooldown):
                    self.last_fired[key] = frame
                    events.append(Event(frame, w.person.track_id, v, w.person.box))
        for key, h in self.history.items():
            if key not in seen:
                h.append(0)
        return events


def ground_truth_violations(dets: list[Detection], cfg: SiteConfig = SiteConfig()) -> list[set[str]]:
    """Apply the same rules to ground-truth boxes -> per-person violation sets (for evaluation)."""
    return [set(w.violations) for w in evaluate_frame(dets, cfg)]


def match_persons(gt: list[Detection], pred: list[Detection], iou_thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy IoU matching of person boxes (gt index, pred index)."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0.0

    pairs = sorted(((iou(g.box, p.box), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for s, i, j in pairs:
        if s < iou_thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def as_array(workers: list[WorkerStatus]) -> np.ndarray:
    return np.array([w.person.box for w in workers], dtype=float).reshape(-1, 4)
