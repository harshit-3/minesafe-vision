"""Second-stage helmet classifier on high-resolution head crops (ONNX Runtime, no torch).

The detector sees a frame downscaled to 960 px, where a distant worker's head is ~25 px and a
beanie or bald head seen from behind looks like a hard hat. This stage crops each worker's head
from the *full-resolution* frame and asks a dedicated MobileNetV3 classifier: helmet or not?

It also creates head boxes for workers the detector found but whose head it did not label,
using the top of the person box, so "unknown" workers get a decision too.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .rules import PPE_RULES, Detection, _in_band

INPUT = 96
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HEAD_CLASSES = {"helmet", "no_helmet"}


def crop_head(img: np.ndarray, box, pad: float = 0.25) -> np.ndarray:
    """Square crop around a head box with context padding, resized to INPUT x INPUT (BGR)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = max(side, 8)
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    c, d = a + int(round(side)), b + int(round(side))
    canvas = np.full((d - b, c - a, 3), 114, dtype=np.uint8)
    sa, sb, sc, sd = max(a, 0), max(b, 0), min(c, w), min(d, h)
    if sc > sa and sd > sb:
        canvas[sb - b:sd - b, sa - a:sc - a] = img[sb:sd, sa:sc]
    return cv2.resize(canvas, (INPUT, INPUT), interpolation=cv2.INTER_AREA if side > INPUT else cv2.INTER_LINEAR)


def head_from_person(person) -> tuple[float, float, float, float]:
    """Estimate the head box from a person box (upright worker: head ~ top 18% of height)."""
    x1, y1, x2, y2 = person
    pw, ph = x2 - x1, y2 - y1
    side = min(0.6 * pw, 0.2 * ph) if ph > pw else 0.45 * min(pw, ph)
    cx = (x1 + x2) / 2
    return cx - side / 2, y1, cx + side / 2, y1 + side


class HeadClassifier:
    def __init__(self, path: str | Path, providers: list[str] | None = None,
                 helmet_thr: float = 0.95, bare_thr: float = 0.5, min_head_px: float = 10):
        # helmet_thr / bare_thr / min_head_px are chosen on the validation split (scripts/eval_heads.py)
        self.session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.helmet_thr, self.bare_thr, self.min_head_px = helmet_thr, bare_thr, min_head_px

    def predict(self, img: np.ndarray, boxes: list) -> np.ndarray:
        """P(helmet) for each head box."""
        if not boxes:
            return np.zeros(0, dtype=np.float32)
        x = np.stack([crop_head(img, b)[:, :, ::-1] for b in boxes]).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(0, 3, 1, 2).astype(np.float32)
        logits = self.session.run(None, {"image": x})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True))[:, 1]

    def refine(self, img: np.ndarray, dets: list[Detection]) -> list[Detection]:
        """Replace detector helmet/no_helmet boxes with classifier decisions, and add a head decision
        for every person that has none. Asymmetric thresholds: calling a head 'helmet' needs strong
        evidence; uncertain heads stay unlabelled (-> 'unknown' in the rules engine)."""
        persons = [d for d in dets if d.cls == "person"]
        heads = [d for d in dets if d.cls in HEAD_CLASSES]
        others = [d for d in dets if d.cls not in HEAD_CLASSES]
        band = PPE_RULES["helmet"][2]

        def big_enough(box) -> bool:  # a head a few pixels wide carries no evidence either way
            return max(box[2] - box[0], box[3] - box[1]) >= self.min_head_px

        small = [h for h in heads if not big_enough(h.box)]  # keep the detector's own call
        boxes = [h.box for h in heads if big_enough(h.box)]
        for p in persons:  # workers with no head detection get an estimated head region
            if not any(_in_band(h, p, band) for h in heads):
                est = head_from_person(p.box)
                if big_enough(est):
                    boxes.append(est)
        probs = self.predict(img, boxes)
        refined = list(small)
        for box, ph in zip(boxes, probs, strict=True):
            if ph >= self.helmet_thr:
                refined.append(Detection("helmet", box, float(ph)))
            elif ph <= self.bare_thr:
                refined.append(Detection("no_helmet", box, float(1 - ph)))
        return others + refined
