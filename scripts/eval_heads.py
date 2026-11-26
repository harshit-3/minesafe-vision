"""Head-level helmet evaluation on the held-out test set.

For every ground-truth head (helmet / no_helmet box) find the best-overlapping predicted head box
and record the predicted label. Reports the confusion matrix and the dangerous error rate:
bare heads called "helmet" (a violation silently marked compliant).

    python scripts/eval_heads.py [--classifier weights/head_cls.onnx]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minesafe.detector import OnnxDetector  # noqa: E402
from minesafe.rules import CLASS_NAMES  # noqa: E402
from minesafe.tracker import iou  # noqa: E402

HEAD = {"helmet", "no_helmet"}


def gt_heads(label: Path, w: int, h: int):
    out = []
    for line in label.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        name = CLASS_NAMES[int(c)]
        if name in HEAD:
            out.append((name, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def evaluate(det: OnnxDetector, classifier=None, conf: float = 0.25, split: str = "test") -> dict:
    """split='test' -> official val images (our held-out test); split='val' -> our validation list."""
    ds = ROOT / "datasets" / "construction-ppe"
    if split == "test":
        files = sorted((ds / "images" / "val").glob("*"))
    else:
        files = [Path(x) for x in (ds / "val.txt").read_text().split()]
    cm = Counter()
    for p in files:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        dets = [d for d in det(img) if d.conf >= conf]
        if classifier:
            dets = classifier.refine(img, dets)
        preds = [d for d in dets if d.cls in HEAD]
        for name, box in gt_heads(ds / "labels" / p.parent.name / f"{p.stem}.txt", w, h):
            gx, gy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # match: IoU >= 0.3, or the GT head centre falls inside an (estimated) head box
            cands = [(iou(box, d.box), d) for d in preds
                     if iou(box, d.box) >= 0.3 or (d.box[0] <= gx <= d.box[2] and d.box[1] <= gy <= d.box[3])]
            pred = max(cands, key=lambda c: (c[0] > 0.5, c[1].conf))[1].cls if cands else "missed"
            cm[(name, pred)] += 1
    nh = sum(v for (g, _), v in cm.items() if g == "no_helmet")
    hh = sum(v for (g, _), v in cm.items() if g == "helmet")
    return {
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(cm.items())},
        "bare_head_called_helmet": round(cm[("no_helmet", "helmet")] / max(nh, 1), 3),
        "bare_head_detected_as_violation": round(cm[("no_helmet", "no_helmet")] / max(nh, 1), 3),
        "helmet_called_bare": round(cm[("helmet", "no_helmet")] / max(hh, 1), 3),
        "n_no_helmet": nh, "n_helmet": hh,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "ppe_s960.onnx"))
    ap.add_argument("--classifier", default=None)
    a = ap.parse_args()
    clf = None
    if a.classifier:
        from minesafe.head_classifier import HeadClassifier
        clf = HeadClassifier(a.classifier)
    res = evaluate(OnnxDetector(a.model, conf=0.05), clf)
    print(json.dumps(res, indent=1))
