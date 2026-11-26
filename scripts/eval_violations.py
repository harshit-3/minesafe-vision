"""Worker-level violation evaluation on the held-out test set.

mAP tells you how well boxes are drawn. A safety team cares whether *the right workers get
flagged*. We apply the same rules engine to ground-truth boxes and to model predictions,
match workers by IoU, and score each violation type:

  TP = worker truly violating and flagged      FN = violating worker missed
  FP = compliant/unknown worker flagged (false accusation, or a ghost person)

    python scripts/eval_violations.py --model weights/ppe_s960.onnx
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
from minesafe.rules import CLASS_NAMES, Detection, SiteConfig, evaluate_frame, match_persons  # noqa: E402

TYPES = ["no_helmet", "no_vest"]


def load_gt(label_path: Path, w: int, h: int) -> list[Detection]:
    dets = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = map(float, line.split()[:5])
        dets.append(Detection(CLASS_NAMES[int(c)], ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                                    (cx + bw / 2) * w, (cy + bh / 2) * h), 1.0))
    return dets


def evaluate(model_path: str, conf: float, gt_cfg: SiteConfig, pred_cfg: SiteConfig) -> dict:
    det = OnnxDetector(model_path, conf=conf)
    img_dir = ROOT / "datasets" / "construction-ppe" / "images" / "val"  # official val == our TEST
    counts = {t: Counter() for t in TYPES}
    persons = Counter()
    for img_path in sorted(img_dir.glob("*")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_w = evaluate_frame(load_gt(img_path.parent.parent.parent / "labels" / "val" / f"{img_path.stem}.txt",
                                      w, h), gt_cfg)
        pr_w = evaluate_frame(det(img), pred_cfg)
        pairs = match_persons([g.person for g in gt_w], [p.person for p in pr_w])
        persons["gt"] += len(gt_w)
        persons["matched"] += len(pairs)
        mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
        for t in TYPES:
            for gi, pj in pairs:
                g, p = t in gt_w[gi].violations, t in pr_w[pj].violations
                counts[t]["tp"] += g and p
                counts[t]["fn"] += g and not p
                counts[t]["fp"] += p and not g
            counts[t]["fn"] += sum(t in gt_w[i].violations for i in range(len(gt_w)) if i not in mg)
            counts[t]["fp"] += sum(t in pr_w[j].violations for j in range(len(pr_w)) if j not in mp)
    res = {"conf": conf, "worker_recall": round(persons["matched"] / max(persons["gt"], 1), 3)}
    for t, c in counts.items():
        p = c["tp"] / max(c["tp"] + c["fp"], 1)
        r = c["tp"] / max(c["tp"] + c["fn"], 1)
        res[t] = {"precision": round(p, 3), "recall": round(r, 3),
                  "f1": round(2 * p * r / max(p + r, 1e-9), 3), "support": c["tp"] + c["fn"], **c}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "ppe_s960.onnx"))
    a = ap.parse_args()
    # Confidence filtering happens in the detector; ground truth always uses the lenient policy so
    # "truth" means a labelled no_helmet/no_vest. Strict mode is scored against the same truth.
    results = []
    for mode, strict in (("lenient", False), ("strict", True)):
        for c in (0.25, 0.35, 0.5):
            results.append({"policy": mode, **evaluate(a.model, c, SiteConfig(min_conf=0.0),
                                                       SiteConfig(min_conf=0.0, unknown_is_violation=strict))})
    (ROOT / "runs" / f"violations_{Path(a.model).stem}.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(json.dumps(r))
