from pathlib import Path

import cv2
import numpy as np
import pytest

from minesafe.head_classifier import INPUT, crop_head, head_from_person
from minesafe.rules import Detection, evaluate_frame

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "head_cls.onnx"
DET = Path(__file__).resolve().parents[1] / "weights" / "ppe_s960.onnx"
FIX = Path(__file__).parent / "fixtures"


def test_crop_head_pads_out_of_frame_boxes():
    img = np.full((100, 100, 3), 200, np.uint8)
    c = crop_head(img, (-20, -20, 10, 10))
    assert c.shape == (INPUT, INPUT, 3)
    assert (c == 114).any() and (c == 200).any()


def test_head_from_person_is_at_top_centre():
    x1, y1, x2, y2 = head_from_person((100, 100, 200, 400))
    assert y1 == 100 and 100 < x1 < x2 < 200 and y2 < 200


@pytest.mark.skipif(not (WEIGHTS.exists() and DET.exists()), reason="weights not present")
def test_regression_bare_heads_seen_from_behind():
    """Regression for a user-reported bug: 4 bare/beanie heads seen from behind were all marked
    'compliant' by the detector alone. With the head classifier, at least 3 must be flagged."""
    from minesafe.pipeline import load_models

    img = cv2.imread(str(FIX / "bare_heads_from_behind.jpg"))
    det, head = load_models(DET, WEIGHTS)
    flagged_without = sum("no_helmet" in w.violations for w in evaluate_frame(det(img)))
    flagged_with = sum("no_helmet" in w.violations for w in evaluate_frame(head.refine(img, det(img))))
    assert flagged_without == 0  # documents the original failure
    assert flagged_with >= 3


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not present")
def test_refine_adds_head_decision_for_person_without_head_box():
    from minesafe.head_classifier import HeadClassifier

    img = cv2.imread(str(FIX / "bare_heads_from_behind.jpg"))
    person = Detection("person", (873, 492, 961, 678), 0.9)
    refined = HeadClassifier(WEIGHTS).refine(img, [person])
    assert any(d.cls in ("helmet", "no_helmet") for d in refined)
