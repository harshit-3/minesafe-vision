from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from minesafe.api import MODEL, app

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="ONNX model not exported yet")
FIX = Path(__file__).parent / "fixtures"
client = TestClient(app)


def _post(name, path="/detect", **params):
    with open(FIX / name, "rb") as f:
        return client.post(path, files={"file": (name, f, "image/jpeg")}, params=params)


def test_health_and_model():
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model").json()
    assert "person" in info["classes"] and info["input_size"] > 0


def test_detects_workers_and_forklift():
    r = _post("forklift.jpg")
    assert r.status_code == 200
    body = r.json()
    assert body["workers"] >= 3
    assert any(d["cls"] == "forklift" for d in body["detections"])
    for w in body["details"]:
        assert set(w["ppe"]) == {"helmet", "vest"}
        assert all(v in ("ok", "missing", "unknown") for v in w["ppe"].values())


def test_boxes_are_in_image_coordinates():
    img = cv2.imread(str(FIX / "forklift.jpg"))
    h, w = img.shape[:2]
    for d in _post("forklift.jpg").json()["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_required_ppe_param():
    body = _post("forklift.jpg", required_ppe="helmet").json()
    assert all(set(w["ppe"]) == {"helmet"} for w in body["details"])
    assert _post("forklift.jpg", required_ppe="cape").status_code == 422


def test_annotated_returns_jpeg():
    r = _post("forklift.jpg", path="/detect/annotated")
    assert r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0


def test_rejects_non_image():
    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415
