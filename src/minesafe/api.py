"""FastAPI service.  Run:  uvicorn minesafe.api:app --port 8000   (PYTHONPATH=src)

POST /detect          image -> workers, PPE status, violations (JSON)
POST /detect/annotated image -> annotated JPEG
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .detector import OnnxDetector
from .pipeline import process_image, summarise
from .rules import PPE_RULES, SiteConfig

MODEL = Path(os.getenv("MINESAFE_MODEL", Path(__file__).resolve().parents[2] / "weights" / "ppe_s960.onnx"))
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="MineSafe Vision API", version="1.0.0",
              description="PPE compliance and vehicle-proximity checks for site camera frames.")


@lru_cache
def detector() -> OnnxDetector:
    return OnnxDetector(MODEL)


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 15 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "could not decode image (use JPEG/PNG)")
    return img


def _config(required: str, margin: float) -> SiteConfig:
    items = tuple(s.strip() for s in required.split(",") if s.strip())
    bad = [i for i in items if i not in PPE_RULES]
    if bad:
        raise HTTPException(422, f"unknown PPE items {bad}; choose from {sorted(PPE_RULES)}")
    return SiteConfig(required_ppe=items, vehicle_margin=margin)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    d = detector()
    return {"model": MODEL.name, "input_size": d.size, "classes": d.names,
            "providers": d.session.get_providers()}


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 required_ppe: str = Query("helmet,vest", description="comma-separated PPE items"),
                 vehicle_margin: float = Query(0.5, ge=0, le=3)):
    img = await _read_image(file)
    cfg = _config(required_ppe, vehicle_margin)
    t0 = time.perf_counter()
    dets, workers, _ = process_image(detector(), img, cfg)
    return {**summarise(workers),
            "detections": [{"cls": d.cls, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box]}
                           for d in dets],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), required_ppe: str = Query("helmet,vest"),
                           vehicle_margin: float = Query(0.5, ge=0, le=3)):
    img = await _read_image(file)
    _, _, annotated = process_image(detector(), img, _config(required_ppe, vehicle_margin))
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")
