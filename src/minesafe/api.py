"""FastAPI service.  Run:  uvicorn minesafe.api:app --port 8000   (PYTHONPATH=src)

POST /detect             image -> workers, PPE status, violations (JSON)
POST /detect/annotated   image -> annotated JPEG
GET  /live               browser page: live annotated camera feed + alert list
GET  /live/stream        MJPEG stream of annotated frames for a configured camera
GET  /live/events        alerts raised so far on that camera

Live cameras are configured server-side, never taken from the request:
    MINESAFE_CAMERAS="gate=rtsp://user:pass@10.0.0.5/stream1,webcam=0,demo=assets/demo_site.mp4"
"""
from __future__ import annotations

import html
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .pipeline import LiveProcessor, load_models, process_image, summarise
from .rules import PPE_RULES, SiteConfig

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.getenv("MINESAFE_MODEL", ROOT / "weights" / "ppe_s960.onnx"))
HEAD_MODEL = os.getenv("MINESAFE_HEAD_MODEL", "auto")
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="MineSafe Vision API", version="1.1.0",
              description="PPE compliance and vehicle-proximity checks for site camera frames and live streams.")


@lru_cache
def models():
    return load_models(MODEL, None if HEAD_MODEL in ("", "off") else HEAD_MODEL)


def detector():
    return models()[0]


def cameras() -> dict[str, str]:
    raw = os.getenv("MINESAFE_CAMERAS", "webcam=0")
    out = {}
    for item in raw.split(","):
        name, sep, src = item.partition("=")
        if sep and name.strip():
            src = src.strip()
            if not (src.isdigit() or "://" in src):  # relative file paths resolve against the repo
                src = str((ROOT / src).resolve())
            out[name.strip()] = src
    return out


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
    d, head = models()
    return {"model": MODEL.name, "input_size": d.size, "classes": d.names, "head_classifier": head is not None,
            "providers": d.session.get_providers(), "cameras": sorted(cameras())}


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 required_ppe: str = Query("helmet,vest", description="comma-separated PPE items"),
                 vehicle_margin: float = Query(0.5, ge=0, le=3)):
    img = await _read_image(file)
    cfg = _config(required_ppe, vehicle_margin)
    t0 = time.perf_counter()
    det, head = models()
    dets, workers, _ = process_image(det, img, cfg, head)
    return {**summarise(workers),
            "detections": [{"cls": d.cls, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box]}
                           for d in dets],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/detect/annotated")
async def detect_annotated(file: UploadFile = File(...), required_ppe: str = Query("helmet,vest"),
                           vehicle_margin: float = Query(0.5, ge=0, le=3)):
    img = await _read_image(file)
    det, head = models()
    _, _, annotated = process_image(det, img, _config(required_ppe, vehicle_margin), head)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


# ----------------------------------------------------------------------------- live streams
_LIVE: dict[str, LiveProcessor] = {}
_LIVE_LOCK = threading.Lock()


def _camera(name: str) -> str:
    cams = cameras()
    if name not in cams:
        raise HTTPException(404, f"unknown camera {name!r}; configured: {sorted(cams)}")
    return cams[name]


@app.get("/live/stream")
def live_stream(camera: str = Query("webcam"), required_ppe: str = Query("helmet,vest")):
    src = _camera(camera)
    cfg = _config(required_ppe, 0.5)
    det, head = models()
    proc = LiveProcessor(det, head, cfg)
    with _LIVE_LOCK:
        _LIVE[camera] = proc

    def gen():
        from .live import frames
        for out, _new, _dropped in frames(src, proc):
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/live/events")
def live_events(camera: str = Query("webcam")):
    _camera(camera)
    proc = _LIVE.get(camera)
    return {"camera": camera, "events": proc.events[-50:] if proc else []}


@app.get("/live", response_class=HTMLResponse)
def live_page(camera: str = Query("webcam")):
    _camera(camera)
    cam = html.escape(camera)
    options = "".join(f'<option {"selected" if c == camera else ""}>{html.escape(c)}</option>' for c in cameras())
    return f"""<!doctype html><html><head><title>MineSafe live: {cam}</title>
<style>body{{font-family:system-ui;background:#111;color:#eee;margin:20px}} img{{max-width:72vw;border:1px solid #333}}
.wrap{{display:flex;gap:20px}} #ev{{width:26vw;font-size:14px}} .a{{background:#3a1515;padding:6px;margin:4px 0;
border-left:3px solid #e53e3e}}</style></head><body>
<h2>MineSafe Vision: live feed</h2>
<form>Camera <select name="camera" onchange="this.form.submit()">{options}</select></form>
<div class="wrap"><img src="/live/stream?camera={cam}">
<div id="ev"><b>Alerts</b><div id="list">none yet</div></div></div>
<script>
setInterval(async () => {{
  const r = await fetch('/live/events?camera={cam}'); const d = await r.json();
  document.getElementById('list').innerHTML = d.events.slice().reverse().map(e =>
    `<div class="a">t=${{e.time_s}}s · worker #${{e.track_id}} · ${{e.violation}}</div>`).join('') || 'none yet';
}}, 1000);
</script></body></html>"""
