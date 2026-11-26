"""Image + video pipelines: detect -> track -> rules -> debounce -> annotate / log events.

    python -m minesafe.pipeline video.mp4 --model weights/ppe_s960.onnx --out out.mp4
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .detector import OnnxDetector
from .rules import Detection, SiteConfig, ViolationDebouncer, WorkerStatus, evaluate_frame, expand_box
from .tracker import IoUTracker

GREEN, RED, AMBER, GREY = (60, 180, 75), (40, 40, 220), (0, 165, 255), (160, 160, 160)


def annotate(img: np.ndarray, dets: list[Detection], workers: list[WorkerStatus], cfg: SiteConfig) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(img.shape[:2]) / 400))
    for d in dets:
        if d.cls == "forklift":
            x1, y1, x2, y2 = map(int, d.box)
            ex = tuple(map(int, expand_box(d.box, cfg.vehicle_margin)))
            overlay = out.copy()
            cv2.rectangle(overlay, ex[:2], ex[2:], AMBER, -1)
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), AMBER, t)
            cv2.putText(out, "VEHICLE ZONE", (ex[0] + 4, ex[1] + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, AMBER, t)
    for name, poly in cfg.restricted_zones.items():
        cv2.polylines(out, [np.array(poly, dtype=np.int32)], True, RED, t)
        cv2.putText(out, name, tuple(map(int, poly[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, RED, t)
    for w in workers:
        x1, y1, x2, y2 = map(int, w.person.box)
        color = RED if w.violations else (GREEN if all(v == "ok" for v in w.ppe.values()) else GREY)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 * t)
        tag = f"#{w.person.track_id} " if w.person.track_id else ""
        tag += "operator " if w.is_operator else ""
        label = tag + (", ".join(w.violations) if w.violations else
                       ("compliant" if color == GREEN else "check PPE"))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, t)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (255, 255, 255), t)
    return out


def process_image(det: OnnxDetector, bgr: np.ndarray, cfg: SiteConfig = SiteConfig()):
    dets = det(bgr)
    workers = evaluate_frame(dets, cfg)
    return dets, workers, annotate(bgr, dets, workers, cfg)


def summarise(workers: list[WorkerStatus]) -> dict:
    return {
        "workers": len(workers),
        "compliant": sum(1 for w in workers if not w.violations and all(v == "ok" for v in w.ppe.values())),
        "violations": sum(1 for w in workers if w.violations),
        "details": [{"box": [round(v, 1) for v in w.person.box], "ppe": w.ppe, "violations": w.violations,
                     "operator": w.is_operator}
                    for w in workers],
    }


def process_video(det: OnnxDetector, src: str, out_path: str | None, cfg: SiteConfig = SiteConfig(),
                  events_path: str | None = None, max_frames: int | None = None) -> dict:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if out_path else None
    tracker = IoUTracker()
    debouncer = ViolationDebouncer(window=int(fps * 0.6), min_hits=int(fps * 0.4), cooldown=int(fps * 30))
    events, n, t_infer = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        t0 = time.perf_counter()
        dets = tracker.update(det(frame))
        t_infer += time.perf_counter() - t0
        workers = evaluate_frame(dets, cfg)
        for e in debouncer.update(n, workers):
            events.append({**asdict(e), "time_s": round(n / fps, 2)})
        if writer:
            writer.write(annotate(frame, dets, workers, cfg))
        n += 1
    cap.release()
    if writer:
        writer.release()
    if events_path:
        Path(events_path).write_text("\n".join(json.dumps(e) for e in events))
    return {"frames": n, "events": events, "model_fps": round(n / t_infer, 1) if t_infer else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--model", default="weights/ppe_s960.onnx")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--events", default="events.jsonl")
    a = ap.parse_args()
    res = process_video(OnnxDetector(a.model), a.source, a.out, events_path=a.events)
    print(json.dumps({k: v for k, v in res.items() if k != "events"}), f"{len(res['events'])} events")
