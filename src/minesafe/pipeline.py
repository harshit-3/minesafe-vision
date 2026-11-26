"""Frame pipeline: detect -> head check -> track -> rules -> debounce -> annotate / events.

`LiveProcessor` handles one frame at a time, so the same code serves single images, video files
and live camera streams (see minesafe.live).

    python -m minesafe.pipeline video.mp4 --out annotated.mp4 --events events.jsonl
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
WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


def load_models(detector_path: str | Path = WEIGHTS / "ppe_s960.onnx", head_path: str | Path | None = "auto",
                conf: float = 0.35):
    """Detector + optional second-stage head classifier (used automatically when its weights exist)."""
    det = OnnxDetector(detector_path, conf=conf)
    head = None
    if head_path == "auto":
        head_path = WEIGHTS / "head_cls.onnx"
        head_path = head_path if Path(head_path).exists() else None
    if head_path:
        from .head_classifier import HeadClassifier
        head = HeadClassifier(head_path)
    return det, head


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


def detect(det: OnnxDetector, bgr: np.ndarray, head=None) -> list[Detection]:
    dets = det(bgr)
    return head.refine(bgr, dets) if head else dets


def process_image(det: OnnxDetector, bgr: np.ndarray, cfg: SiteConfig = SiteConfig(), head=None):
    dets = detect(det, bgr, head)
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


class LiveProcessor:
    """Stateful per-stream processor: keeps worker identities and debounces alerts across frames."""

    def __init__(self, det: OnnxDetector, head=None, cfg: SiteConfig = SiteConfig(), fps: float = 15.0):
        self.det, self.head, self.cfg = det, head, cfg
        self.tracker = IoUTracker()
        self.debouncer = ViolationDebouncer(window=max(3, int(fps * 0.6)), min_hits=max(2, int(fps * 0.4)),
                                            cooldown=int(fps * 30))
        self.frame_idx, self.started = 0, time.time()
        self.events: list[dict] = []
        self._fps_ema = None

    def process(self, frame: np.ndarray, overlay_stats: bool = True) -> tuple[np.ndarray, list[WorkerStatus], list]:
        t0 = time.perf_counter()
        dets = self.tracker.update(detect(self.det, frame, self.head))
        workers = evaluate_frame(dets, self.cfg)
        new = [{**asdict(e), "time_s": round(time.time() - self.started, 2)}
               for e in self.debouncer.update(self.frame_idx, workers)]
        self.events.extend(new)
        out = annotate(frame, dets, workers, self.cfg)
        dt = time.perf_counter() - t0
        self._fps_ema = 1 / dt if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 / max(dt, 1e-6)
        if overlay_stats:
            s = summarise(workers)
            txt = f"{self._fps_ema:4.1f} FPS | workers {s['workers']} | violations {s['violations']} | " \
                  f"alerts {len(self.events)}"
            cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 11 * len(txt)), 30), (0, 0, 0), -1)
            cv2.putText(out, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self.frame_idx += 1
        return out, workers, new


def process_video(det: OnnxDetector, src: str, out_path: str | None, cfg: SiteConfig = SiteConfig(),
                  events_path: str | None = None, max_frames: int | None = None, head=None) -> dict:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if out_path else None
    proc = LiveProcessor(det, head, cfg, fps)
    events, n, t_proc = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        t0 = time.perf_counter()
        out, _, new = proc.process(frame, overlay_stats=False)
        t_proc += time.perf_counter() - t0
        events += [{**e, "time_s": round(e["frame"] / fps, 2)} for e in new]
        if writer:
            writer.write(out)
        n += 1
    cap.release()
    if writer:
        writer.release()
    if events_path:
        Path(events_path).write_text("\n".join(json.dumps(e) for e in events))
    return {"frames": n, "events": events, "pipeline_fps": round(n / t_proc, 1) if t_proc else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--model", default=str(WEIGHTS / "ppe_s960.onnx"))
    ap.add_argument("--no-head-classifier", action="store_true")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--events", default="events.jsonl")
    a = ap.parse_args()
    det, head = load_models(a.model, None if a.no_head_classifier else "auto")
    res = process_video(det, a.source, a.out, events_path=a.events, head=head)
    print(json.dumps({k: v for k, v in res.items() if k != "events"}), f"{len(res['events'])} events")
