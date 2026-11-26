"""Live monitoring: annotate a webcam / RTSP camera / video file in real time.

    python -m minesafe.live                         # default webcam (0)
    python -m minesafe.live --source rtsp://user:pass@192.168.1.20/stream1
    python -m minesafe.live --source site.mp4       # plays the file as if it were a live camera

A background thread always holds only the *latest* frame, so if inference is slower than the
camera, stale frames are dropped instead of building up lag: the display stays live.
Press Q / Esc to quit. Alerts print to the console and append to --events (JSONL).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2

from .pipeline import LiveProcessor, load_models
from .rules import SiteConfig


class LatestFrameReader:
    """Reads a source on a background thread and keeps only the newest frame.

    For files, frames are paced to the file's FPS so a recording behaves like a live camera."""

    def __init__(self, source: str | int):
        self.source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        self.is_file = isinstance(self.source, str) and Path(self.source).exists()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame, self.seq, self.ended = None, 0, False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.fps if self.is_file else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.ended = True
                break
            with self._lock:
                self.frame, self.seq = frame, self.seq + 1
            if period:
                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))

    def read(self, last_seq: int = -1, timeout: float = 5.0):
        """Block until a frame newer than `last_seq` is available. Returns (seq, frame) or (None, None)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self._lock:
                if self.frame is not None and self.seq != last_seq:
                    return self.seq, self.frame.copy()
            if self.ended:
                return None, None
            time.sleep(0.002)
        return None, None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.cap.release()


def frames(source, processor: LiveProcessor):
    """Generator of (annotated_frame, new_events, dropped_frames) for any live source."""
    reader = LatestFrameReader(source)
    last, dropped = -1, 0
    try:
        while True:
            seq, frame = reader.read(last)
            if frame is None:
                break
            if last >= 0:
                dropped += seq - last - 1
            last = seq
            out, _, new = processor.process(frame)
            yield out, new, dropped
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index, RTSP/HTTP URL, or video file")
    ap.add_argument("--required-ppe", default="helmet,vest")
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--max-width", type=int, default=1280)
    a = ap.parse_args()
    det, head = load_models()
    proc = LiveProcessor(det, head, SiteConfig(required_ppe=tuple(a.required_ppe.split(","))))
    print(f"head classifier: {'on' if head else 'off'} | providers: {det.session.get_providers()[0]}")
    with open(a.events, "a", encoding="utf-8") as ev:
        for out, new, dropped in frames(a.source, proc):
            for e in new:
                print(f"ALERT t={e['time_s']}s worker #{e['track_id']}: {e['violation']}")
                ev.write(json.dumps(e) + "\n")
                ev.flush()
            if dropped:
                cv2.putText(out, f"dropped {dropped} stale frames to stay live", (8, out.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            if out.shape[1] > a.max_width:
                out = cv2.resize(out, (a.max_width, round(a.max_width * out.shape[0] / out.shape[1])))
            cv2.imshow("MineSafe Vision - live (Q to quit)", out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
