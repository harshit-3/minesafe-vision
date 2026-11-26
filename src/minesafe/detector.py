"""Lightweight ONNX Runtime detector for end-to-end (NMS-free) YOLO26 exports.

No torch / ultralytics at inference time: just numpy + OpenCV + onnxruntime.
Picks CUDA / TensorRT execution providers automatically when available.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .rules import CLASS_NAMES, Detection


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square `size`. Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


class OnnxDetector:
    def __init__(self, path: str | Path, conf: float = 0.25, providers: list[str] | None = None):
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail and hasattr(ort, "preload_dlls"):
            try:  # Windows: load pip-installed CUDA/cuDNN DLLs before creating a CUDA session
                ort.preload_dlls()
            except Exception:
                pass
        providers = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.conf = conf
        self.names = CLASS_NAMES

    def preprocess(self, bgr: np.ndarray):
        img, r, pad = letterbox(bgr, self.size)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        return np.ascontiguousarray(x), r, pad

    def postprocess(self, out: np.ndarray, r: float, pad: tuple[int, int], shape) -> list[Detection]:
        # End-to-end output: [1, max_det, 6] = x1, y1, x2, y2, score, class
        preds = out[0]
        preds = preds[preds[:, 4] >= self.conf]
        h, w = shape[:2]
        dets = []
        for x1, y1, x2, y2, s, c in preds:
            x1 = float(np.clip((x1 - pad[0]) / r, 0, w))
            x2 = float(np.clip((x2 - pad[0]) / r, 0, w))
            y1 = float(np.clip((y1 - pad[1]) / r, 0, h))
            y2 = float(np.clip((y2 - pad[1]) / r, 0, h))
            dets.append(Detection(self.names[int(c)], (x1, y1, x2, y2), float(s)))
        return dets

    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        x, r, pad = self.preprocess(bgr)
        out = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(out, r, pad, bgr.shape)
