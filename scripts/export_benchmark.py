"""Export the trained model and benchmark accuracy + speed across deployment formats.

    python scripts/export_benchmark.py --run s960 --imgsz 960

Formats: PyTorch FP32, PyTorch FP16, ONNX (ONNX Runtime CPU via our lightweight detector),
TensorRT FP16. Accuracy is re-measured on the held-out test set for each format so speed-ups
are shown to cost no accuracy.
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install mid-benchmark

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train import data_yaml  # noqa: E402

from minesafe.detector import OnnxDetector  # noqa: E402


def time_fn(fn, x, n=100, warmup=10):
    for _ in range(warmup):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), statistics.quantiles(ts, n=20)[18]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s960")
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()

    pt = ROOT / "runs" / a.run / "weights" / "best.pt"
    data = data_yaml()
    img = cv2.imread(sorted((ROOT / "datasets" / "construction-ppe" / "images" / "val").glob("*"))[0].as_posix())
    rows = []

    def val(model, **kw):
        m = model.val(data=data, split="test", imgsz=a.imgsz, batch=1, verbose=False, plots=False,
                      project=str(ROOT / "runs"), name="bench", exist_ok=True, **kw)
        return round(m.box.map50, 4), round(m.box.map, 4)

    # PyTorch FP32 / FP16 (end-to-end predict on a real 2048px frame, includes pre/post-processing)
    model = YOLO(str(pt))
    for half in (False, True):
        m50, m5095 = val(model, half=half)
        p50, p95 = time_fn(lambda x, h=half: model.predict(x, imgsz=a.imgsz, half=h, verbose=False), img)
        rows.append({"format": f"PyTorch {'FP16' if half else 'FP32'} (GPU)", "mAP50": m50, "mAP50_95": m5095,
                     "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1), "fps": round(1000 / p50, 1)})

    # ONNX: export once, copy to weights/ for the API and Docker image
    onnx_path = Path(model.export(format="onnx", imgsz=a.imgsz, simplify=True, opset=17, end2end=True))
    dst = ROOT / "weights" / f"ppe_{a.run}.onnx"
    shutil.copy(onnx_path, dst)
    m50, m5095 = val(YOLO(str(onnx_path), task="detect"))
    det = OnnxDetector(dst, providers=["CPUExecutionProvider"])
    p50, p95 = time_fn(det, img, n=30, warmup=3)
    rows.append({"format": "ONNX Runtime (CPU, i9-14900HX)", "mAP50": m50, "mAP50_95": m5095,
                 "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1), "fps": round(1000 / p50, 1),
                 "size_mb": round(dst.stat().st_size / 1e6, 1)})
    try:
        det_gpu = OnnxDetector(dst, providers=["CUDAExecutionProvider"])
        p50, p95 = time_fn(det_gpu, img)
        rows.append({"format": "ONNX Runtime (CUDA)", "mAP50": m50, "mAP50_95": m5095,
                     "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1), "fps": round(1000 / p50, 1)})
    except Exception as e:
        print("ORT CUDA skipped:", e)

    # TensorRT FP16
    try:
        eng = Path(YOLO(str(pt)).export(format="engine", imgsz=a.imgsz, half=True, workspace=4))
        trt = YOLO(str(eng), task="detect")
        m50, m5095 = val(trt, half=True)
        p50, p95 = time_fn(lambda x: trt.predict(x, imgsz=a.imgsz, half=True, verbose=False), img)
        rows.append({"format": "TensorRT FP16 (GPU)", "mAP50": m50, "mAP50_95": m5095,
                     "latency_p50_ms": round(p50, 1), "latency_p95_ms": round(p95, 1), "fps": round(1000 / p50, 1)})
    except Exception as e:  # TensorRT is optional
        print("TensorRT export skipped:", e)

    out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "imgsz": a.imgsz, "input_frame": list(img.shape[:2]), "results": rows}
    (ROOT / "runs" / f"benchmark_{a.run}.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(r)
