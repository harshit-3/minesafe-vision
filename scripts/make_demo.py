"""Build README assets: an annotated results grid and a demo GIF.

The GIF runs the full video pipeline (detect -> track -> rules -> debounce) on a simulated
camera pan across a held-out test image, so tracking and event logic are exercised end to end.

    python scripts/make_demo.py
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minesafe.detector import OnnxDetector  # noqa: E402
from minesafe.pipeline import process_image, process_video  # noqa: E402

EX = ROOT / "assets" / "examples"
det = OnnxDetector(ROOT / "weights" / "ppe_s960.onnx", conf=0.35)

# 1) results grid
tiles = []
for name in ["forklift_1", "no_helmet_2", "compliant_1", "forklift_2"]:
    _, _, ann = process_image(det, cv2.imread(str(EX / f"{name}.jpg")))
    ann = cv2.resize(ann, (960, round(960 * ann.shape[0] / ann.shape[1])))
    ann = ann[:600] if ann.shape[0] >= 600 else cv2.copyMakeBorder(ann, 0, 600 - ann.shape[0], 0, 0,
                                                                   cv2.BORDER_CONSTANT)
    tiles.append(ann)
cv2.imwrite(str(ROOT / "assets" / "results_grid.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]),
            [cv2.IMWRITE_JPEG_QUALITY, 82])

# 2) simulated pan video -> pipeline -> GIF
src = cv2.imread(str(EX / "forklift_2.jpg"))
h, w = src.shape[:2]
cw, ch = int(w * 0.7), int(h * 0.7)
tmp = Path(tempfile.mkdtemp())
raw, out = tmp / "pan.mp4", tmp / "pan_annotated.mp4"
vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (cw, ch))
n = 90
for i in range(n):
    t = 0.5 - 0.5 * np.cos(np.pi * i / (n - 1))  # ease in/out
    x, y = int(t * (w - cw)), int((1 - t) * (h - ch))
    vw.write(src[y:y + ch, x:x + cw])
vw.release()
res = process_video(det, str(raw), str(out))
print(json.dumps({"frames": res["frames"], "events": res["events"]}, indent=1))

cap = cv2.VideoCapture(str(out))
frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % 3 == 0:
        f = cv2.resize(f, (480, round(480 * f.shape[0] / f.shape[1])))
        rgb = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        frames.append(rgb.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    i += 1
frames[0].save(ROOT / "assets" / "demo.gif", save_all=True, append_images=frames[1:], duration=200, loop=0,
               optimize=True)
print("gif frames", len(frames), round((ROOT / "assets" / "demo.gif").stat().st_size / 1e6, 2), "MB")
