"""Downscale dataset images in place to max side 1280 px (YOLO labels are normalised, so
they stay valid). Cuts JPEG decode time per epoch ~3x; 1280 > 960 training size."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "construction-ppe" / "images"
MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1280


def shrink(p: Path) -> bool:
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    if max(h, w) <= MAX:
        return False
    r = MAX / max(h, w)
    img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


files = [p for p in ROOT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
with ThreadPoolExecutor(16) as ex:
    print(f"resized {sum(ex.map(shrink, files))}/{len(files)} images to max {MAX}px")
