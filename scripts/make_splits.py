"""Create train/val/test splits.

The official 200-image `val` set becomes our untouched TEST set. A seeded 15% of the
official train set becomes VAL (early stopping / model selection).
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "construction-ppe"
imgs = sorted((ROOT / "images" / "train").glob("*"))
random.Random(42).shuffle(imgs)
n_val = int(0.15 * len(imgs))
splits = {"train": imgs[n_val:], "val": imgs[:n_val], "test": sorted((ROOT / "images" / "val").glob("*"))}
for name, files in splits.items():
    (ROOT / f"{name}.txt").write_text("\n".join(str(p.resolve().as_posix()) for p in files) + "\n")
    print(name, len(files))
