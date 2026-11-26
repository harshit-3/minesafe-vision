"""Train the second-stage helmet / bare-head classifier on high-resolution head crops.

Crops come from ground-truth helmet / no_helmet boxes. To match what the classifier sees at
inference time (detector boxes, or head regions estimated from a person box), training boxes are
randomly shifted and scaled. A heavy blur/downscale augmentation simulates distant workers.

    python scripts/train_head_classifier.py
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minesafe.head_classifier import INPUT, crop_head  # noqa: E402
from minesafe.rules import CLASS_NAMES  # noqa: E402

DS = ROOT / "datasets" / "construction-ppe"
LABELS = {"no_helmet": 0, "helmet": 1}
rng = random.Random(42)


def samples(list_file: Path):
    out = []
    for img_path in Path(list_file).read_text().split():
        p = Path(img_path)
        lab = DS / "labels" / p.parent.name / f"{p.stem}.txt"
        for line in lab.read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            name = CLASS_NAMES[int(c)]
            if name in LABELS:
                out.append((str(p), (cx, cy, bw, bh), LABELS[name]))
    return out


class HeadDataset(torch.utils.data.Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def _img(self, path):
        if path not in self.cache:
            self.cache[path] = cv2.imread(path)
        return self.cache[path]

    def __getitem__(self, i):
        path, (cx, cy, bw, bh), y = self.items[i]
        img = self._img(path)
        h, w = img.shape[:2]
        box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        if self.train:  # jitter box like an imperfect detector / person-box estimate
            s = rng.uniform(0.85, 1.6)
            dx, dy = rng.uniform(-0.2, 0.2) * bw * w, rng.uniform(-0.2, 0.25) * bh * h
            mx, my = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
            hw, hh = (box[2] - box[0]) * s / 2, (box[3] - box[1]) * s / 2
            box = [mx - hw, my - hh, mx + hw, my + hh]
        crop = crop_head(img, box)
        if self.train:
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            if rng.random() < 0.35:  # distant worker: downscale then upscale
                f = rng.uniform(0.25, 0.6)
                small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                crop = cv2.resize(small, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= rng.uniform(0.6, 1.4)
            hsv[..., 2] *= rng.uniform(0.6, 1.4)
            crop = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        x = np.ascontiguousarray(crop[:, :, ::-1]).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)), y


def run_epoch(model, loader, opt=None, device="cuda"):
    model.train(opt is not None)
    tot, correct, loss_sum = 0, 0, 0.0
    weights = torch.tensor([2.0, 1.0], device=device)  # missing a bare head is the costly error
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y, weight=weights)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
    return loss_sum / tot, correct / tot


if __name__ == "__main__":
    torch.manual_seed(0)
    train, val = samples(DS / "train.txt"), samples(DS / "val.txt")
    print("train", len(train), "val", len(val), "bare heads in train:", sum(y == 0 for *_, y in train))
    dl = lambda items, tr: torch.utils.data.DataLoader(HeadDataset(items, tr), batch_size=64, shuffle=tr,  # noqa: E731
                                                        num_workers=0)
    tl, vl = dl(train, True), dl(val, False)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    best, best_state = 0.0, None
    for ep in range(25):
        tr_loss, tr_acc = run_epoch(model, tl, opt)
        va_loss, va_acc = run_epoch(model, vl)
        sched.step()
        print(f"epoch {ep + 1:2d} train {tr_acc:.3f}  val {va_acc:.3f} (loss {va_loss:.3f})")
        if va_acc > best:
            best, best_state = va_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval().cpu()
    out = ROOT / "weights" / "head_cls.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, INPUT, INPUT), str(out), input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    (ROOT / "runs" / "head_cls.json").write_text(json.dumps({"best_val_acc": best, "n_train": len(train),
                                                             "n_val": len(val)}, indent=2))
    print(f"best val acc {best:.3f} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
