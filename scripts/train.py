"""Fine-tune YOLO26 on Construction-PPE.

    python scripts/train.py --model yolo26n.pt --imgsz 640  --name n640    # fast baseline
    python scripts/train.py --model yolo26s.pt --imgsz 960  --name s960    # main model
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def data_yaml() -> str:
    """Ultralytics resolves a relative `path:` against its global datasets dir, so pin it."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "ppe.yaml").read_text())
    cfg["path"] = str((ROOT / "datasets" / "construction-ppe").resolve())
    out = ROOT / "runs" / "ppe_resolved.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="s960")
    a = ap.parse_args()

    weights = ROOT / "weights" / a.model
    model = YOLO(str(weights) if weights.exists() else a.model)
    model.train(
        data=data_yaml(), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
        patience=30, cos_lr=True, close_mosaic=15, seed=42, deterministic=False,
        project=str(ROOT / "runs"), name=a.name, exist_ok=True, workers=10, plots=True, cache="ram",
        # Site footage: varied lighting/dust -> stronger HSV jitter; no vertical flips (gravity).
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, fliplr=0.5, flipud=0.0, mixup=0.1,
    )
    best = YOLO(str(ROOT / "runs" / a.name / "weights" / "best.pt"))
    m = best.val(data=data_yaml(), split="test", imgsz=a.imgsz, batch=a.batch,
                 project=str(ROOT / "runs"), name=f"{a.name}_test", exist_ok=True, plots=True)
    names = best.names
    out = {"model": a.model, "imgsz": a.imgsz, "test_mAP50": m.box.map50, "test_mAP50_95": m.box.map,
           "precision": m.box.mp, "recall": m.box.mr,
           "per_class_mAP50": {names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50, strict=False)}}
    (ROOT / "runs" / f"{a.name}_test_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
