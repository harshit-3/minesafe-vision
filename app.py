"""Gradio demo (runs locally or on Hugging Face Spaces, CPU is fine).

    python app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from minesafe.detector import OnnxDetector  # noqa: E402
from minesafe.pipeline import process_image, process_video, summarise  # noqa: E402
from minesafe.rules import SiteConfig  # noqa: E402

DET = OnnxDetector(ROOT / "weights" / "ppe_s960.onnx")
EXAMPLES = sorted(str(p) for p in (ROOT / "assets" / "examples").glob("*.jpg"))


def run_image(img_rgb, ppe, margin, conf):
    if img_rgb is None:
        return None, None, ""
    DET.conf = conf
    cfg = SiteConfig(required_ppe=tuple(ppe), vehicle_margin=margin)
    _, workers, annotated = process_image(DET, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cfg)
    s = summarise(workers)
    table = pd.DataFrame([{"worker": i + 1, **{k: v for k, v in d["ppe"].items()},
                           "violations": ", ".join(d["violations"]) or "-"} for i, d in enumerate(s["details"])])
    headline = (f"**{s['workers']} workers** · {s['compliant']} fully compliant · "
                f"**{s['violations']} with violations**")
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), table, headline


def run_video(path, ppe, margin):
    if not path:
        return None, None
    cfg = SiteConfig(required_ppe=tuple(ppe), vehicle_margin=margin)
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    res = process_video(DET, path, out, cfg, max_frames=300)
    ev = pd.DataFrame(res["events"])[["time_s", "track_id", "violation"]] if res["events"] else \
        pd.DataFrame(columns=["time_s", "track_id", "violation"])
    return out, ev


with gr.Blocks(title="MineSafe Vision") as demo:
    gr.Markdown("# MineSafe Vision\nPPE compliance and vehicle-proximity monitoring for site cameras. "
                "YOLO26s fine-tuned on Construction-PPE, served with ONNX Runtime on CPU.")
    with gr.Row():
        ppe = gr.CheckboxGroup(["helmet", "vest", "goggles", "gloves", "boots"], value=["helmet", "vest"],
                               label="Required PPE")
        margin = gr.Slider(0, 2, 0.5, step=0.1, label="Vehicle exclusion margin (× vehicle size)")
        conf = gr.Slider(0.1, 0.9, 0.35, step=0.05, label="Detection confidence")
    with gr.Tab("Image"):
        with gr.Row():
            inp = gr.Image(label="Site photo", type="numpy")
            out = gr.Image(label="Result")
        headline = gr.Markdown()
        table = gr.Dataframe(label="Workers")
        gr.Button("Check compliance", variant="primary").click(run_image, [inp, ppe, margin, conf],
                                                              [out, table, headline])
        if EXAMPLES:
            gr.Examples(EXAMPLES, inp)
    with gr.Tab("Video"):
        gr.Markdown("Tracks each worker and raises an event only when a violation persists "
                    "(~0.4 s), so detector flicker doesn't spam alerts. The first 300 frames are processed.")
        with gr.Row():
            vin = gr.Video(label="Site video")
            vout = gr.Video(label="Annotated")
        events = gr.Dataframe(label="Violation events")
        gr.Button("Process video", variant="primary").click(run_video, [vin, ppe, margin], [vout, events])

if __name__ == "__main__":
    demo.launch()
