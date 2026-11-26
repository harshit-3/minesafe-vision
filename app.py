"""Gradio demo (runs locally or on Hugging Face Spaces, CPU is fine).

    python app.py

Everything is processed live: webcam frames are annotated as they stream in, videos play back
annotated frame-by-frame while they are processed (alerts appear as they fire), and images are
checked as soon as they are uploaded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from minesafe.pipeline import LiveProcessor, load_models, process_image, summarise  # noqa: E402
from minesafe.rules import SiteConfig  # noqa: E402

DET, HEAD = load_models()
EXAMPLES = sorted(str(p) for p in (ROOT / "assets" / "examples").glob("*.jpg"))
DEMO_VIDEO = ROOT / "assets" / "demo_site.mp4"
ON_GPU = "CUDAExecutionProvider" in DET.session.get_providers()
EVENT_COLS = ["time_s", "track_id", "violation"]


def _cfg(ppe, margin):
    return SiteConfig(required_ppe=tuple(ppe), vehicle_margin=margin)


def _events_df(events):
    return pd.DataFrame(events)[EVENT_COLS] if events else pd.DataFrame(columns=EVENT_COLS)


def run_image(img_rgb, ppe, margin, conf):
    if img_rgb is None:
        return None, None, ""
    DET.conf = conf
    _, workers, annotated = process_image(DET, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), _cfg(ppe, margin), HEAD)
    s = summarise(workers)
    table = pd.DataFrame([{"worker": i + 1, **d["ppe"], "violations": ", ".join(d["violations"]) or "-"}
                          for i, d in enumerate(s["details"])])
    headline = f"**{s['workers']} workers** · {s['compliant']} fully compliant · **{s['violations']} with violations**"
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), table, headline


def run_webcam(frame_rgb, state, ppe, margin, conf):
    """Called for every webcam frame; the LiveProcessor in `state` keeps track IDs and alerts."""
    if frame_rgb is None:
        return None, state, _events_df([])
    key = (tuple(ppe), margin)
    if state is None or state[0] != key:
        state = (key, LiveProcessor(DET, HEAD, _cfg(ppe, margin), fps=10))
    DET.conf = conf
    out, _, _ = state[1].process(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), state, _events_df(state[1].events[::-1])


def run_video(path, ppe, margin, conf):
    """Stream annotated frames while the video is processed (no waiting for the whole file)."""
    if not path:
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # On CPU (e.g. a free Hugging Face Space) analyse ~5 frames per video-second so playback keeps up;
    # the debouncer is told the effective frame rate so alert timing stays correct.
    stride = 1 if ON_GPU else max(1, round(fps / 5))
    DET.conf = conf
    proc = LiveProcessor(DET, HEAD, _cfg(ppe, margin), fps / stride)
    last_emit = 0.0
    while True:
        for _ in range(stride - 1):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            break
        out, _, new = proc.process(frame)
        # Push to the browser at most ~12 times/s; always push when a new alert fires.
        if new or time.perf_counter() - last_emit > 1 / 12:
            last_emit = time.perf_counter()
            yield cv2.cvtColor(out, cv2.COLOR_BGR2RGB), _events_df(proc.events[::-1])
    cap.release()
    yield gr.skip(), _events_df(proc.events[::-1])


with gr.Blocks(title="MineSafe Vision") as demo:
    gr.Markdown("# MineSafe Vision\nLive PPE compliance and vehicle-proximity monitoring. YOLO26s detector + "
                "high-resolution head classifier, served with ONNX Runtime.")
    with gr.Row():
        ppe = gr.CheckboxGroup(["helmet", "vest", "goggles", "gloves", "boots"], value=["helmet", "vest"],
                               label="Required PPE")
        margin = gr.Slider(0, 2, 0.5, step=0.1, label="Vehicle exclusion margin (× vehicle size)")
        conf = gr.Slider(0.1, 0.9, 0.35, step=0.05, label="Detection confidence")

    with gr.Tab("Live camera"):
        gr.Markdown("Allow camera access. Every frame is checked as it arrives; alerts fire when a violation "
                    "persists for the same tracked worker.")
        state = gr.State(None)
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
            live_out = gr.Image(label="Live result", streaming=True)
        live_events = gr.Dataframe(label="Alerts", headers=EVENT_COLS)
        cam.stream(run_webcam, [cam, state, ppe, margin, conf], [live_out, state, live_events],
                   stream_every=0.1, concurrency_limit=1)

    with gr.Tab("Video"):
        gr.Markdown("Upload a site video: annotated frames play back **while** it is processed.")
        with gr.Row():
            vin = gr.Video(label="Site video")
            vout = gr.Image(label="Live playback (annotated)")
        events = gr.Dataframe(label="Alerts", headers=EVENT_COLS)
        vin.change(run_video, [vin, ppe, margin, conf], [vout, events])
        if DEMO_VIDEO.exists():
            gr.Examples([str(DEMO_VIDEO)], vin, label="Demo clip")

    with gr.Tab("Image"):
        with gr.Row():
            inp = gr.Image(label="Site photo", type="numpy")
            out = gr.Image(label="Result")
        headline = gr.Markdown()
        table = gr.Dataframe(label="Workers")
        for trigger in (inp.change, ppe.change, margin.release, conf.release):
            trigger(run_image, [inp, ppe, margin, conf], [out, table, headline])
        if EXAMPLES:
            gr.Examples(EXAMPLES, inp)

if __name__ == "__main__":
    demo.launch()
