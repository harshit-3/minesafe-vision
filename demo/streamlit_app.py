"""Streamlit demo for free hosting (Streamlit Community Cloud).  Run:  streamlit run demo/streamlit_app.py

Same pipeline as the Gradio app (app.py): detector + head classifier + tracker + rules + debouncer.
The Live tab streams a camera over WebRTC: each frame is annotated on the server and streamed back in real
time, and stale frames are dropped (never queued) when inference is slower than the camera.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT).startswith("/mount/src"):  # Streamlit Community Cloud: small shared CPU
    os.environ.setdefault("MINESAFE_CPU_THREADS", "2")
sys.path.insert(0, str(ROOT / "src"))
from minesafe.pipeline import LiveProcessor, load_models, process_image, summarise  # noqa: E402
from minesafe.rules import SiteConfig  # noqa: E402

st.set_page_config(page_title="MineSafe Vision", page_icon="🦺", layout="wide")
EXAMPLES = sorted((ROOT / "assets" / "examples").glob("*.jpg"))
DEMO_VIDEO = ROOT / "assets" / "demo_site.mp4"
EVENT_COLS = ["time_s", "track_id", "violation"]
MAX_VIDEO_S = 30  # keep a shared CPU responsive


@st.cache_resource
def models():
    return load_models()


DET, HEAD = models()
ON_GPU = "CUDAExecutionProvider" in DET.session.get_providers()

with st.sidebar:
    st.header("Site rules")
    ppe = st.multiselect("Required PPE", ["helmet", "vest", "goggles", "gloves", "boots"], ["helmet", "vest"])
    margin = st.slider("Vehicle exclusion margin (× vehicle size)", 0.0, 2.0, 0.5, 0.1)
    DET.conf = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
    st.divider()
    st.caption("Green = compliant · red = violation · grey = PPE not visible · amber = vehicle exclusion zone.")
    st.caption("Code and evaluation: [github.com/harshit-3/minesafe-vision]"
               "(https://github.com/harshit-3/minesafe-vision)")

cfg = SiteConfig(required_ppe=tuple(ppe), vehicle_margin=margin)

st.title("MineSafe Vision")
st.markdown("PPE compliance and vehicle-proximity monitoring for site cameras: fine-tuned **YOLO26s** + a "
            "high-resolution **head classifier**, served with ONNX Runtime (no PyTorch).")
if not ON_GPU:
    st.caption("This free demo runs on a 2-core shared CPU, so the live feed is annotated at a few frames per second "
               "(newest frame first, stale frames dropped). On a laptop RTX 4070 the same pipeline runs at 26 FPS.")
LIVE_FPS = 15 if ON_GPU else 3  # expected processing rate, used by the debouncer's frame-based timing
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}


def show_image_result(bgr: np.ndarray):
    _, workers, ann = process_image(DET, bgr, cfg, HEAD)
    s = summarise(workers)
    c1, c2 = st.columns([3, 2])
    c1.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), width="stretch")
    with c2:
        st.markdown(f"**{s['workers']} workers** · {s['compliant']} fully compliant · "
                    f"**{s['violations']} with violations**")
        if s["details"]:
            st.dataframe(pd.DataFrame([{"worker": i + 1, **d["ppe"], "violations": ", ".join(d["violations"]) or "-"}
                                       for i, d in enumerate(s["details"])]), hide_index=True)


tab_live, tab_vid, tab_img = st.tabs(["Live camera", "Recorded video", "Image"])

with tab_img:
    up = st.file_uploader("Site photo", type=["jpg", "jpeg", "png"])
    names = [p.stem for p in EXAMPLES]
    pick = st.radio("…or try a held-out test image", names, horizontal=True) if not up else None
    if up:
        bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(str(EXAMPLES[names.index(pick)]))
    if bgr is None:
        st.error("Could not read that image.")
    else:
        show_image_result(bgr)

with tab_vid:
    st.caption("For reviewing footage after an incident. Monitoring itself runs on the live feed.")
    vup = st.file_uploader("Site video", type=["mp4", "mov", "avi", "mkv"])
    run = st.button("Analyse demo clip" if not vup else "Analyse video", type="primary")
    if run:
        if vup:
            tmp = Path(tempfile.mkdtemp()) / vup.name
            tmp.write_bytes(vup.getvalue())
            src = str(tmp)
        else:
            src = str(DEMO_VIDEO)
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = 1 if ON_GPU else max(1, round(fps / 3))
        proc = LiveProcessor(DET, HEAD, cfg, fps / stride)  # debouncer uses the effective frame rate
        c1, c2 = st.columns([3, 2])
        frame_slot, table_slot = c1.empty(), c2.empty()
        bar = st.progress(0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        limit = min(total, int(MAX_VIDEO_S * fps))
        idx, last = 0, 0.0
        while idx < limit:
            for _ in range(stride - 1):
                cap.grab()
                idx += 1
            ok, frame = cap.read()
            idx += 1
            if not ok:
                break
            out, _, new = proc.process(frame)
            if new or time.perf_counter() - last > 0.15:
                last = time.perf_counter()
                frame_slot.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), width="stretch")
                ev = pd.DataFrame(proc.events[::-1])[EVENT_COLS] if proc.events else pd.DataFrame(columns=EVENT_COLS)
                table_slot.dataframe(ev, hide_index=True, width="stretch")
                bar.progress(min(idx / limit, 1.0))
        cap.release()
        bar.progress(1.0)
        st.success(f"Done: {len(proc.events)} alert(s). Each alert fires once per tracked worker and violation, "
                   "only after it persists, not once per frame.")
        if total > limit:
            st.caption(f"Public demo analyses the first {MAX_VIDEO_S} s only.")



class LiveState:
    """Shared between the Streamlit script (config, alert table) and the WebRTC frame thread."""

    def __init__(self):
        self.lock, self.proc, self.key = threading.Lock(), None, None

    def configure(self, key, cfg):
        with self.lock:
            if key != self.key:
                self.proc, self.key = LiveProcessor(DET, HEAD, cfg, fps=LIVE_FPS), key

    def callback(self, frame):
        import av

        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            out, _, _ = self.proc.process(img)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


def demo_camera():
    from aiortc.contrib.media import MediaPlayer

    return MediaPlayer(str(DEMO_VIDEO), loop=True)  # plays at the clip's own frame rate, like a CCTV feed


with tab_live:
    source = st.radio("Camera", ["Demo site camera (streamed from the server)", "Your webcam"], horizontal=True)
    webcam = source == "Your webcam"
    st.caption("Press **START**. Every frame is checked as it arrives; an alert fires when the same tracked worker "
               "keeps a violation, once per worker, not once per frame." +
               (" Your browser will ask for camera access; video goes only to this demo's server and is not stored."
                if webcam else ""))
    if "live" not in st.session_state:
        st.session_state.live = LiveState()
    live = st.session_state.live
    live.configure((tuple(ppe), margin, webcam), cfg)
    c1, c2 = st.columns([3, 2])
    with c1:
        ctx = webrtc_streamer(
            key="live-webcam" if webcam else "live-demo",
            mode=WebRtcMode.SENDRECV if webcam else WebRtcMode.RECVONLY,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": {"width": 640, "height": 480} if webcam else True, "audio": False},
            player_factory=None if webcam else demo_camera,
            video_frame_callback=live.callback,
            async_processing=True,  # drop stale frames instead of queueing them
        )
    with c2:
        st.markdown("**Alerts**")
        slot = st.empty()
        while ctx.state.playing:
            with live.lock:
                events = list(live.proc.events)
            slot.dataframe(pd.DataFrame(events[::-1])[EVENT_COLS] if events else pd.DataFrame(columns=EVENT_COLS),
                           hide_index=True, width="stretch")
            time.sleep(1)
        slot.caption("Alerts appear here while the stream is running.")
