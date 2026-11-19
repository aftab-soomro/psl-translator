import sys
import time
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.live import LiveHandTracker, draw_hand

st.set_page_config(page_title="Live test", page_icon="🖐️", layout="wide")
st.title("Live test: 21 points")
st.caption("Press START, allow the camera, and show your hand.")


@st.cache_resource
def get_tracker():
    return LiveHandTracker()


tracker = get_tracker()
state = {"t": time.time(), "fps": 0.0}


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)                              # mirror view (selfie)
    lm = tracker.detect(img)
    if lm is not None:
        draw_hand(img, lm)

    now = time.time()
    state["fps"] = 0.9 * state["fps"] + 0.1 * (1.0 / max(now - state["t"], 1e-6))
    state["t"] = now
    msg = f"FPS {state['fps']:.0f} | " + ("hand found" if lm is not None else "no hand")
    cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="live-test",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=callback,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)