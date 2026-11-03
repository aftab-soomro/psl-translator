"""
live_app.py  -  PSL Live Translator (single page)

Chalane ka tareeqa:  python -m streamlit run app/live_app.py
"""

import hashlib
import sys
import time
from pathlib import Path

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.letters import LETTER_MAP
from src.live import LiveHandTracker, LivePredictor, WordState, make_callback

AUDIO_DIR = ROOT / "app" / "audio_cache"

st.set_page_config(page_title="PSL Live Translator", page_icon="🖐️", layout="wide")


@st.cache_resource
def get_tracker():
    return LiveHandTracker()


@st.cache_resource
def get_predictor():
    return LivePredictor()


if "word" not in st.session_state:
    st.session_state.word = WordState()
state = st.session_state.word


def urdu(name):
    if name == " ":
        return " "
    v = LETTER_MAP.get(name, name)
    return v if isinstance(v, str) else str(v[0])


def speak(text):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    f = AUDIO_DIR / (hashlib.md5(text.encode("utf-8")).hexdigest() + ".mp3")
    if not f.exists():
        from gtts import gTTS
        gTTS(text=text, lang="ur").save(str(f))
    return f.read_bytes()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    state.hold_seconds = st.slider("Harf jodne ke liye kitni der rukna hai (second)", 0.5, 2.5, 1.0, 0.1)
    state.min_conf = st.slider("Kam az kam yaqeen", 0.30, 0.95, 0.60, 0.05)
    state.show_points = st.checkbox("21 points aur lines dikhao", value=True)
    state.show_dist = st.checkbox("Faasle dikhao (peela: wrist→tip, neela: tip→tip)", value=True)
    st.caption("Faasle palm ki lambai (wrist→point 9) ki unit mein hain, isliye haath door/qareeb karne se nahi badalte.")

st.title("🖐️ PSL Live Translator")
st.caption("START dabao, haath camera ke saamne rakho. Harf 1 second tak ek hi rahe to khud lafz mein jud jayega.")

left, right = st.columns([3, 2])

with left:
    ctx = webrtc_streamer(
        key="psl-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=make_callback(state, get_tracker(), get_predictor()),
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

with right:
    st.subheader("Lafz")
    word_box = st.empty()
    cur_box = st.empty()
    hold_box = st.empty()

    b1, b2, b3, b4 = st.columns(4)
    b1.button("␣ Space", on_click=state.add_space, use_container_width=True)
    b2.button("⌫ Mitao", on_click=state.backspace, use_container_width=True)
    b3.button("🗑 Saaf", on_click=state.clear, use_container_width=True)
    speak_clicked = b4.button("🔊 Sunao", use_container_width=True)

    if speak_clicked:
        text = "".join(urdu(n) for n in state.snapshot()["letters"]).strip()
        if text:
            try:
                st.audio(speak(text), format="audio/mp3")
            except Exception as e:                      # internet na ho to
                st.warning(f"Awaaz nahi ban saki (internet?): {e}")


def render():
    s = state.snapshot()
    word = "".join(urdu(n) for n in s["letters"])
    word_box.markdown(
        "<div dir='rtl' style=\"font-size:68px; line-height:1.6; min-height:110px; padding:6px 16px;"
        "border:2px solid #8884; border-radius:12px; "
        "font-family:'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif;\">"
        f"{word if word else '…'}</div>",
        unsafe_allow_html=True,
    )
    if s["top3"]:
        lines = [f"**{urdu(n)} {n}** — {p * 100:.0f}%" for n, p in s["top3"]]
        cur_box.markdown("Abhi ka andaza:  \n" + "  \n".join(lines))
    else:
        cur_box.markdown("Abhi ka andaza:  \n*haath nahi mila*")
    hold_box.progress(float(s["hold"]), text="Rukte raho… harf jurne wala hai" if 0 < s["hold"] < 1 else "Hold meter")


render()
while ctx.state.playing:
    render()
    time.sleep(0.25)