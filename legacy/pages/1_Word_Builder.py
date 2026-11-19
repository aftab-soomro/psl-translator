import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st
from gtts import gTTS

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.predict import predict
from src.letters import LETTER_MAP

LOW_CONFIDENCE = 0.40
AUDIO_DIR = Path(__file__).resolve().parent.parent / "app" / "audio_cache"

st.set_page_config(page_title="Word Builder", page_icon="✍️")
st.title("Word Builder")
st.caption("Make the sign for each letter, take a photo, pick the right letter and press 'Add letter'.")

# --- memory: list of letters and the camera counter ---
if "letters" not in st.session_state:
    st.session_state.letters = []
if "cam_n" not in st.session_state:
    st.session_state.cam_n = 0


def add_letter(letter):
    st.session_state.letters.append(letter)
    st.session_state.cam_n += 1          # so the camera is empty, ready for the next letter


def backspace():
    st.session_state.letters = st.session_state.letters[:-1]


def add_space():
    st.session_state.letters.append(" ")


def clear_all():
    st.session_state.letters = []
    st.session_state.cam_n += 1


def speak(text):
    """Returns the mp3 of an Urdu word. Checks the cache first, otherwise uses gTTS (needs internet)."""
    AUDIO_DIR.mkdir(exist_ok=True)
    f = AUDIO_DIR / (hashlib.md5(text.encode("utf-8")).hexdigest() + ".mp3")
    if f.exists():
        return f.read_bytes(), None
    try:
        gTTS(text=text, lang="ur").save(str(f))
        return f.read_bytes(), None
    except Exception as e:
        if f.exists():
            f.unlink()                   # so no half-written file is left
        return None, str(e)


# --- the word so far (Urdu, right-to-left) ---
spoken = "".join(st.session_state.letters).strip()
st.markdown(
    f"<div dir='rtl' style='font-size:64px; text-align:center; padding:12px; "
    f"border:1px solid #888; border-radius:8px;'>{spoken or '—'}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.button("⌫ Backspace", on_click=backspace, width="stretch")
c2.button("␣ Space", on_click=add_space, width="stretch")
c3.button("🗑 Clear", on_click=clear_all, width="stretch")

if st.button("🔊 Speak word", disabled=not spoken, width="stretch"):
    audio, err = speak(spoken)
    if audio:
        st.audio(audio, format="audio/mp3", autoplay=True)
    else:
        st.warning("Could not make the audio. It needs internet, or this word was not cached before.")
        st.caption(f"Reason: {err}")

st.divider()

# --- camera and prediction ---
snapshot = st.camera_input("Take a photo of the sign", key=f"cam_{st.session_state.cam_n}")

if snapshot is not None:
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "snapshot.jpg"
        img_path.write_bytes(snapshot.getvalue())
        result = predict(img_path)

    if result["error"]:
        st.warning(result["error"])
    else:
        top3 = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        probs = dict(top3)
        names = [name for name, _ in top3]

        if result["confidence"] < LOW_CONFIDENCE:
            st.warning("The model is not sure. Pick the right letter below or take the photo again.")

        choice = st.radio(
            "Pick the right letter (first = model's answer):",
            names,
            format_func=lambda n: f"{LETTER_MAP.get(n, n)}   {n}   {probs[n]:.0%}",
            key=f"pick_{st.session_state.cam_n}",
            horizontal=True,
        )
        st.button("➕ Add letter", on_click=add_letter, args=(LETTER_MAP.get(choice, choice),),
                  type="primary", width="stretch")