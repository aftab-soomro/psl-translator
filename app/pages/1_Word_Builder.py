import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.predict import predict
from src.letters import LETTER_MAP

LOW_CONFIDENCE = 0.40

st.set_page_config(page_title="Word Builder", page_icon="✍️")
st.title("Word Builder")
st.caption("Har harf ka sign banao, photo lo, sahi harf chuno aur 'Harf jodo' dabao.")

# --- yaaddasht: harfon ki list aur camera ka counter ---
if "letters" not in st.session_state:
    st.session_state.letters = []
if "cam_n" not in st.session_state:
    st.session_state.cam_n = 0


def add_letter(letter):
    st.session_state.letters.append(letter)
    st.session_state.cam_n += 1          # camera khali ho jaye, agla harf lene ke liye


def backspace():
    st.session_state.letters = st.session_state.letters[:-1]


def add_space():
    st.session_state.letters.append(" ")


def clear_all():
    st.session_state.letters = []
    st.session_state.cam_n += 1


# --- abhi tak ka lafz (Urdu right-to-left) ---
word = "".join(st.session_state.letters).strip() or "—"
st.markdown(
    f"<div dir='rtl' style='font-size:64px; text-align:center; padding:12px; "
    f"border:1px solid #888; border-radius:8px;'>{word}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.button("⌫ Backspace", on_click=backspace, width="stretch")
c2.button("␣ Space", on_click=add_space, width="stretch")
c3.button("🗑 Clear", on_click=clear_all, width="stretch")

st.divider()

# --- camera aur prediction ---
snapshot = st.camera_input("Sign ki photo lo", key=f"cam_{st.session_state.cam_n}")

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
            st.warning("Model ko pakka yaqeen nahi. Neeche se sahi harf chun lein ya dobara photo lein.")

        choice = st.radio(
            "Sahi harf chuno (pehla = model ka jawab):",
            names,
            format_func=lambda n: f"{LETTER_MAP.get(n, n)}   {n}   {probs[n]:.0%}",
            key=f"pick_{st.session_state.cam_n}",
            horizontal=True,
        )
        st.button("➕ Harf jodo", on_click=add_letter, args=(LETTER_MAP.get(choice, choice),),
                  type="primary", width="stretch")