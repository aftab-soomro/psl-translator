"""
live_app.py  -  PSL Live Translator (single page)

Chalane ka tareeqa:  python -m streamlit run app/live_app.py
"""

import hashlib
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.letters import LETTER_MAP
from src.live import (DYNAMIC_LOG, DYNAMIC_MAP, DYNAMIC_MODEL_PATH, DynamicRecorder,
                      LiveHandTracker, LivePredictor, WordState, make_callback, pick_names)

AUDIO_DIR = ROOT / "app" / "audio_cache"
NONE_LABEL = "kuch nahi"
TEST_OPTIONS = ["(test nahi)"] + list(DYNAMIC_MAP) + [NONE_LABEL]

st.set_page_config(page_title="PSL Live Translator", page_icon="🖐️", layout="wide")


@st.cache_resource
def get_tracker():
    return LiveHandTracker()


@st.cache_resource
def get_predictor():
    return LivePredictor()


@st.cache_resource
def get_dynamic_model():
    return joblib.load(DYNAMIC_MODEL_PATH)


if "word" not in st.session_state:
    st.session_state.word = WordState()
if "recorder" not in st.session_state:
    st.session_state.recorder = DynamicRecorder(get_dynamic_model())
state = st.session_state.word
recorder = st.session_state.recorder


def urdu(name):
    if name == " ":
        return " "
    if name in DYNAMIC_MAP:
        return DYNAMIC_MAP[name]
    v = LETTER_MAP.get(name, name)
    return v if isinstance(v, str) else str(v[0])


def speak(text):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    f = AUDIO_DIR / (hashlib.md5(text.encode("utf-8")).hexdigest() + ".mp3")
    if not f.exists():
        from gtts import gTTS
        gTTS(text=text, lang="ur").save(str(f))
    return f.read_bytes()


def start_dynamic():
    label = st.session_state.get("dyn_true", TEST_OPTIONS[0])
    recorder.start(None if label == TEST_OPTIONS[0] else label)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    st.subheader("Static letters")
    state.hold_seconds = st.slider("Harf jodne ke liye kitni der rukna hai (second)", 0.5, 2.5, 1.0, 0.1)
    state.min_conf = st.slider("Static: kam az kam yaqeen", 0.30, 0.95, 0.60, 0.05)
    state.smooth_seconds = st.slider("Smoothing (second): pichle itne waqt ki probabilities ka average", 0.1, 1.0, 0.4, 0.1)
    state.show_points = st.checkbox("21 points aur lines dikhao", value=True)
    state.show_dist = st.checkbox("Faasle dikhao (peela: wrist→tip, neela: tip→tip)", value=True)

    st.subheader("Dynamic letters")
    recorder.record_seconds = st.slider("Recording kitni lambi (second)", 2.0, 4.0, 3.0, 0.5)
    recorder.min_conf = st.slider("Dynamic: kam az kam yaqeen", 0.30, 0.95, 0.60, 0.05)
    recorder.aspect_fix = st.checkbox("16:9 fix (training videos jaisa)", value=True)
    st.caption("Training ke videos 3 second ke aur chaure (16:9) the. Right haath use karo.")

# Tuey/Daal ke do buttons: chota number (1, 2) button ke top-left mein
st.markdown("""
<style>
.block-container { padding-top: 4rem; }
.st-key-pick1 button, .st-key-pick2 button { position: relative; height: 58px; font-size: 24px; }
.st-key-pick1 button::before, .st-key-pick2 button::before {
    position: absolute; top: 4px; left: 8px; font-size: 13px; font-weight: 700; line-height: 1;
    padding: 3px 7px; border-radius: 5px; background: rgba(128,128,128,0.35);
}
.st-key-pick1 button::before { content: "1"; }
.st-key-pick2 button::before { content: "2"; }
</style>
""", unsafe_allow_html=True)

pending_at_start = state.snapshot()["pending"]

left, right = st.columns([3, 2])

with left:
    try:
        pick_area = st.container(height=84, border=False)      # camera ke upar khali jagah
    except TypeError:
        pick_area = st.container()
    with pick_area:
        c1, c2 = st.columns(2)
        if pending_at_start:
            names = pick_names(pending_at_start)
            c1.button(f"{urdu(names[0])}  {names[0]}", key="pick1", on_click=state.pick_pending,
                      args=(names[0],), use_container_width=True)
            c2.button(f"{urdu(names[1])}  {names[1]}", key="pick2", on_click=state.pick_pending,
                      args=(names[1],), use_container_width=True)
        else:
            c1.empty()        # purane buttons isi jagah par overwrite ho jayen (warna dhundhle reh jate hain)
            c2.empty()
    ctx = webrtc_streamer(
        key="psl-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=make_callback(state, get_tracker(), get_predictor(), recorder),
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

with right:
    st.subheader("Lafz")
    word_box = st.empty()
    cur_box = st.empty()
    hold_box = st.empty()
    pend_box = st.empty()

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

    st.divider()
    st.subheader("Dynamic harf")
    st.selectbox("Test ke liye: asal mein kaunsa harf kar rahe ho?", TEST_OPTIONS, key="dyn_true",
                 help=f"'{NONE_LABEL}' = jaan boojh kar koi dynamic harf NAHI (sirf haath hilao / static badlo).")
    st.button("🎬 Dynamic harf record karo", on_click=start_dynamic, type="primary", use_container_width=True)
    dyn_box = st.empty()

with st.expander("📊 Dynamic test ka hisaab (results/live_dynamic_log.csv)"):
    st.button("🔄 Taaza karo")
    if DYNAMIC_LOG.exists():
        df = pd.read_csv(DYNAMIC_LOG)
        df["true"] = df["true"].fillna("")
        df["pred"] = df["pred"].fillna("(haath kam)")
        tl = df[df["true"].isin(list(DYNAMIC_MAP))].copy()
        if len(tl):
            tl["sahi"] = tl["true"] == tl["pred"]
            tl["qubool"] = tl["conf"] >= recorder.min_conf
            st.markdown(f"**Kul accuracy: {tl['sahi'].mean() * 100:.0f}%** ({tl['sahi'].sum()}/{len(tl)} koshishen)")
            st.markdown(f"Jo qubool hue (yaqeen ≥ {recorder.min_conf:.2f}) unme sahi: "
                        f"{tl[tl['qubool']]['sahi'].mean() * 100 if tl['qubool'].any() else 0:.0f}% "
                        f"({tl['qubool'].sum()} qubool)")
            st.dataframe(tl.groupby("true").agg(koshishen=("sahi", "size"), sahi=("sahi", "sum"),
                                                avg_yaqeen=("conf", "mean")).round(2))
            st.markdown("Confusion (row = asal, column = model ka jawab):")
            st.dataframe(pd.crosstab(tl["true"], tl["pred"]))
            if tl["aspect_fix"].nunique() > 1:
                st.markdown("16:9 fix ke saath (1) aur baghair (0):")
                st.dataframe(tl.groupby("aspect_fix")["sahi"].agg(["size", "mean"]).round(2))
        nn = df[df["true"] == NONE_LABEL]
        if len(nn):
            bad = int((nn["conf"] >= recorder.min_conf).sum())
            st.markdown(f"**'{NONE_LABEL}'**: {len(nn)} koshishon mein se **{bad}** baar fazool dynamic harf "
                        f"jur jata (yaqeen ≥ {recorder.min_conf:.2f}). Average yaqeen: {nn['conf'].mean():.2f}")
        if not len(tl) and not len(nn):
            st.caption("Abhi test wali koi recording nahi. Upar asal harf chun kar record karo.")
    else:
        st.caption("Abhi koi recording nahi.")


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
    if recorder.busy():
        cur_box.markdown("Abhi ka andaza:  \n*dynamic recording chal rahi hai…*")
    elif s["top3"]:
        lines = [f"**{urdu(n)} {n}** — {p * 100:.0f}%" for n, p in s["top3"]]
        cur_box.markdown("Abhi ka andaza:  \n" + "  \n".join(lines))
    else:
        cur_box.markdown("Abhi ka andaza:  \n*haath nahi mila*")
    hold_box.progress(float(s["hold"]), text="Rukte raho… harf jurne wala hai" if 0 < s["hold"] < 1 else "Hold meter")
    if s["pending"]:
        n1, n2 = pick_names(s["pending"])
        pend_box.warning(f"⚠️ {urdu(n1)} {n1} ya {urdu(n2)} {n2}? Camera ke upar button dabao, "
                         f"ya keyboard par **1** / **2** dabao.")
    else:
        pend_box.empty()
    if (s["pending"] is not None) != (pending_at_start is not None):
        st.rerun()                                              # buttons aane / jaane ke liye page dobara banao

    r = recorder.result
    if r is None:
        dyn_box.caption("Abhi koi dynamic recording nahi hui.")
    elif not r["ok"]:
        dyn_box.warning(f"Haath kam mila ({r['n_hand']}/{r['n_frames']} frames). Dobara koshish karo.")
    else:
        ok = r["conf"] >= recorder.min_conf
        second = f" · dusra: {r['top2'][1][0]} {r['top2'][1][1] * 100:.0f}%" if len(r["top2"]) > 1 else ""
        verdict = ""
        if r["true"] in DYNAMIC_MAP:
            verdict = " ✅ sahi" if r["pred"] == r["true"] else f" ❌ ghalat (asal: {r['true']})"
        elif r["true"] == NONE_LABEL:
            verdict = " ⚠️ fazool harf jur jata" if ok else " ✅ theek, nahi jura"
        dyn_box.markdown(
            f"Aakhri: **{urdu(r['pred'])} {r['pred']} {r['conf'] * 100:.0f}%**"
            f"{'' if ok else ' (kam yaqeen, nahi jura)'}{second}{verdict}  \n"
            f"<small>{r['n_frames']} frames, haath {r['n_hand']} mein</small>",
            unsafe_allow_html=True,
        )


# keyboard: 1 / 2 dabane se upar wale button par click ho jaye
components.html("""
<script>
const doc = window.parent.document;
if (window.parent.__pslKeys) doc.removeEventListener('keydown', window.parent.__pslKeys);
window.parent.__pslKeys = function (e) {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (e.key !== '1' && e.key !== '2') return;
  const b = doc.querySelector('.st-key-pick' + e.key + ' button');
  if (b && b.offsetWidth > 0) { b.click(); e.preventDefault(); }   // sirf nazar aane wala button
};
doc.addEventListener('keydown', window.parent.__pslKeys);
</script>
""", height=0)

render()
while ctx.state.playing:
    render()
    time.sleep(0.25)