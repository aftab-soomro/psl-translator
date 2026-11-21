"""
live_app.py  -  PSL Live Translator (single page)

How to run:  python -m streamlit run app/live_app.py

Layout (one screen, no scrolling):
  left  : space above the camera for the Tuey/Daal 1/2 buttons, then the live camera
  right : header (Settings, Stats) / Word card / Live recognition card / Dynamic letter card
  sidebar: Practice (PSL Coach: pick a letter, the coach says which finger to fix) / Letter guide
"""

import base64
import hashlib
import sys
import time
from pathlib import Path

import cv2
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.coach import FINGERS, Z_TIP, Coach, CoachReference
from src.letters import LETTER_MAP
from src.live import (AUTO_LOG, DYNAMIC_LOG, DYNAMIC_MAP, DYNAMIC_MODEL_PATH, GROUPS, DynamicRecorder,
                      LiveHandTracker, LivePredictor, WordState, dynamic_verdict, make_callback, pick_names)

AUDIO_DIR = ROOT / "app" / "audio_cache"
GUIDE_DIR = ROOT / "app" / "assets" / "letters"
DYN_GUIDE_DIR = ROOT / "app" / "assets" / "dynamic"       # optional: one small GIF per dynamic letter (make_dynamic_guide.py)
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"      # the coach's reference angles come from here
PRACTICE_OFF = "Off"
NONE_LABEL = "no gesture"
RESULT_SHOW_SECONDS = 8      # a dynamic result stays in the Dynamic card this long, then the hint comes back
TEST_OPTIONS = ["(no test)"] + list(DYNAMIC_MAP) + [NONE_LABEL]

GREEN, ORANGE, BLUE, RED, GRAY = "#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#94a3b8"

st.set_page_config(page_title="PSL Live Translator", page_icon="🖐️", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ style (works in both dark and light themes)
st.markdown("""
<style>
.block-container { padding-top: 3rem; padding-bottom: 0.3rem; max-width: 1560px; }
[data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.15rem 0.2rem; }

/* Letter guide (sidebar) */
.gtitle { font-size: 18px; font-weight: 700; margin: 0; }
.gsub { font-size: 12px; opacity: .65; margin: 2px 0 10px 0; }
.ggrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.gcard { border-radius: 10px; padding: 5px 5px 6px 5px; text-align: center;
         background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25); }
.gcard img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; border-radius: 7px; display: block; }
.gcard .gph { width: 100%; aspect-ratio: 1 / 1; border-radius: 7px; display: flex; align-items: center;
              justify-content: center; font-size: 12px; opacity: .6; background: rgba(128,128,128,.12); }
.gcard .gu { font-size: 26px; line-height: 1.5; font-family: 'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif; }
.gcard .gn { font-size: 12px; font-weight: 600; opacity: .85; margin-top: -2px; }
.gcard.dyn { border-color: rgba(245,158,11,.55); }
.gcard.dyn .gph { color: #f59e0b; opacity: 1; background: rgba(245,158,11,.10); text-align: center; line-height: 1.3; }
.st-key-leftcol { max-width: 680px; }
.st-key-choicebar { min-height: 66px; }

/* The two Tuey/Daal buttons, with a small number at the top left */
.st-key-pick1 button, .st-key-pick2 button {
    position: relative; height: 58px; font-size: 24px; font-weight: 600; border-width: 2px;
}
.st-key-pick1 button { border-color: #3b82f6; }
.st-key-pick2 button { border-color: #f59e0b; }
.st-key-pick1 button::before, .st-key-pick2 button::before {
    position: absolute; top: 4px; left: 8px; font-size: 13px; font-weight: 700; line-height: 1;
    padding: 3px 7px; border-radius: 5px; color: #fff;
}
.st-key-pick1 button::before { content: "1"; background: #3b82f6; }
.st-key-pick2 button::before { content: "2"; background: #f59e0b; }

.stButton button { border-radius: 10px; font-weight: 600; }

.ctitle { font-size: 11.5px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
          opacity: .6; margin: 0 0 6px 0; }
.brand { font-size: 20px; font-weight: 700; line-height: 1.1; margin-top: 4px; }
.brand small { display: block; font-size: 12px; font-weight: 500; opacity: .6; margin-top: 3px; }

.chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; margin: 0 6px 6px 0;
        border-radius: 999px; font-size: 12px; font-weight: 600; color: var(--c);
        background: color-mix(in srgb, var(--c) 14%, transparent);
        border: 1px solid color-mix(in srgb, var(--c) 40%, transparent); }
.chip i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); display: inline-block; }

.word { direction: rtl; display: flex; align-items: center; font-size: 56px; line-height: 1.5; min-height: 84px; margin-bottom: 4px;
        padding: 0 16px; border-radius: 12px; background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.28); overflow-x: auto; white-space: nowrap;
        font-family: 'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif; }
.word .ph { opacity: .35; font-size: 18px; direction: ltr; font-family: inherit; }

.live { display: flex; gap: 16px; align-items: center; }
.glyph { flex: 0 0 96px; height: 96px; border-radius: 16px; display: flex; align-items: center;
         justify-content: center; font-size: 46px; direction: rtl; background: rgba(128,128,128,.08);
         border: 2px solid var(--c); color: var(--c);
         font-family: 'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif; }
.lmeta { flex: 1 1 130px; min-width: 110px; }
.lname { font-size: 24px; font-weight: 700; line-height: 1.1; }
.lconf { font-size: 13px; opacity: .7; margin-top: 3px; }
.bars { flex: 1.5 1 200px; min-width: 190px; }
.brow { display: flex; align-items: center; gap: 8px; font-size: 12.5px; margin: 3px 0; }
.bl { flex: 0 0 82px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.btrack { flex: 1; height: 8px; border-radius: 6px; background: rgba(128,128,128,.22); overflow: hidden; }
.bfill { height: 100%; border-radius: 6px; background: var(--c); }
.bp { flex: 0 0 34px; text-align: right; opacity: .75; font-variant-numeric: tabular-nums; }
.hold { height: 8px; border-radius: 6px; background: rgba(128,128,128,.22); margin-top: 12px; overflow: hidden; }
.hfill { height: 100%; border-radius: 6px; background: var(--c); transition: width .15s linear; }
.hnote { font-size: 12px; opacity: .65; margin-top: 4px; min-height: 16px; }
.pend { margin-top: 8px; padding: 7px 12px; border-radius: 10px; font-size: 13px; font-weight: 600;
        color: #f59e0b; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.4); }
.dres { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 4px 0 6px 0; font-size: 14px; min-height: 34px; }
.dres .dl { font-size: 30px; line-height: 1; font-family: 'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif; }
.dres small { opacity: .6; }
.cphoto { flex: 0 0 96px; width: 96px; height: 96px; border-radius: 16px; object-fit: cover; border: 3px solid var(--c); }
.tips { margin: 8px 0 0 20px; padding: 0; font-size: 16px; font-weight: 600; }
.tips li { margin: 3px 0; }
.fchips { margin-top: 8px; }
.dglyphs { font-size: 24px; font-weight: 400; margin-left: 12px;
           font-family: 'Noto Nastaliq Urdu','Jameel Noto Nastaleeq','Urdu Typesetting',serif; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_tracker():
    return LiveHandTracker()


@st.cache_resource
def get_predictor():
    return LivePredictor()


@st.cache_resource
def get_dynamic_model():
    return joblib.load(DYNAMIC_MODEL_PATH)


@st.cache_resource
def get_coach_reference():
    return CoachReference.from_csv(TRAIN_CSV)


if "word" not in st.session_state:
    st.session_state.word = WordState()
if "recorder" not in st.session_state:
    st.session_state.recorder = DynamicRecorder(get_dynamic_model())
if "coach" not in st.session_state:
    st.session_state.coach = Coach(get_coach_reference())
state = st.session_state.word
recorder = st.session_state.recorder
coach = st.session_state.coach
STATIC_LETTERS = [n for n in LETTER_MAP if n not in DYNAMIC_MAP]
PRACTICE_LETTERS = STATIC_LETTERS + list(DYNAMIC_MAP)      # 36 static letters, then the 4 moving ones


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
    if recorder.practice_label:                        # Practice: the true letter is the one being practised
        recorder.start(recorder.practice_label)
        return
    label = st.session_state.get("dyn_true", TEST_OPTIONS[0])
    recorder.start(None if label == TEST_OPTIONS[0] else label)


def chip(text, color, dot=True):
    return f"<span class='chip' style='--c:{color}'>{'<i></i>' if dot else ''}{text}</span>"


def container(**kw):
    try:
        return st.container(**kw)
    except TypeError:                    # older Streamlit: no key / border option
        kw.pop("key", None)
        kw.pop("border", None)
        return st.container()


@st.cache_data(show_spinner=False)
def dyn_gif_uri(name):
    """Small animated GIF showing how the dynamic letter is made ('' if make_dynamic_guide.py was not run)."""
    f = DYN_GUIDE_DIR / f"{name}.gif"
    return ("data:image/gif;base64," + base64.b64encode(f.read_bytes()).decode()) if f.exists() else ""


@st.cache_data(show_spinner=False)
def guide_cards_html():
    """A small card for each letter in the sidebar: photo + Urdu letter + Roman name."""
    cards = []
    for name in LETTER_MAP:
        if name in DYNAMIC_MAP:
            continue
        f = GUIDE_DIR / f"{name}.jpg"
        img = cv2.imread(str(f)) if f.exists() else None
        if img is not None:
            img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            pic = "<img src='data:image/jpeg;base64,%s'>" % base64.b64encode(buf).decode()
        else:
            pic = "<div class='gph'>no photo</div>"
        cards.append(f"<div class='gcard'>{pic}<div class='gu'>{urdu(name)}</div><div class='gn'>{name}</div></div>")
    for name in DYNAMIC_MAP:
        gif = dyn_gif_uri(name)
        pic = f"<img src='{gif}'>" if gif else "<div class='gph'>🎬<br>moving<br>sign</div>"
        cards.append(f"<div class='gcard dyn'>{pic}<div class='gu'>{DYNAMIC_MAP[name]}</div><div class='gn'>{name}</div></div>")
    return "<div class='ggrid'>" + "".join(cards) + "</div>"


def next_practice():
    """'Next letter' button: next static letter, then the 4 dynamic ones, then start again."""
    opts = [PRACTICE_OFF] + PRACTICE_LETTERS
    cur = st.session_state.get("practice_sel", PRACTICE_OFF)
    i = opts.index(cur) if cur in opts else 0
    st.session_state.practice_sel = opts[i % (len(opts) - 1) + 1]


@st.cache_data(show_spinner=False)
def guide_photo_uri(name, px=192):
    f = GUIDE_DIR / f"{name}.jpg"
    img = cv2.imread(str(f)) if f.exists() else None
    if img is None:
        return ""
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


with st.sidebar:
    st.markdown("<p class='gtitle'>🎯 Practice (Coach)</p>"
                "<p class='gsub'>Pick a letter and copy the photo. Your fingers turn green, orange or red, "
                "and the coach tells you which finger to fix. For the moving letters (🎬) press Record and do the "
                "gesture; the app says if it was right. Nothing is added to the word while you practise.</p>",
                unsafe_allow_html=True)
    practice = st.selectbox("Letter to practise", [PRACTICE_OFF] + PRACTICE_LETTERS, key="practice_sel",
                            format_func=lambda n: n if n == PRACTICE_OFF else
                            f"{urdu(n)}  {n}" + ("  🎬 moving" if n in DYNAMIC_MAP else ""),
                            label_visibility="collapsed")
    st.button("Next letter ▶", on_click=next_practice, use_container_width=True)
    coach.set_target(practice if practice in STATIC_LETTERS else None)
    recorder.set_practice(practice if practice in DYNAMIC_MAP else None)
    st.markdown("<p class='gtitle'>📖 Letter guide</p>"
                "<p class='gsub'>Find your letter and copy the sign in the photo. "
                "Letters with an orange border are made with movement.</p>",
                unsafe_allow_html=True)
    st.markdown(guide_cards_html(), unsafe_allow_html=True)


pending_at_start = state.snapshot()["pending"]

left, right = st.columns([6, 7], gap="large")

# ------------------------------------------------------------------ left: choice bar + camera
with left:
    with container(key="leftcol"):
        with container(key="choicebar"):                        # empty space above the camera
            c1, c2 = st.columns(2)
            if pending_at_start:
                names = pick_names(pending_at_start)
                c1.button(f"{urdu(names[0])}  {names[0]}", key="pick1", on_click=state.pick_pending,
                          args=(names[0],), use_container_width=True)
                c2.button(f"{urdu(names[1])}  {names[1]}", key="pick2", on_click=state.pick_pending,
                          args=(names[1],), use_container_width=True)
            else:
                c1.empty()        # so old buttons are overwritten in place (otherwise they stay faded)
                c2.empty()
        ctx = webrtc_streamer(
            key="psl-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=make_callback(state, get_tracker(), get_predictor(), recorder, coach),
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

# ------------------------------------------------------------------ right
with right:
    top_brand, top_set, top_stats = st.columns([5, 2, 2])
    top_brand.markdown("<div class='brand'>🖐️ PSL Live Translator<small>Pakistan Sign Language → Urdu</small></div>",
                       unsafe_allow_html=True)

    with top_set.popover("⚙️ Settings", use_container_width=True):
        st.markdown("**Static letter**")
        state.hold_seconds = st.slider("Hold time to add a letter (seconds)", 0.5, 2.5, 1.0, 0.1)
        state.min_conf = st.slider("Static: minimum confidence", 0.30, 0.95, 0.60, 0.05)
        state.smooth_seconds = st.slider("Smoothing (second)", 0.1, 1.0, 0.4, 0.1,
                                         help="Average of the probabilities over this much recent time")
        state.show_points = st.checkbox("Show 21 points and lines", value=True)
        state.show_dist = st.checkbox("Show distances (yellow: wrist→tip, blue: tip→tip)", value=True)
        st.markdown("**Dynamic letter**")
        recorder.record_seconds = st.slider("Recording length (seconds)", 2.0, 4.0, 3.0, 0.5)
        recorder.min_conf = st.slider("Dynamic: minimum confidence", 0.30, 0.95, 0.60, 0.05)
        recorder.aspect_fix = st.checkbox("16:9 fix (like the training videos)", value=True)
        st.caption("For dynamic letters use your right hand and spread the gesture over the full 3 seconds.")
        st.markdown("**Auto mode** (no button)")
        recorder.auto = st.checkbox("Auto mode on", value=False,
                                    help="Move your hand fast and recording starts by itself; stop and it ends. "
                                         "'AUTO v ..' on the video shows the hand speed.")
        recorder.v_hi = st.slider("Start threshold (speed above)", 0.2, 2.0, 0.8, 0.05)
        recorder.v_lo = st.slider("End threshold (speed below)", 0.1, 1.0, 0.35, 0.05)
        recorder.min_dur = st.slider("Min duration (seconds)", 0.3, 1.5, 0.9, 0.1)
        recorder.min_energy = st.slider("Min motion", 0.3, 3.0, 1.3, 0.1)
        if recorder.v_lo >= recorder.v_hi:
            st.warning("End threshold must be lower than Start threshold, otherwise a gesture never ends cleanly.")

    with top_stats.popover("📊 Stats", use_container_width=True):
        st.markdown("**Dynamic test stats** (`results/live_dynamic_log.csv`)")
        st.button("🔄 Refresh", key="refresh_stats")
        if DYNAMIC_LOG.exists():
            df = pd.read_csv(DYNAMIC_LOG)
            df["true"] = df["true"].fillna("")
            df["pred"] = df["pred"].fillna("(hand missing)")
            tl = df[df["true"].isin(list(DYNAMIC_MAP))].copy()
            if len(tl):
                tl["correct"] = tl["true"] == tl["pred"]
                tl["accepted"] = tl["conf"] >= recorder.min_conf
                st.markdown(f"**Overall accuracy: {tl['correct'].mean() * 100:.0f}%** "
                            f"({tl['correct'].sum()}/{len(tl)} attempts)")
                st.markdown(f"Correct among accepted (confidence ≥ {recorder.min_conf:.2f}): "
                            f"{tl[tl['accepted']]['correct'].mean() * 100 if tl['accepted'].any() else 0:.0f}% "
                            f"({tl['accepted'].sum()} accepted)")
                st.dataframe(tl.groupby("true").agg(attempts=("correct", "size"), correct=("correct", "sum"),
                                                    avg_conf=("conf", "mean")).round(2))
                st.markdown("Confusion (row = true, column = model answer):")
                st.dataframe(pd.crosstab(tl["true"], tl["pred"]))
                if tl["aspect_fix"].nunique() > 1:
                    st.markdown("With 16:9 fix (1) and without (0):")
                    st.dataframe(tl.groupby("aspect_fix")["correct"].agg(["size", "mean"]).round(2))
            nn = df[df["true"] == NONE_LABEL]
            if len(nn):
                bad = int((nn["conf"] >= recorder.min_conf).sum())
                st.markdown(f"**'{NONE_LABEL}'**: in {len(nn)} attempts, a false dynamic letter would be added **{bad}** times "
                            f"(confidence ≥ {recorder.min_conf:.2f}). Average confidence: {nn['conf'].mean():.2f}")
            if not len(tl) and not len(nn):
                st.caption("No test recordings yet. Pick the true letter in the Dynamic card and record.")
        else:
            st.caption("No recordings yet.")
        if AUTO_LOG.exists():
            st.markdown("---")
            st.markdown("**Auto mode stats** (`results/live_dynamic_auto_log.csv`)")
            ad = pd.read_csv(AUTO_LOG)
            ad["true"] = ad["true"].fillna("")
            ad["pred"] = ad["pred"].fillna("")
            st.markdown(f"Total triggers: **{len(ad)}**; ignored (short/weak movement): "
                        f"**{int(ad['why'].isin(['short', 'weak']).sum())}**")
            used = ad[ad["why"] == "ok"]
            at = used[used["true"].isin(list(DYNAMIC_MAP))]
            if len(at):
                st.markdown(f"Correct on the true letter: **{(at['true'] == at['pred']).sum()}/{len(at)}**; "
                            f"accepted (confidence ≥ {recorder.min_conf:.2f}): **{int((at['conf'] >= recorder.min_conf).sum())}**")
                st.dataframe(pd.crosstab(at["true"], at["pred"]))
            an = used[used["true"] == NONE_LABEL]
            if len(an):
                st.markdown(f"Triggers on '{NONE_LABEL}': **{len(an)}**, of which a false letter was added "
                            f"(confidence ≥ {recorder.min_conf:.2f}): **{int((an['conf'] >= recorder.min_conf).sum())}**")

    # ---- Word card
    with container(border=True):
        st.markdown("<div class='ctitle'>Word</div>", unsafe_allow_html=True)
        word_box = st.empty()
        b1, b2, b3, b4 = st.columns(4)
        b1.button("␣ Space", on_click=state.add_space, use_container_width=True)
        b2.button("⌫ Delete", on_click=state.backspace, use_container_width=True)
        b3.button("🗑 Clear", on_click=state.clear, use_container_width=True)
        speak_clicked = b4.button("🔊 Speak", use_container_width=True)
        if speak_clicked:
            text = "".join(urdu(n) for n in state.snapshot()["letters"]).strip()
            if text:
                try:
                    st.audio(speak(text), format="audio/mp3")
                except Exception as e:                      # if there is no internet
                    st.warning(f"Could not make the audio (internet?): {e}")

    # ---- Live recognition card
    with container(border=True):
        st.markdown("<div class='ctitle'>Live recognition</div>", unsafe_allow_html=True)
        live_box = st.empty()

    # ---- Dynamic letter card
    with container(border=True):
        st.markdown("<div style='display:flex;align-items:center;margin-bottom:6px'><div class='ctitle' style='margin:0'>"
                    "Dynamic letter</div><span class='dglyphs'>ج &nbsp; آ &nbsp; ڑ &nbsp; ہ</span></div>",
                    unsafe_allow_html=True)
        d1, d2 = st.columns([2, 3])
        if recorder.practice_label:                        # Practice mode: the letter is already chosen in the sidebar
            d1.markdown(f"<div class='ctitle' style='margin-top:10px'>Practising "
                        f"{urdu(recorder.practice_label)} {recorder.practice_label}</div>", unsafe_allow_html=True)
            recorder.auto_label = recorder.practice_label
        else:
            _sel = d1.selectbox("True letter (for testing)", TEST_OPTIONS, key="dyn_true", label_visibility="collapsed",
                         help=f"Pick the true letter for testing. '{NONE_LABEL}' = deliberately NO dynamic letter "
                              f"(only move your hand or change the static sign).")
            recorder.auto_label = None if _sel == TEST_OPTIONS[0] else _sel
        d2.button("🎬 Record (3-2-1)", on_click=start_dynamic, type="primary", use_container_width=True)
        dyn_box = st.empty()


def live_html(s):
    busy = recorder.blocks_static(time.time())          # a tiny tremor in Auto mode is not "recording"
    fps = state.fps
    chips = chip("Hand found", GREEN) if s["has_hand"] else chip("No hand", GRAY)
    chips += chip("Dynamic recording", RED) if busy else chip("Static mode", BLUE)
    chips += chip(f"{fps:.0f} FPS", RED if 0 < fps < 10 else GRAY, dot=False)

    label, conf = s["top_label"], s["top_conf"]
    if busy or not label:
        color, glyph = GRAY, "—"
        name = "Recording…" if busy else "Show your hand"
        sub = ""
    elif "?" in label and label.replace(" ?", "") in GROUPS:
        members = GROUPS[label.replace(" ?", "")]
        color, glyph = ORANGE, " ".join(urdu(m) for m in members)
        name, sub = " / ".join(members) + " ?", "Model cannot tell them apart"
    else:
        color, glyph = (GREEN if conf >= state.min_conf else ORANGE), urdu(label)
        name, sub = label, f"confidence {conf * 100:.0f}%"

    rows = ""
    for n, p in ([] if busy else s["top3"]):
        rows += (f"<div class='brow' style='--c:{color if n == label else BLUE}'><span class='bl'>{urdu(n)} {n}</span>"
                 f"<div class='btrack'><div class='bfill' style='width:{p * 100:.0f}%'></div></div>"
                 f"<span class='bp'>{p * 100:.0f}%</span></div>")
    hold = 0.0 if busy else s["hold"]
    note = "Keep holding… letter will be added" if 0 < hold < 1 else ("✓ Added, change the sign" if hold >= 1 else "")
    pend = ""
    if s["pending"]:
        n1, n2 = pick_names(s["pending"])
        pend = (f"<div class='pend'>⚠️ {urdu(n1)} {n1} or {urdu(n2)} {n2}? "
                f"Press a button above the camera, or key <b>1</b> / <b>2</b> on the keyboard.</div>")
    return (f"<div>{chips}</div>"
            f"<div class='live' style='--c:{color}'><div class='glyph'>{glyph}</div>"
            f"<div class='lmeta'><div class='lname'>{name}</div><div class='lconf'>{sub}</div></div>"
            f"<div class='bars'>{rows}</div></div>"
            f"<div class='hold' style='--c:{GREEN}'><div class='hfill' style='width:{hold * 100:.0f}%'></div></div>"
            f"<div class='hnote'>{note}</div>{pend}")


def coach_html(cs):
    """Live recognition card in practice mode: photo of the target, match with the model, finger colours, tips."""
    t = cs["target"]
    fps = state.fps
    chips = chip("Hand found", GREEN) if cs["has_hand"] else chip("No hand", GRAY)
    chips += chip("Practice mode", ORANGE) + chip(f"{fps:.0f} FPS", RED if 0 < fps < 10 else GRAY, dot=False)
    if not cs["has_hand"]:
        color, status = GRAY, "Show your hand and copy the photo"
    elif cs["accepted"]:
        color = GREEN
        status = "✓ Done! Press Next letter ▶" if cs["done"] else "✓ Correct! Hold it for a second"
    else:
        color, status = ORANGE, "Not yet"
    match = cs["match"] if cs["has_hand"] else 0.0
    photo = guide_photo_uri(t)
    pic = f"<img class='cphoto' src='{photo}'>" if photo else f"<div class='glyph'>{urdu(t)}</div>"
    fingers = ""
    if cs["has_hand"] and cs["finger_z"]:
        for f, z in enumerate(cs["finger_z"]):
            c = RED if (f == cs["worst"] and not cs["accepted"]) else (ORANGE if z > Z_TIP else GREEN)
            fingers += chip(FINGERS[f].title(), c)
    tips = ""
    if cs["has_hand"] and not cs["accepted"]:
        if cs["tips"]:
            tips = "<ol class='tips'>" + "".join(f"<li>{x}</li>" for x in cs["tips"]) + "</ol>"
        else:
            tips = ("<div class='hnote'>Your fingers look like the photos; check where your palm faces "
                    "and compare with the photo.</div>")
    elif cs["polish"] and cs["tips"]:
        tips = f"<div class='hnote'>Even better (to look more like the photos): {cs['tips'][0].lower()}</div>"
    return (f"<div>{chips}</div>"
            f"<div class='live' style='--c:{color}'>{pic}"
            f"<div class='lmeta'><div class='lname'>{urdu(t)} {t}</div><div class='lconf'>{status}</div></div>"
            f"<div class='bars'><div class='brow' style='--c:{color}'><span class='bl'>match</span>"
            f"<div class='btrack'><div class='bfill' style='width:{match * 100:.0f}%'></div></div>"
            f"<span class='bp'>{match * 100:.0f}%</span></div>"
            f"<div class='fchips'>{fingers}</div></div></div>"
            f"<div class='hold' style='--c:{GREEN}'><div class='hfill' style='width:{cs['good_frac'] * 100:.0f}%'></div></div>"
            f"{tips}")


def dyn_practice_html():
    """Live recognition card while practising a moving letter: target, how to do it, last attempt, score."""
    t = recorder.practice_label
    fps = state.fps
    now = time.time()
    phase, _ = recorder.status(now)
    chips = chip("Practice mode", ORANGE) + chip("Moving letter", BLUE)
    chips += chip("Auto mode on" if recorder.auto else "Press Record", GRAY, dot=False)
    chips += chip(f"{fps:.0f} FPS", RED if 0 < fps < 10 else GRAY, dot=False)
    r = recorder.result
    fresh = r is not None and recorder.result_age(now) <= RESULT_SHOW_SECONDS
    text, kind = dynamic_verdict(r, t, recorder.min_conf) if fresh else (None, "info")
    color = {"good": GREEN, "warn": ORANGE, "bad": RED, "info": BLUE}[kind]
    if phase in ("countdown", "recording"):
        status, color = "Recording... do the gesture now" if phase == "recording" else "Get ready: 3-2-1", RED
    elif phase == "auto" and recorder.blocks_static(now):
        status, color = "Recording (Auto mode)...", RED
    elif text:
        status = text
    else:
        status = ("Do the gesture: move your hand fast, then stop" if recorder.auto
                  else "Press 🎬 Record, then do the gesture after 3-2-1")
    gif = dyn_gif_uri(t)
    pic = (f"<img class='cphoto' src='{gif}' style='width:150px;object-fit:contain'>" if gif
           else f"<div class='glyph'>{urdu(t)}</div>")
    score = (f"{recorder.practice_ok} of {recorder.practice_tries} attempts correct"
             if recorder.practice_tries else "No attempts yet")
    return (f"<div>{chips}</div>"
            f"<div class='live' style='--c:{color}'>{pic}"
            f"<div class='lmeta'><div class='lname'>{urdu(t)} {t}</div><div class='lconf'>{status}</div>"
            f"<div class='hnote'>{score}</div></div></div>")


def dyn_html():
    r = recorder.result
    if r is not None and recorder.result_age(time.time()) > RESULT_SHOW_SECONDS:
        r = None                                                # old result: show the hint again (history is in Stats)
    if r is None:
        hint = ("Auto mode on: do a dynamic gesture (move your hand fast, then stop)." if recorder.auto
                else "Press 🎬, then do the gesture after 3-2-1." if recorder.practice_label
                else "Pick the true letter, press 🎬, then do the gesture after 3-2-1.")
        return f"<div class='dres'><small>{hint}</small></div>"
    if r.get("ignored") in ("short", "weak"):
        what = "movement too short" if r["ignored"] == "short" else "movement too weak"
        return f"<div class='dres'>{chip('Ignored', GRAY)}<small>{what}, do the gesture again</small></div>"
    if not r["ok"]:
        return (f"<div class='dres'>{chip('Few hand frames', ORANGE)}"
                f"<small>{r['n_hand']}/{r['n_frames']} frames, try again</small></div>")
    ok = r["conf"] >= recorder.min_conf
    verdict = ""
    if r["true"] in DYNAMIC_MAP:
        verdict = chip("correct", GREEN) if r["pred"] == r["true"] else chip(f"wrong (true: {r['true']})", RED)
    elif r["true"] == NONE_LABEL:
        verdict = chip("false letter would be added", ORANGE) if ok else chip("OK, not added", GREEN)
    second = f" · second: {r['top2'][1][0]} {r['top2'][1][1] * 100:.0f}%" if len(r["top2"]) > 1 else ""
    return (f"<div class='dres'><span class='dl'>{urdu(r['pred'])}</span><b>{r['pred']} {r['conf'] * 100:.0f}%</b>"
            f"{'' if ok else chip('low confidence, not added', ORANGE, dot=False)}{verdict}"
            f"<small>{r['n_frames']} frames, hand in {r['n_hand']} of them{second}</small></div>")


def render():
    s = state.snapshot()
    word = "".join(urdu(n) for n in s["letters"])
    word_box.markdown(f"<div class='word'>{word if word else '<span class=ph>letters will appear here…</span>'}</div>",
                      unsafe_allow_html=True)
    cs = coach.snapshot()
    live_box.markdown(coach_html(cs) if cs["target"] else dyn_practice_html() if recorder.practice_label
                      else live_html(s), unsafe_allow_html=True)
    dyn_box.markdown(dyn_html(), unsafe_allow_html=True)
    if (s["pending"] is not None) != (pending_at_start is not None):
        st.rerun()                                              # rebuild the page so the buttons appear / disappear


# keyboard: pressing 1 / 2 clicks the button above the camera
components.html("""
<script>
const doc = window.parent.document;
if (window.parent.__pslKeys) doc.removeEventListener('keydown', window.parent.__pslKeys);
window.parent.__pslKeys = function (e) {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  if (e.key !== '1' && e.key !== '2') return;
  const b = doc.querySelector('.st-key-pick' + e.key + ' button');
  if (b && b.offsetWidth > 0) { b.click(); e.preventDefault(); }   // only a visible button
};
doc.addEventListener('keydown', window.parent.__pslKeys);
</script>
""", height=0)

render()
while ctx.state.playing:
    render()
    time.sleep(0.25)