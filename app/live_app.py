"""
live_app.py  -  PSL Live Translator (single page)

Chalane ka tareeqa:  python -m streamlit run app/live_app.py

Layout (ek hi screen, scroll nahi):
  left  : camera ke upar Tuey/Daal ke 1/2 buttons ki khali jagah, phir live camera
  right : header (Settings, Hisaab) / Lafz card / Live pehchaan card / Dynamic harf card
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
from src.letters import LETTER_MAP
from src.live import (AUTO_LOG, DYNAMIC_LOG, DYNAMIC_MAP, DYNAMIC_MODEL_PATH, GROUPS, DynamicRecorder,
                      LiveHandTracker, LivePredictor, WordState, make_callback, pick_names)

AUDIO_DIR = ROOT / "app" / "audio_cache"
GUIDE_DIR = ROOT / "app" / "assets" / "letters"
NONE_LABEL = "kuch nahi"
TEST_OPTIONS = ["(test nahi)"] + list(DYNAMIC_MAP) + [NONE_LABEL]

GREEN, ORANGE, BLUE, RED, GRAY = "#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#94a3b8"

st.set_page_config(page_title="PSL Live Translator", page_icon="🖐️", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ style (dark aur light dono mein theek)
st.markdown("""
<style>
.block-container { padding-top: 3rem; padding-bottom: 0.3rem; max-width: 1560px; }
[data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.15rem 0.2rem; }

/* Harf guide (sidebar) */
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

/* Tuey/Daal ke do buttons, top-left par chota number */
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


def chip(text, color, dot=True):
    return f"<span class='chip' style='--c:{color}'>{'<i></i>' if dot else ''}{text}</span>"


def container(**kw):
    try:
        return st.container(**kw)
    except TypeError:                    # purana Streamlit: key / border ka option nahi
        kw.pop("key", None)
        kw.pop("border", None)
        return st.container()


@st.cache_data(show_spinner=False)
def guide_cards_html():
    """Sidebar ke liye har harf ka chota card: photo + Urdu harf + Roman naam."""
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
            pic = "<div class='gph'>photo nahi</div>"
        cards.append(f"<div class='gcard'>{pic}<div class='gu'>{urdu(name)}</div><div class='gn'>{name}</div></div>")
    for name in DYNAMIC_MAP:
        pic = "<div class='gph'>🎬<br>hilke<br>karo</div>"
        cards.append(f"<div class='gcard dyn'>{pic}<div class='gu'>{DYNAMIC_MAP[name]}</div><div class='gn'>{name}</div></div>")
    return "<div class='ggrid'>" + "".join(cards) + "</div>"


with st.sidebar:
    st.markdown("<p class='gtitle'>📖 Harf guide</p>"
                "<p class='gsub'>Apna harf dhoondo, photo jaisa ishara karo. "
                "Naarangi border wale harf hilke (movement se) bante hain.</p>",
                unsafe_allow_html=True)
    st.markdown(guide_cards_html(), unsafe_allow_html=True)


pending_at_start = state.snapshot()["pending"]

left, right = st.columns([6, 7], gap="large")

# ------------------------------------------------------------------ left: choice bar + camera
with left:
    with container(key="leftcol"):
        with container(key="choicebar"):                        # camera ke upar khali jagah
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

# ------------------------------------------------------------------ right
with right:
    top_brand, top_set, top_stats = st.columns([5, 2, 2])
    top_brand.markdown("<div class='brand'>🖐️ PSL Live Translator<small>Pakistan Sign Language → Urdu</small></div>",
                       unsafe_allow_html=True)

    with top_set.popover("⚙️ Settings", use_container_width=True):
        st.markdown("**Static harf**")
        state.hold_seconds = st.slider("Harf jodne ke liye kitni der rukna hai (second)", 0.5, 2.5, 1.0, 0.1)
        state.min_conf = st.slider("Static: kam az kam yaqeen", 0.30, 0.95, 0.60, 0.05)
        state.smooth_seconds = st.slider("Smoothing (second)", 0.1, 1.0, 0.4, 0.1,
                                         help="Pichle itne waqt ki probabilities ka average")
        state.show_points = st.checkbox("21 points aur lines dikhao", value=True)
        state.show_dist = st.checkbox("Faasle dikhao (peela: wrist→tip, neela: tip→tip)", value=True)
        st.markdown("**Dynamic harf**")
        recorder.record_seconds = st.slider("Recording kitni lambi (second)", 2.0, 4.0, 3.0, 0.5)
        recorder.min_conf = st.slider("Dynamic: kam az kam yaqeen", 0.30, 0.95, 0.60, 0.05)
        recorder.aspect_fix = st.checkbox("16:9 fix (training videos jaisa)", value=True)
        st.caption("Dynamic ke liye right haath use karo; gesture poore 3 second mein phaila do.")
        st.markdown("**Auto mode** (button ke baghair)")
        recorder.auto = st.checkbox("Auto mode on", value=False,
                                    help="Haath tez hilao to recording khud shuru, ruko to khatam. "
                                         "Video par 'AUTO v ..' haath ki raftaar dikhata hai.")
        recorder.v_hi = st.slider("Shuru: raftaar is se oopar", 0.2, 2.0, 0.6, 0.05)
        recorder.v_lo = st.slider("Khatam: raftaar is se neeche", 0.1, 1.0, 0.3, 0.05)
        recorder.min_dur = st.slider("Harkat kam az kam (second)", 0.3, 1.5, 0.7, 0.1)
        recorder.min_energy = st.slider("Harkat ki kam az kam miqdar", 0.3, 3.0, 1.0, 0.1)

    with top_stats.popover("📊 Hisaab", use_container_width=True):
        st.markdown("**Dynamic test ka hisaab** (`results/live_dynamic_log.csv`)")
        st.button("🔄 Taaza karo", key="refresh_stats")
        if DYNAMIC_LOG.exists():
            df = pd.read_csv(DYNAMIC_LOG)
            df["true"] = df["true"].fillna("")
            df["pred"] = df["pred"].fillna("(haath kam)")
            tl = df[df["true"].isin(list(DYNAMIC_MAP))].copy()
            if len(tl):
                tl["sahi"] = tl["true"] == tl["pred"]
                tl["qubool"] = tl["conf"] >= recorder.min_conf
                st.markdown(f"**Kul accuracy: {tl['sahi'].mean() * 100:.0f}%** "
                            f"({tl['sahi'].sum()}/{len(tl)} koshishen)")
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
                st.caption("Abhi test wali koi recording nahi. Dynamic card mein asal harf chun kar record karo.")
        else:
            st.caption("Abhi koi recording nahi.")
        if AUTO_LOG.exists():
            st.markdown("---")
            st.markdown("**Auto mode ka hisaab** (`results/live_dynamic_auto_log.csv`)")
            ad = pd.read_csv(AUTO_LOG)
            ad["true"] = ad["true"].fillna("")
            ad["pred"] = ad["pred"].fillna("")
            st.markdown(f"Kul trigger: **{len(ad)}**; ignore (chhoti/kamzor harkat): "
                        f"**{int(ad['why'].isin(['short', 'weak']).sum())}**")
            used = ad[ad["why"] == "ok"]
            at = used[used["true"].isin(list(DYNAMIC_MAP))]
            if len(at):
                st.markdown(f"Asli harf par sahi: **{(at['true'] == at['pred']).sum()}/{len(at)}**; "
                            f"qubool (yaqeen ≥ {recorder.min_conf:.2f}): **{int((at['conf'] >= recorder.min_conf).sum())}**")
                st.dataframe(pd.crosstab(at["true"], at["pred"]))
            an = used[used["true"] == NONE_LABEL]
            if len(an):
                st.markdown(f"'{NONE_LABEL}' par trigger: **{len(an)}**, jin me fazool harf jura "
                            f"(yaqeen ≥ {recorder.min_conf:.2f}): **{int((an['conf'] >= recorder.min_conf).sum())}**")

    # ---- Lafz card
    with container(border=True):
        st.markdown("<div class='ctitle'>Lafz</div>", unsafe_allow_html=True)
        word_box = st.empty()
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

    # ---- Live pehchaan card
    with container(border=True):
        st.markdown("<div class='ctitle'>Live pehchaan</div>", unsafe_allow_html=True)
        live_box = st.empty()

    # ---- Dynamic harf card
    with container(border=True):
        st.markdown("<div class='ctitle'>Dynamic harf &nbsp; ج آ ڑ ھ</div>", unsafe_allow_html=True)
        d1, d2 = st.columns([2, 3])
        _sel = d1.selectbox("Asal harf (test ke liye)", TEST_OPTIONS, key="dyn_true", label_visibility="collapsed",
                     help=f"Test ke liye asal harf chuno. '{NONE_LABEL}' = jaan boojh kar koi dynamic harf NAHI "
                          f"(sirf haath hilao ya static badlo).")
        recorder.auto_label = None if _sel == TEST_OPTIONS[0] else _sel
        d2.button("🎬 Record karo (3-2-1)", on_click=start_dynamic, type="primary", use_container_width=True)
        dyn_box = st.empty()


def live_html(s):
    busy = recorder.busy()
    fps = state.fps
    chips = chip("Haath mila", GREEN) if s["has_hand"] else chip("Haath nahi", GRAY)
    chips += chip("Dynamic recording", RED) if busy else chip("Static mode", BLUE)
    chips += chip(f"{fps:.0f} FPS", RED if 0 < fps < 10 else GRAY, dot=False)

    label, conf = s["top_label"], s["top_conf"]
    if busy or not label:
        color, glyph = GRAY, "—"
        name = "Recording…" if busy else "Haath dikhao"
        sub = ""
    elif "?" in label and label.replace(" ?", "") in GROUPS:
        members = GROUPS[label.replace(" ?", "")]
        color, glyph = ORANGE, " ".join(urdu(m) for m in members)
        name, sub = " / ".join(members) + " ?", "Model farq nahi kar sakta"
    else:
        color, glyph = (GREEN if conf >= state.min_conf else ORANGE), urdu(label)
        name, sub = label, f"yaqeen {conf * 100:.0f}%"

    rows = ""
    for n, p in ([] if busy else s["top3"]):
        rows += (f"<div class='brow' style='--c:{color if n == label else BLUE}'><span class='bl'>{urdu(n)} {n}</span>"
                 f"<div class='btrack'><div class='bfill' style='width:{p * 100:.0f}%'></div></div>"
                 f"<span class='bp'>{p * 100:.0f}%</span></div>")
    hold = 0.0 if busy else s["hold"]
    note = "Rukte raho… harf jurne wala hai" if 0 < hold < 1 else ("✓ Jur gaya, harf badlo" if hold >= 1 else "")
    pend = ""
    if s["pending"]:
        n1, n2 = pick_names(s["pending"])
        pend = (f"<div class='pend'>⚠️ {urdu(n1)} {n1} ya {urdu(n2)} {n2}? "
                f"Camera ke upar button dabao, ya keyboard par <b>1</b> / <b>2</b>.</div>")
    return (f"<div>{chips}</div>"
            f"<div class='live' style='--c:{color}'><div class='glyph'>{glyph}</div>"
            f"<div class='lmeta'><div class='lname'>{name}</div><div class='lconf'>{sub}</div></div>"
            f"<div class='bars'>{rows}</div></div>"
            f"<div class='hold' style='--c:{GREEN}'><div class='hfill' style='width:{hold * 100:.0f}%'></div></div>"
            f"<div class='hnote'>{note}</div>{pend}")


def dyn_html():
    r = recorder.result
    if r is None:
        hint = ("Auto mode on: dynamic gesture karo (haath tez hilao, phir ruko)." if recorder.auto
                else "Asal harf chuno, 🎬 dabao, 3-2-1 ke baad gesture karo.")
        return f"<div class='dres'><small>{hint}</small></div>"
    if r.get("ignored") in ("short", "weak"):
        what = "harkat chhoti thi" if r["ignored"] == "short" else "harkat kamzor thi"
        return f"<div class='dres'>{chip('Ignore', GRAY)}<small>{what}, gesture dobara karo</small></div>"
    if not r["ok"]:
        return (f"<div class='dres'>{chip('Haath kam mila', ORANGE)}"
                f"<small>{r['n_hand']}/{r['n_frames']} frames, dobara koshish karo</small></div>")
    ok = r["conf"] >= recorder.min_conf
    verdict = ""
    if r["true"] in DYNAMIC_MAP:
        verdict = chip("sahi", GREEN) if r["pred"] == r["true"] else chip(f"ghalat (asal: {r['true']})", RED)
    elif r["true"] == NONE_LABEL:
        verdict = chip("fazool harf jur jata", ORANGE) if ok else chip("theek, nahi jura", GREEN)
    second = f" · dusra: {r['top2'][1][0]} {r['top2'][1][1] * 100:.0f}%" if len(r["top2"]) > 1 else ""
    return (f"<div class='dres'><span class='dl'>{urdu(r['pred'])}</span><b>{r['pred']} {r['conf'] * 100:.0f}%</b>"
            f"{'' if ok else chip('kam yaqeen, nahi jura', ORANGE, dot=False)}{verdict}"
            f"<small>{r['n_frames']} frames, haath {r['n_hand']} mein{second}</small></div>")


def render():
    s = state.snapshot()
    word = "".join(urdu(n) for n in s["letters"])
    word_box.markdown(f"<div class='word'>{word if word else '<span class=ph>harf yahan jurte jayenge…</span>'}</div>",
                      unsafe_allow_html=True)
    live_box.markdown(live_html(s), unsafe_allow_html=True)
    dyn_box.markdown(dyn_html(), unsafe_allow_html=True)
    if (s["pending"] is not None) != (pending_at_start is not None):
        st.rerun()                                              # buttons aane / jaane ke liye page dobara banao


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