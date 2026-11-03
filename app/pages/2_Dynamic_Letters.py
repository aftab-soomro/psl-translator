import sys
import tempfile
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.video_features import video_features

MODEL_PATH = ROOT / "models" / "dynamic_rf_T8.joblib"
# Urdu harf: FESF PSL dictionary (psl.org.pk) se ek dafa verify karna baaqi hai
DYNAMIC_MAP = {"2-Hay": "ھ", "Alifmad": "آ", "Aray": "ڑ", "Jeem": "ج"}
LOW_CONFIDENCE = 0.50

st.set_page_config(page_title="Dynamic Letters", page_icon="🎥")
st.title("Dynamic Letters (video)")
st.caption("4 harf jinme haath hilta hai: 2-Hay, Alifmad, Aray, Jeem. "
           "2-3 second ki video upload karo, ek haath, saaf roshni, sada background.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


video = st.file_uploader("Video chuno", type=["mp4", "mov", "avi", "m4v"])

if video is not None:
    data = video.getvalue()
    st.video(data)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ("clip" + Path(video.name).suffix)
        path.write_bytes(data)
        with st.spinner("Video ke frames se haath ke points nikal rahe hain..."):
            feats = video_features(path)

    if feats is None:
        st.warning("Video mein haath nahi mila. Saaf roshni mein, haath poora frame mein rakh kar dobara record karein.")
    else:
        model = load_model()
        probs = model.predict_proba(feats)[0]
        order = probs.argsort()[::-1]
        classes = list(model.classes_)
        top, conf = classes[order[0]], float(probs[order[0]])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div style='font-size:96px; text-align:center;'>{DYNAMIC_MAP.get(top, top)}</div>"
                f"<div style='text-align:center;'>{top}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.metric("Confidence", f"{conf:.1%}")
            st.progress(conf)

        if conf < LOW_CONFIDENCE:
            st.warning("Model ko pakka yaqeen nahi. Dobara video banayein.")

        st.subheader("Saare 4 harf")
        for i in order:
            st.write(f"{DYNAMIC_MAP.get(classes[i], classes[i])}  {classes[i]}")
            st.progress(float(probs[i]))