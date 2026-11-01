import streamlit as st
import sys
import tempfile
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import predict

LOW_CONFIDENCE = 0.40

LETTER_MAP = {
    "Alif": "ا", "Bay": "ب", "Pay": "پ", "Tay": "ت", "Taay": "ٹ", "Say": "ث",
    "Chay": "چ", "1-Hay": "ح", "Khay": "خ", "Daal": "د", "Dal": "ڈ", "Zaal": "ذ",
    "Ray": "ر", "Zay": "ز", "Zaey": "ژ", "Seen": "س", "Sheen": "ش", "Suad": "ص",
    "Zuad": "ض", "Tuey": "ط", "Zuey": "ظ", "Ain": "ع", "Ghain": "غ", "Fay": "ف",
    "Kaf": "ق", "Kiaf": "ک", "Gaaf": "گ", "Lam": "ل", "Meem": "م", "Nuun": "ن",
    "Nuungh": "ں", "Wao": "و", "Dochahay": "ھ", "Hamza": "ء", "Cyeh": "ی", "Byeh": "ے",
}

st.set_page_config(page_title="PSL Translator", page_icon="🤟")

st.title("Pakistan Sign Language — Urdu Alphabet Translator")
st.write("Apna haath webcam ke saamne rakhein aur ek Urdu harf ka sign karein.")
snapshot = st.camera_input("Apna haath camera ke saamne rakhein aur photo lein")

if snapshot is not None:
    # Photo ek temporary folder me jati hai, jo khatam hote hi khud delete ho jata hai
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "snapshot.jpg"
        temp_path.write_bytes(snapshot.getvalue())
        result = predict(temp_path)
        img = cv2.imread(str(temp_path))

    if result["error"]:
        st.warning(result["error"])
    else:
        # Asal image par landmarks (laal dots) draw karke dikhana
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        lm = result["landmarks"]
        for i in range(0, len(lm), 3):
            x, y = int(lm[i] * w), int(lm[i + 1] * h)
            cv2.circle(img_rgb, (x, y), 6, (255, 0, 0), -1)
        st.image(img_rgb, caption="Detected hand landmarks", width="stretch")

        # Bara Urdu harf aur confidence
        urdu_letter = LETTER_MAP.get(result["letter"], "?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{urdu_letter}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{result['letter']}</p>", unsafe_allow_html=True)
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.progress(result["confidence"])

        # Kam confidence par warning
        if result["confidence"] < LOW_CONFIDENCE:
            st.warning("Model ko is sign par kam yaqeen hai. Haath thora saaf aur roshni mein rakh kar dobara photo lein.")

        # Top 3 andaze
        st.subheader("Top 3 andaze")
        top3 = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        for name, prob in top3:
            c1, c2, c3 = st.columns([1, 2, 5])
            c1.markdown(f"<h2 style='text-align: center;'>{LETTER_MAP.get(name, '?')}</h2>", unsafe_allow_html=True)
            c2.write(name)
            c3.progress(prob, text=f"{prob:.1%}")