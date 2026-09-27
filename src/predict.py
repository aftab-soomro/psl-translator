"""
predict.py

Poora pipeline ek function me: image -> landmarks -> normalize -> prediction.
Streamlit app sirf isi predict() function ko call karegi.
"""

from pathlib import Path
import joblib
import numpy as np

from src.landmarks import extract_landmarks
from src.features import normalize_landmarks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_class_names = None


def _load_model():
    """Model aur class names sirf ek dafa load hote hain (pehli call par)."""
    global _model, _class_names
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_model.joblib")
        _class_names = joblib.load(MODELS_DIR / "class_names.joblib")
    return _model, _class_names


def predict(image_path):
    """
    Ek image se harf predict karta hai.

    Args:
        image_path: image file ka path (str ya Path)

    Returns:
        dict with:
            "letter": predicted harf ka naam (str), ya None agar haath na mile
            "confidence": us harf ki probability (0-1), ya None
            "probabilities": saare 36 harfon ki probabilities ka dict, ya None
            "error": agar kuch ghalat hua to wajah (str), warna None
    """
    model, class_names = _load_model()

    raw_landmarks = extract_landmarks(image_path)
    if raw_landmarks is None:
        return {
            "letter": None,
            "confidence": None,
            "probabilities": None,
            "error": "Haath nahi mila. Saaf roshni me, ek haath camera ke saamne rakhein.",
        }

    normalized = normalize_landmarks(raw_landmarks)
    features = normalized.reshape(1, -1)  # model ek "batch" expect karta hai, isliye 2D shape

    predicted_letter = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    prob_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
    confidence = prob_dict[predicted_letter]

    return {
        "letter": predicted_letter,
        "confidence": confidence,
        "probabilities": prob_dict,
        "error": None,
    }