"""
predict.py

The whole pipeline in one function: image -> landmarks -> normalize -> prediction.
The Streamlit app only calls this predict() function.
"""

from pathlib import Path
import joblib
import numpy as np

from src.landmarks import extract_landmarks
from src.features import normalize_landmarks

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = "best_model_v3_eng.joblib"   # v2: "best_model_v2_mirror.joblib", core: "best_model_core_v1.joblib"

_model = None
_class_names = None


def _load_model():
    """The model and class names are loaded only once (on the first call)."""
    global _model, _class_names
    if _model is None:
        _model = joblib.load(MODELS_DIR / MODEL_FILE)
        _class_names = joblib.load(MODELS_DIR / "class_names.joblib")
    return _model, _class_names


def predict(image_path):
    """
    Predicts a letter from one image.

    Args:
        image_path: path of the image file (str or Path)

    Returns:
        dict with:
            "letter": name of the predicted letter (str), or None if no hand is found
            "confidence": probability of that letter (0-1), or None
            "probabilities": dict of the probabilities of all 36 letters, or None
            "landmarks": list of the 63 raw landmark numbers, or None
            "error": the reason (str) if something went wrong, otherwise None
    """
    model, class_names = _load_model()

    raw_landmarks = extract_landmarks(image_path)
    if raw_landmarks is None:
        return {
            "letter": None,
            "confidence": None,
            "probabilities": None,
            "landmarks": None,
            "error": "No hand found. Hold one hand in front of the camera in good light.",
        }

    normalized = normalize_landmarks(raw_landmarks)
    features = normalized.reshape(1, -1)  # the model expects a "batch", so a 2D shape

    predicted_letter = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    prob_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
    confidence = prob_dict[predicted_letter]

    return {
        "letter": predicted_letter,
        "confidence": confidence,
        "probabilities": prob_dict,
        "landmarks": raw_landmarks,
        "error": None,
    }