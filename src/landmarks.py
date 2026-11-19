"""
landmarks.py

Extracts MediaPipe hand landmarks from one image: 21 points x 3 (x, y, z) = 63 numbers.
"""

from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# path of models/hand_landmarker.task inside the project root, worked out automatically
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

_detector = None


def _get_detector():
    """The detector is created only once (on the first call) and then reused."""
    global _detector
    if _detector is None:
        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        _detector = mp_vision.HandLandmarker.create_from_options(options)
    return _detector


def extract_landmarks(image_path):
    """
    Extracts the 21 hand landmarks from one image file.

    Args:
        image_path: path of the image file (str or Path)

    Returns:
        list of 63 numbers [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        if a hand is found, otherwise None.
    """
    detector = _get_detector()
    mp_image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]  # num_hands=1, so only one hand is returned
    features = []
    for point in hand:
        features.extend([point.x, point.y, point.z])

    return features