"""
Feature engineering for hand landmarks.
"""


def normalize_landmarks(landmarks):
    """
    Normalize raw hand landmarks into a scale- and translation-invariant
    feature vector suitable for input to a classical ML model.

    Args:
        landmarks: A flat list/array of 63 float values (21 landmarks x 3
            coordinates [x, y, z]) as returned by
            src.landmarks.extract_landmarks.

    Returns:
        A flat list/array of normalized feature values (e.g. translated
        relative to a reference landmark and scaled to a consistent range)
        ready to be passed to a trained classifier.
    """
    raise NotImplementedError
