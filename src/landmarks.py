"""
Hand landmark extraction using MediaPipe.
"""


def extract_landmarks(image):
    """
    Extract 21 hand landmarks (x, y, z) from a single image using MediaPipe Hands.

    Args:
        image: A single image (e.g. as a numpy array in BGR or RGB format,
            such as one read via OpenCV or captured from a webcam frame)
            containing a hand to detect.

    Returns:
        A flat list/array of 63 float values (21 landmarks x 3 coordinates
        [x, y, z]) if exactly one hand is detected in the image, or None if
        no hand is detected.
    """
    raise NotImplementedError
