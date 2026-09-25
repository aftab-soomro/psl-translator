"""
End-to-end inference: image in, predicted Urdu alphabet letter out.
"""


def predict(image):
    """
    Run the full inference pipeline on a single image: extract hand
    landmarks, normalize them into features, and classify with the trained
    model to predict the corresponding Urdu alphabet letter.

    Args:
        image: A single image (e.g. as a numpy array in BGR or RGB format,
            such as one read via OpenCV or captured from a webcam frame)
            containing a hand to classify.

    Returns:
        A dict with the following keys:
            "letter": str, the predicted Urdu alphabet letter (or None if
                no hand was detected).
            "confidence": float, the model's confidence score for the
                predicted letter (or None if no hand was detected).
            "probabilities": dict mapping every class label (str) to its
                predicted probability (float), for all classes the model
                was trained on (or None if no hand was detected).
    """
    raise NotImplementedError
