"""
features.py

Raw landmarks (kalai ki position aur haath ke size par depend karte hain) ko
normalize karta hai, taake camera se doori aur image me position ka asar khatam ho jaye.
"""

import numpy as np


def normalize_landmarks(landmarks):
    """
    63 raw numbers [x0,y0,z0, x1,y1,z1, ...] ko normalize karta hai:
    - kalai (point 0) ko (0,0,0) bana deta hai
    - middle finger ke base (point 9) tak ki doori se sabko divide karta hai

    Args:
        landmarks: 63 numbers ki list (extract_landmarks() ka output)

    Returns:
        63 normalized numbers ki numpy array
    """
    points = np.array(landmarks).reshape(21, 3)  # 21 rows, har row me [x, y, z]

    wrist = points[0].copy()
    points = points - wrist  # ab sab points kalai ke "muqable" hain

    # haath ka size: kalai se middle finger ke base (point 9) tak ki doori
    hand_size = np.linalg.norm(points[9])
    if hand_size < 1e-6:  # bohat chhota number, divide by zero se bachne ke liye
        hand_size = 1e-6

    points = points / hand_size

    return points.flatten()  # wapas 63 numbers ki seedhi list