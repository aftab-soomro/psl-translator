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

TIPS = [4, 8, 12, 16, 20]                         # fingertip points
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]   # har ungli ke points


def add_engineered_features(X):
    """63 normalized landmarks (N, 63) -> (N, 93): fasle aur angles jod kar."""
    P = X.reshape(len(X), 21, 3)
    feats = []
    for t in TIPS:                                 # 5: tip se kalai ka fasla
        feats.append(np.linalg.norm(P[:, t], axis=1))
    for i in range(5):                             # 10: tip se tip ka fasla
        for j in range(i + 1, 5):
            feats.append(np.linalg.norm(P[:, TIPS[i]] - P[:, TIPS[j]], axis=1))
    for chain in CHAINS:                           # 15: joints ke angles
        for k in range(1, 4):
            a = P[:, chain[k - 1]] - P[:, chain[k]]
            b = P[:, chain[k + 1]] - P[:, chain[k]]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            feats.append(np.arccos(np.clip(cos, -1, 1)))
    return np.hstack([X, np.column_stack(feats)])