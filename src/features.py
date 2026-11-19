"""
features.py

Normalizes raw landmarks (they depend on the wrist position and the hand size)
so that the distance from the camera and the position in the image no longer matter.
"""

import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalizes 63 raw numbers [x0,y0,z0, x1,y1,z1, ...]:
    - moves the wrist (point 0) to (0,0,0)
    - divides everything by the distance to the base of the middle finger (point 9)

    Args:
        landmarks: list of 63 numbers (output of extract_landmarks())

    Returns:
        numpy array of 63 normalized numbers
    """
    points = np.array(landmarks).reshape(21, 3)  # 21 rows, each row is [x, y, z]

    wrist = points[0].copy()
    points = points - wrist  # now all points are relative to the wrist

    # hand size: distance from the wrist to the base of the middle finger (point 9)
    hand_size = np.linalg.norm(points[9])
    if hand_size < 1e-6:  # a very small number, to avoid dividing by zero
        hand_size = 1e-6

    points = points / hand_size

    return points.flatten()  # back to a flat list of 63 numbers

TIPS = [4, 8, 12, 16, 20]                         # fingertip points
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]   # points of each finger


def add_engineered_features(X):
    """63 normalized landmarks (N, 63) -> (N, 93): adding distances and angles."""
    P = X.reshape(len(X), 21, 3)
    feats = []
    for t in TIPS:                                 # 5: distance from tip to wrist
        feats.append(np.linalg.norm(P[:, t], axis=1))
    for i in range(5):                             # 10: distance from tip to tip
        for j in range(i + 1, 5):
            feats.append(np.linalg.norm(P[:, TIPS[i]] - P[:, TIPS[j]], axis=1))
    for chain in CHAINS:                           # 15: joint angles
        for k in range(1, 4):
            a = P[:, chain[k - 1]] - P[:, chain[k]]
            b = P[:, chain[k + 1]] - P[:, chain[k]]
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            feats.append(np.arccos(np.clip(cos, -1, 1)))
    return np.hstack([X, np.column_stack(feats)])