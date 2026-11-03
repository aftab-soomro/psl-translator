"""
video_features.py

Video -> 4 dynamic harf (2-Hay, Alifmad, Aray, Jeem) ke liye features.
Notebook 15 mein jo kiya tha, bilkul wahi yahan functions ki shakal mein hai,
taake Streamlit app bhi use kar sake.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.landmarks import extract_landmarks
from src.features import normalize_landmarks

T_STEPS = 8          # video ko kitne time steps mein baantna hai (CV se chuna)


def video_landmarks(video_path, step=2, width=640):
    """Video ke har step-wein frame ke raw landmarks (n, 63). Haath na mile to us frame ke 63 NaN."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "f.jpg"
        cap = cv2.VideoCapture(str(video_path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (width, int(h * width / w)))
                cv2.imwrite(str(frame_path), frame)
                lm = extract_landmarks(frame_path)
                rows.append(list(lm) if lm is not None else [np.nan] * 63)
            i += 1
        cap.release()
    return np.array(rows, dtype=np.float64).reshape(-1, 63)


def video_sequence(arr):
    """(n, 63) raw landmarks -> (n, 65): 63 normalized shape + 2 kalai ki harkat.
    None agar haath kisi frame mein nahi mila."""
    if len(arr) < 2:
        return None
    arr = pd.DataFrame(arr).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        return None
    P = arr.reshape(len(arr), 21, 3)
    size = np.linalg.norm(P[:, 9] - P[:, 0], axis=1).mean() + 1e-6
    wrist = (P[:, 0, :2] - P[0, 0, :2]) / size       # pehle frame se kalai ki harkat
    shape = np.vstack([normalize_landmarks(r) for r in arr])
    return np.hstack([shape, wrist])


def resample(seq, T=T_STEPS):
    """Kisi bhi lambai ki sequence ko T barabar time steps mein badalta hai."""
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, T)
    return np.column_stack([np.interp(t_new, t_old, seq[:, j]) for j in range(seq.shape[1])])


def video_features(video_path, T=T_STEPS):
    """Video file -> (1, T*65) feature, ya None agar haath na mile."""
    seq = video_sequence(video_landmarks(video_path))
    if seq is None:
        return None
    return resample(seq, T).ravel().reshape(1, -1)