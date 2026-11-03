"""
live.py

Live webcam ke liye:
  - LiveHandTracker : MediaPipe HandLandmarker VIDEO mode mein (frame se frame tracking)
  - LivePredictor   : 63 landmarks -> normalize -> model -> top-3 harf
  - WordState       : 1 second ek hi harf dikhane par harf khud jud jata hai (thread-safe)
  - draw_hand / draw_measurements : 21 points, lines aur faasle frame par draw karna
  - make_callback   : webrtc ke liye frame callback
"""

import threading
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.predict import MODEL_FILE          # model badalna ho to sirf wahan badlo

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIPS = [4, 8, 12, 16, 20]


# --------------------------------------------------------------------------- tracker
class LiveHandTracker:
    """Ek hi HandLandmarker jo live frames par chalta hai."""

    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0
        self._lock = threading.Lock()

    def detect(self, bgr):
        """BGR frame -> 63 raw numbers [x0,y0,z0, ...] (extract_landmarks jaisa format), ya None."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # timestamp hamesha barhta rahe
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        return [v for p in result.hand_landmarks[0] for v in (p.x, p.y, p.z)]


# --------------------------------------------------------------------------- prediction
def normalize63(lm):
    """Raw 63 numbers -> (1, 63) normalized: wrist origin par, wrist->point 9 distance se divide."""
    pts = np.asarray(lm, dtype=float).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-9:
        scale = 1.0
    return (pts / scale).reshape(1, 63)


class LivePredictor:
    """Wahi model jo app mein deploy hai (src/predict.py ka MODEL_FILE)."""

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / MODEL_FILE)
        self.names = joblib.load(MODELS_DIR / "class_names.joblib")

    def _name(self, c):
        if isinstance(c, (int, np.integer)):
            return str(self.names[int(c)])
        return str(c)

    def top3(self, lm):
        """63 raw numbers -> [(harf_ka_naam, probability), ...] 3 tak."""
        proba = self.model.predict_proba(normalize63(lm))[0]
        idx = np.argsort(proba)[::-1][:3]
        return [(self._name(self.model.classes_[i]), float(proba[i])) for i in idx]


# --------------------------------------------------------------------------- word state
class WordState:
    """
    Callback thread likhta hai, Streamlit page parhta hai, isliye har cheez lock ke andar.

    Qaida: agar top harf min_conf se upar ho aur hold_seconds tak wahi rahe -> harf lafz mein jud jata hai.
    Wahi harf dobara tab judega jab (a) haath hata kar wapas lao, ya (b) koi aur harf pehle jur jaye.
    """

    MISS_RESET = 8      # itne frames lagataar haath na mile to "haath hata diya" maana jata hai
    SWITCH_AFTER = 3    # itne frames lagataar dusra harf aaye tabhi candidate badalta hai (flicker se bachao)

    def __init__(self):
        self.lock = threading.Lock()
        self.letters = []           # class names, jaise "Alif"; space ke liye " "
        self.top3 = []
        self.has_hand = False
        self.fps = 0.0
        self.hold = 0.0             # 0..1
        self.locked = None          # aakhri jura hua harf
        self._cand = None
        self._cand_start = 0.0
        self._flick = 0
        self._miss = 0
        # settings (page se badalti hain)
        self.hold_seconds = 1.0
        self.min_conf = 0.60
        self.show_points = True
        self.show_dist = True

    def _reset_candidate(self):
        self._cand = None
        self._flick = 0
        self.hold = 0.0

    def update(self, top3, now):
        """top3 = [(name, prob), ...] ya None (haath nahi mila)."""
        with self.lock:
            if top3 is None:
                self.has_hand = False
                self.top3 = []
                self._miss += 1
                if self._miss >= self.MISS_RESET:
                    self._reset_candidate()
                    self.locked = None
                return

            self._miss = 0
            self.has_hand = True
            self.top3 = top3
            name, conf = top3[0]
            cur = name if conf >= self.min_conf else None

            if cur is None:                                  # yaqeen kam
                self._flick += 1
                if self._flick >= self.SWITCH_AFTER:
                    self._reset_candidate()
                return

            if cur == self._cand:
                self._flick = 0
            elif self._cand is None:
                self._cand, self._cand_start, self._flick = cur, now, 0
            else:
                self._flick += 1
                if self._flick >= self.SWITCH_AFTER:
                    self._cand, self._cand_start, self._flick = cur, now, 0

            if self._cand is None:
                return
            self.hold = min((now - self._cand_start) / self.hold_seconds, 1.0)
            if self.hold >= 1.0 and self._cand != self.locked:
                self.letters.append(self._cand)
                self.locked = self._cand

    # buttons
    def add_space(self):
        with self.lock:
            if self.letters and self.letters[-1] != " ":
                self.letters.append(" ")

    def backspace(self):
        with self.lock:
            if self.letters:
                self.letters.pop()
            self.locked = None

    def clear(self):
        with self.lock:
            self.letters.clear()
            self.locked = None

    def snapshot(self):
        with self.lock:
            return {
                "letters": list(self.letters),
                "top3": list(self.top3),
                "has_hand": self.has_hand,
                "hold": self.hold,
                "locked": self.locked,
            }


# --------------------------------------------------------------------------- drawing
def draw_hand(bgr, landmarks):
    """Frame par 21 points aur unki lines draw karta hai (fingertips bare aur laal)."""
    h, w = bgr.shape[:2]
    pts = [(int(landmarks[3 * i] * w), int(landmarks[3 * i + 1] * h)) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(bgr, pts[a], pts[b], (255, 255, 255), 2)
    for i, p in enumerate(pts):
        if i in TIPS:
            cv2.circle(bgr, p, 7, (0, 0, 255), -1)
        else:
            cv2.circle(bgr, p, 4, (0, 200, 255), -1)
    return bgr


def _label(bgr, text, p, color):
    cv2.putText(bgr, text, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(bgr, text, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_measurements(bgr, landmarks):
    """
    Faasle (palm length = wrist->point 9 ki unit mein):
      peeli lines  = wrist se har fingertip tak (5)
      neeli lines  = paas paas ki fingertips ke darmiyan (4)
    """
    h, w = bgr.shape[:2]
    pts = np.array([(landmarks[3 * i] * w, landmarks[3 * i + 1] * h) for i in range(21)])
    palm = np.linalg.norm(pts[9] - pts[0])
    if palm < 1:
        return bgr

    def seg(a, b, color):
        pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
        cv2.line(bgr, pa, pb, color, 1)
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        _label(bgr, f"{np.linalg.norm(pts[a] - pts[b]) / palm:.1f}", mid, color)

    for t in TIPS:
        seg(0, t, (0, 255, 255))            # peela (BGR)
    for a, b in zip(TIPS[:-1], TIPS[1:]):
        seg(a, b, (255, 200, 0))            # neela
    return bgr


# --------------------------------------------------------------------------- webrtc callback
def make_callback(state, tracker, predictor):
    import av

    fps = {"t": time.time()}

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        lm = tracker.detect(img)                       # asli (bina ulte) frame par, jaise training mein
        top3 = predictor.top3(lm) if lm is not None else None
        state.update(top3, now)

        img = cv2.flip(img, 1)                         # sirf dikhane ke liye shishe jaisa
        if lm is not None:
            lm_view = [1.0 - v if i % 3 == 0 else v for i, v in enumerate(lm)]   # x ulta
            if state.show_points:
                draw_hand(img, lm_view)
            if state.show_dist:
                draw_measurements(img, lm_view)

        h, w = img.shape[:2]
        if top3:
            name, conf = top3[0]
            ok = conf >= state.min_conf
            cv2.putText(img, f"{name} {conf * 100:.0f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if ok else (0, 165, 255), 3)
        with state.lock:
            hold = state.hold
        cv2.rectangle(img, (0, h - 14), (int(w * hold), h), (0, 255, 0), -1)   # hold meter

        state.fps = 0.9 * state.fps + 0.1 * (1.0 / max(now - fps["t"], 1e-6))
        fps["t"] = now
        cv2.putText(img, f"FPS {state.fps:.0f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback