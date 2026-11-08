"""
live.py

Live webcam ke liye:
  - LiveHandTracker : MediaPipe HandLandmarker VIDEO mode mein (frame se frame tracking)
  - LivePredictor   : 63 landmarks -> normalize -> static model -> top-3 harf
  - WordState       : 1 second ek hi harf dikhane par harf khud jud jata hai (thread-safe)
  - DynamicRecorder : Tareeqa B - button, 3-2-1, phir 3 second recording -> dynamic model
  - draw_hand / draw_measurements : 21 points, lines aur faasle frame par draw karna
  - make_callback   : webrtc ke liye frame callback
"""

import csv
import math
import threading
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.predict import MODEL_FILE          # static model badalna ho to sirf wahan badlo

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_rf_T8.joblib"
DYNAMIC_LOG = ROOT / "results" / "live_dynamic_log.csv"

DYNAMIC_MAP = {"2-Hay": "ھ", "Alifmad": "آ", "Aray": "ڑ", "Jeem": "ج"}

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


# --------------------------------------------------------------------------- static prediction
def normalize63(lm):
    """Raw 63 numbers -> (1, 63) normalized: wrist origin par, wrist->point 9 distance se divide."""
    pts = np.asarray(lm, dtype=float).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-9:
        scale = 1.0
    return (pts / scale).reshape(1, 63)


class LivePredictor:
    """Wahi static model jo app mein deploy hai (src/predict.py ka MODEL_FILE)."""

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
    Dynamic recording ke dauran (paused) static kuch nahi jodta; uske baad pehli static shakal lock hoti hai.
    """

    MISS_RESET = 8      # itne frames lagataar haath na mile to "haath hata diya" maana jata hai
    SWITCH_AFTER = 3    # itne frames lagataar dusra harf aaye tabhi candidate badalta hai (flicker se bachao)

    def __init__(self):
        self.lock = threading.Lock()
        self.letters = []           # class names, jaise "Alif" ya "Jeem"; space ke liye " "
        self.top3 = []
        self.has_hand = False
        self.fps = 0.0
        self.hold = 0.0             # 0..1
        self.locked = None          # aakhri jura hua harf
        self.paused = False         # dynamic recording chal rahi ho to True
        self.wait_change = False    # dynamic ke baad: pehli static shakal na jode
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

    def _set_cand(self, cur, now):
        self._cand, self._cand_start, self._flick = cur, now, 0
        if self.wait_change:                 # dynamic ke baad wali shakal: lock, jodo mat
            self.locked = cur
            self.wait_change = False

    def update(self, top3, now):
        """top3 = [(name, prob), ...] ya None (haath nahi mila)."""
        with self.lock:
            if self.paused:
                self.has_hand = top3 is not None
                self.top3 = top3 or []
                self._reset_candidate()
                return

            if top3 is None:
                self.has_hand = False
                self.top3 = []
                self._miss += 1
                if self._miss >= self.MISS_RESET:
                    self._reset_candidate()
                    self.locked = None
                    self.wait_change = False
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
                self._set_cand(cur, now)
            else:
                self._flick += 1
                if self._flick >= self.SWITCH_AFTER:
                    self._set_cand(cur, now)

            if self._cand is None:
                return
            self.hold = min((now - self._cand_start) / self.hold_seconds, 1.0)
            if self.hold >= 1.0 and self._cand != self.locked:
                self.letters.append(self._cand)
                self.locked = self._cand

    def after_dynamic(self, name=None):
        """Dynamic recording khatam: harf qubool hua ho to jodo, phir static ko agli shakal tak roko."""
        with self.lock:
            if name:
                self.letters.append(name)
            self._reset_candidate()
            self.locked = None
            self.wait_change = True

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


# --------------------------------------------------------------------------- dynamic (Tareeqa B)
def features_from_landmarks(arr):
    """
    Live landmarks (n, 63, NaN = haath nahi) -> (1, 520), BILKUL wahi function jo training mein chala.
    video_features() andar video_landmarks(video) bulata hai; hum thori der ke liye usay
    "ye lo, ye rahe landmarks" se badal dete hain. Baqi sab (normalize, wrist harkat, resample) wahi.
    """
    import src.video_features as vf
    original = vf.video_landmarks
    vf.video_landmarks = lambda *args, **kwargs: arr
    try:
        return vf.video_features("live", T=vf.T_STEPS)
    finally:
        vf.video_landmarks = original


class DynamicRecorder:
    COUNTDOWN = 3.0
    SAMPLE_DT = 1.0 / 15     # training: 30 FPS video ka har doosra frame = 15 frame per second
    SHOW_RESULT = 2.5        # nateeja video par kitni der dikhe

    def __init__(self, model, log_path=DYNAMIC_LOG):
        self.model = model
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        self.phase = "idle"              # idle -> countdown -> recording -> idle
        self.result = None
        self.true_label = None
        self._t0 = 0.0
        self._next_keep = 0.0
        self._frames = []
        self._result_t = 0.0
        # settings (page se badalti hain)
        self.record_seconds = 3.0
        self.min_conf = 0.60
        self.aspect_fix = True

    def start(self, true_label=None):
        """Button dabane par (Streamlit thread)."""
        with self.lock:
            if self.phase in ("countdown", "recording"):
                return
            self.phase = "countdown"
            self._t0 = time.time()
            self.true_label = true_label
            self._frames = []

    def busy(self):
        return self.phase in ("countdown", "recording")

    def status(self, now):
        with self.lock:
            if self.phase == "countdown":
                return "countdown", self.COUNTDOWN - (now - self._t0)
            if self.phase == "recording":
                return "recording", now - self._t0
            if self.result is not None and now - self._result_t < self.SHOW_RESULT:
                return "result", 0.0
            return "idle", 0.0

    def _fix(self, lm, w, h):
        """
        Training videos chaure (16:9) the, webcam aksar 4:3 deta hai. MediaPipe ke x, y frame ki
        chaurai/unchai se divide hote hain, isliye y ko itna scale karte hain jaise frame 16:9 hota.
        """
        out = list(lm)
        if self.aspect_fix:
            k = h / (w * 9.0 / 16.0)
            for i in range(1, 63, 3):
                out[i] *= k
        return out

    def step(self, lm, w, h, now):
        """Har frame par (callback thread). Recording khatam ho to result dict, warna None."""
        with self.lock:
            if self.phase == "countdown":
                if now - self._t0 >= self.COUNTDOWN:
                    self.phase = "recording"
                    self._t0 = now
                    self._next_keep = now
                    self._frames = []
                return None
            if self.phase != "recording":
                return None

            if now >= self._next_keep:                       # ~15 frame per second rakho
                self._frames.append(self._fix(lm, w, h) if lm is not None else [np.nan] * 63)
                self._next_keep = max(self._next_keep + self.SAMPLE_DT, now - self.SAMPLE_DT)
            if now - self._t0 < self.record_seconds:
                return None

            frames, self._frames = self._frames, []
            self.phase = "idle"
            info = {"true": self.true_label, "aspect_fix": self.aspect_fix, "seconds": self.record_seconds}

        result = self._classify(np.array(frames, dtype=float).reshape(-1, 63), info)
        with self.lock:
            self.result = result
            self._result_t = now
        try:
            self._log(result)
        except OSError:
            pass                                             # CSV khuli ho (Excel) to app na ruke
        return result

    def _classify(self, arr, info):
        n_hand = int((~np.isnan(arr[:, 0])).sum())
        r = {**info, "n_frames": len(arr), "n_hand": n_hand, "ok": False, "pred": None, "conf": 0.0, "top2": []}
        X = features_from_landmarks(arr) if n_hand >= 5 else None
        if X is None:
            return r
        proba = self.model.predict_proba(X)[0]
        idx = np.argsort(proba)[::-1][:2]
        r["top2"] = [(str(self.model.classes_[i]), float(proba[i])) for i in idx]
        r["pred"], r["conf"], r["ok"] = r["top2"][0][0], r["top2"][0][1], True
        return r

    def _log(self, r):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        second = r["top2"][1] if len(r["top2"]) > 1 else ("", float("nan"))
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "true", "pred", "conf", "second", "second_conf",
                             "n_frames", "n_hand", "aspect_fix", "seconds"])
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), r["true"] or "", r["pred"] or "",
                         f"{r['conf']:.3f}", second[0], f"{second[1]:.3f}",
                         r["n_frames"], r["n_hand"], int(r["aspect_fix"]), r["seconds"]])


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


def _center_text(img, text, scale, color, thick):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 6)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


# --------------------------------------------------------------------------- webrtc callback
def make_callback(state, tracker, predictor, recorder):
    import av

    fps = {"t": time.time()}

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        h, w = img.shape[:2]

        lm = tracker.detect(img)                       # asli (bina ulte) frame par, jaise training mein
        state.paused = recorder.busy()                 # recording ke dauran static kuch nahi jodega
        top3 = predictor.top3(lm) if lm is not None else None
        state.update(top3, now)

        res = recorder.step(lm, w, h, now)
        if res is not None:
            accepted = res["ok"] and res["conf"] >= recorder.min_conf
            state.after_dynamic(res["pred"] if accepted else None)

        img = cv2.flip(img, 1)                         # sirf dikhane ke liye shishe jaisa
        if lm is not None:
            lm_view = [1.0 - v if i % 3 == 0 else v for i, v in enumerate(lm)]   # x ulta
            if state.show_points:
                draw_hand(img, lm_view)
            if state.show_dist:
                draw_measurements(img, lm_view)

        phase, val = recorder.status(now)
        if phase == "countdown":
            _center_text(img, str(max(1, math.ceil(val))), 5.0, (0, 255, 255), 10)
            cv2.putText(img, "Tayyar ho jao...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        elif phase == "recording":
            cv2.circle(img, (30, 62), 12, (0, 0, 255), -1)
            cv2.putText(img, f"REC {val:.1f} / {recorder.record_seconds:.1f}s", (50, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            frac = min(val / recorder.record_seconds, 1.0)
            cv2.rectangle(img, (0, h - 14), (int(w * frac), h), (0, 0, 255), -1)
        elif phase == "result":
            r = recorder.result
            if r["ok"]:
                ok = r["conf"] >= recorder.min_conf
                text = f"{r['pred']} {r['conf'] * 100:.0f}%" + ("" if ok else " (kam yaqeen)")
                cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0) if ok else (0, 165, 255), 3)
            else:
                cv2.putText(img, "Haath kam mila, dobara", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        else:
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