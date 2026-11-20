"""
live.py

For the live webcam:
  - LiveHandTracker : MediaPipe HandLandmarker in VIDEO mode (frame-to-frame tracking)
  - LivePredictor   : 63 landmarks -> normalize -> static model -> top-3 letters
  - WordState       : average of probabilities + the letter is added by itself after 1 second on the same letter (thread-safe);
                      treats a pair like Tuey/Daal as one group and asks the user
  - DynamicRecorder : Method B - button, 3-2-1, then 3 seconds of recording -> dynamic model
  - draw_hand / draw_measurements : draw 21 points, lines and distances on the frame
  - make_callback   : frame callback for webrtc
"""

import csv
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.predict import MODEL_FILE          # to change the static model, change it only there

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_rf_T8.joblib"
DYNAMIC_LOG = ROOT / "results" / "live_dynamic_log.csv"
AUTO_LOG = ROOT / "results" / "live_dynamic_auto_log.csv"      # Auto mode tests

# Letters that landmarks cannot separate (notebook 16): the model treats them as one group and asks the user
GROUPS = {"Tuey/Daal": ("Tuey", "Daal")}
MEMBER_OF = {m: k for k, ms in GROUPS.items() for m in ms}


def pick_names(pending):
    """pending [(name, prob), ...] -> names in GROUPS order (so 1 is always Tuey and 2 is always Daal)."""
    order = [m for ms in GROUPS.values() for m in ms]
    return sorted((n for n, _ in pending), key=order.index)

DYNAMIC_MAP = {"2-Hay": "ھ", "Alifmad": "آ", "Aray": "ڑ", "Jeem": "ج"}

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIPS = [4, 8, 12, 16, 20]


# --------------------------------------------------------------------------- tracker
class LiveHandTracker:
    """One HandLandmarker that runs on live frames."""

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
        """BGR frame -> 63 raw numbers [x0,y0,z0, ...] (same format as extract_landmarks), or None."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # the timestamp must always increase
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        return [v for p in result.hand_landmarks[0] for v in (p.x, p.y, p.z)]


# --------------------------------------------------------------------------- static prediction
def normalize63(lm):
    """Raw 63 numbers -> (1, 63) normalized: wrist at the origin, divided by the wrist->point 9 distance."""
    pts = np.asarray(lm, dtype=float).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-9:
        scale = 1.0
    return (pts / scale).reshape(1, 63)


class LivePredictor:
    """The same static model that is deployed in the app (MODEL_FILE in src/predict.py)."""

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / MODEL_FILE)
        self.names = joblib.load(MODELS_DIR / "class_names.joblib")

    def _name(self, c):
        if isinstance(c, (int, np.integer)):
            return str(self.names[int(c)])
        return str(c)

    def proba(self, lm):
        """63 raw numbers -> {letter_name: probability} (all 36)."""
        p = self.model.predict_proba(normalize63(lm))[0]
        return {self._name(c): float(v) for c, v in zip(self.model.classes_, p)}

    def top3(self, lm):
        """63 raw numbers -> [(letter_name, probability), ...] up to 3."""
        proba = self.model.predict_proba(normalize63(lm))[0]
        idx = np.argsort(proba)[::-1][:3]
        return [(self._name(self.model.classes_[i]), float(proba[i])) for i in idx]


# --------------------------------------------------------------------------- word state
def _top3(probs):
    return sorted(probs.items(), key=lambda t: -t[1])[:3]


class WordState:
    """
    The callback thread writes and the Streamlit page reads, so everything is inside a lock.

    Rule: the AVERAGE of the probabilities over the last smooth_seconds is used (not the argmax of each frame).
    If the top letter is above min_conf and stays the same for hold_seconds -> the letter is added to the word.
    If the top letter is a GROUP (Tuey/Daal), no letter is added; both go into `pending` and
    the user picks with a button.
    The same letter is added again only when (a) you remove your hand and bring it back, or (b) another letter is added first.
    During a dynamic recording (paused), static adds nothing; after it, the first static shape is locked.
    That lock is cleared only when the hand is really removed for REMOVE_AFTER_DYNAMIC seconds (a short drop-out
    caused by motion blur must not unlock it), or when another letter is added.
    """

    MISS_RESET = 8      # if no hand is found for this many frames in a row, it counts as "hand removed"
    REMOVE_AFTER_DYNAMIC = 1.0   # after a dynamic gesture the hand must be away this long (seconds) to unlock
    SWITCH_AFTER = 3    # the candidate changes only if another letter appears for this many frames in a row (avoids flicker)

    def __init__(self):
        self.lock = threading.Lock()
        self.letters = []           # class names, e.g. "Alif" or "Jeem"; " " for a space
        self.top3 = []
        self.top_label = ""         # to show on the video ("Tuey/Daal ?" for a group)
        self.top_conf = 0.0
        self.pending = None         # [(name, prob), (name, prob)] when the user must choose Tuey/Daal
        self.has_hand = False
        self.fps = 0.0
        self.hold = 0.0             # 0..1
        self.locked = None          # last added letter (or group)
        self.paused = False         # True while a dynamic recording is running
        self.wait_change = False    # after dynamic: do not add the first static shape
        self.post_dynamic = False   # after dynamic: the locked shape stays locked through short drop-outs
        self._win = deque()         # (time, probs) for the last smooth_seconds
        self._cand = None
        self._cand_start = 0.0
        self._flick = 0
        self._miss = 0
        self._miss_t0 = 0.0         # when the current run of "no hand" frames started
        # settings (changed from the page)
        self.hold_seconds = 1.0
        self.smooth_seconds = 0.4
        self.min_conf = 0.60
        self.show_points = True
        self.show_dist = True

    def _reset_candidate(self):
        self._cand = None
        self._flick = 0
        self.hold = 0.0

    def _set_cand(self, cur, now):
        self._cand, self._cand_start, self._flick = cur, now, 0
        if self.wait_change:                 # the shape after dynamic: lock it, do not add it
            self.locked = cur
            self.wait_change = False

    def _average(self, probs, now):
        self._win.append((now, probs))
        while self._win and now - self._win[0][0] > self.smooth_seconds:
            self._win.popleft()
        n = len(self._win)
        return {k: sum(f[k] for _, f in self._win) / n for k in probs}

    def update(self, probs, now):
        """probs = {letter: probability} or None (no hand found)."""
        with self.lock:
            if self.paused:
                self.has_hand = probs is not None
                self.top3 = _top3(probs) if probs else []
                self.top_label, self.top_conf = "", 0.0
                self._win.clear()
                self._reset_candidate()
                return

            if probs is None:
                self.has_hand = False
                self.top3 = []
                self.top_label, self.top_conf = "", 0.0
                if self._miss == 0:
                    self._miss_t0 = now
                self._miss += 1
                if self._miss >= self.MISS_RESET:
                    self._win.clear()
                    self._reset_candidate()
                    if not self.post_dynamic or now - self._miss_t0 >= self.REMOVE_AFTER_DYNAMIC:
                        self.locked = None
                        self.wait_change = False
                        self.post_dynamic = False
                return

            self._miss = 0
            self.has_hand = True
            avg = self._average(probs, now)
            self.top3 = _top3(avg)

            units = {}                                       # add up the probabilities of the group members
            for n, p in avg.items():
                k = MEMBER_OF.get(n, n)
                units[k] = units.get(k, 0.0) + p
            key = max(units, key=units.get)
            conf = units[key]
            self.top_label = key + " ?" if key in GROUPS else key
            self.top_conf = conf
            cur = key if conf >= self.min_conf else None

            if cur is None:                                  # low confidence
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
                self.locked = self._cand
                if self._cand in GROUPS:                     # the model cannot tell them apart: ask the user
                    self.pending = sorted(((m, avg[m]) for m in GROUPS[self._cand]), key=lambda t: -t[1])
                else:
                    self.letters.append(self._cand)
                    self.pending = None
                self.post_dynamic = False                    # a real static letter was chosen: normal rules again

    def after_dynamic(self, name=None):
        """Dynamic recording finished: add the letter if it was accepted, then hold static until the next shape."""
        with self.lock:
            if name:
                self.letters.append(name)
                self.pending = None
            self._win.clear()
            self._reset_candidate()
            self.locked = None
            self.wait_change = True
            self.post_dynamic = True

    # buttons
    def add_letter(self, name):
        """From a button (or the answer to a pending Tuey/Daal): add the letter directly."""
        with self.lock:
            self.letters.append(name)
            self.pending = None

    def pick_pending(self, name):
        """The 1/2 buttons above the camera: work only when a Tuey/Daal question is really pending."""
        with self.lock:
            if self.pending and name in [n for n, _ in self.pending]:
                self.letters.append(name)
                self.pending = None

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
            self.pending = None
            self.locked = None

    def snapshot(self):
        with self.lock:
            return {
                "letters": list(self.letters),
                "top3": list(self.top3),
                "top_label": self.top_label,
                "top_conf": self.top_conf,
                "pending": list(self.pending) if self.pending else None,
                "has_hand": self.has_hand,
                "hold": self.hold,
                "locked": self.locked,
            }


# --------------------------------------------------------------------------- dynamic (Method B)
def features_from_landmarks(arr):
    """
    Live landmarks (n, 63, NaN = no hand) -> (1, 520), EXACTLY the same function that ran in training.
    video_features() calls video_landmarks(video) inside; for a short time we replace it
    with "here are the landmarks". Everything else (normalize, wrist movement, resample) is the same.
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
    SAMPLE_DT = 1.0 / 15     # training: every second frame of a 30 FPS video = 15 frames per second
    SHOW_RESULT = 2.5        # how long the result stays on the video

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
        # settings (changed from the page)
        self.record_seconds = 3.0
        self.min_conf = 0.60
        self.aspect_fix = True
        # --- Auto mode (no button): detects the start / end of a gesture from the hand speed
        self.auto = False
        self.auto_label = None           # true letter for testing (from the page), None = no test
        self.v_hi = 0.80                 # above this speed (palm length / second) => movement starts
        self.v_lo = 0.35                 # below this => the hand has stopped (always lower than v_hi)
        self.min_dur = 0.9               # minimum movement time (seconds), otherwise ignored
        self.min_energy = 1.3            # total amount of movement (palm lengths), otherwise ignored
        self.still_end = 0.4             # the gesture ends if the hand stays still this long
        self.max_seg = 3.0               # maximum gesture length (seconds), same as a manual recording
        self.cooldown = 1.5              # no new gesture starts for this long after a gesture
        self.speed = 0.0                 # smoothed speed (to show on the video)
        self._reset_auto()

    def _reset_auto(self):
        self._hist = deque(maxlen=16)
        self._pre = deque(maxlen=8)      # frames from ~0.5 second earlier (so the start is not cut off)
        self._a_above = None
        self._a_frames = []
        self._a_t0 = 0.0
        self._a_last_move = 0.0
        self._a_tick = 0.0
        self._a_energy = 0.0
        self._a_lost = None
        self._cool_until = 0.0

    def start(self, true_label=None):
        """When the button is pressed (Streamlit thread)."""
        with self.lock:
            if self.phase in ("countdown", "recording"):
                return
            self.phase = "countdown"
            self._t0 = time.time()
            self.true_label = true_label
            self._frames = []
            self._reset_auto()

    def busy(self):
        return self.phase in ("countdown", "recording", "auto")

    def blocks_static(self, now):
        """
        True while the static letters must stay quiet: during a manual recording, or during an Auto gesture
        whose movement has already lasted min_dur seconds (so it will count as a real gesture). A short hand
        tremor (which Auto mode may start and then ignore) does not block the static letters.
        """
        with self.lock:
            if self.phase in ("countdown", "recording"):
                return True
            if self.phase == "auto":
                return self._a_last_move - self._a_t0 >= self.min_dur
            return False

    def result_age(self, now):
        """Seconds since the last result (a very large number if there is none)."""
        with self.lock:
            return now - self._result_t if self.result is not None else 1e9

    def status(self, now):
        with self.lock:
            if self.phase == "countdown":
                return "countdown", self.COUNTDOWN - (now - self._t0)
            if self.phase == "recording":
                return "recording", now - self._t0
            if self.phase == "auto":
                return "auto", now - self._a_t0
            if self.result is not None and now - self._result_t < self.SHOW_RESULT:
                return "result", 0.0
            return "idle", 0.0

    def _fix(self, lm, w, h):
        """
        Training videos were wide (16:9); a webcam often gives 4:3. MediaPipe's x, y are divided by the
        frame's width/height, so we scale y as if the frame were 16:9.
        """
        out = list(lm)
        if self.aspect_fix:
            k = h / (w * 9.0 / 16.0)
            for i in range(1, 63, 3):
                out[i] *= k
        return out

    def step(self, lm, w, h, now):
        """On every frame (callback thread). Returns a result dict when the recording ends, otherwise None."""
        with self.lock:
            manual = self.phase in ("countdown", "recording")
            if self.phase == "auto" and not self.auto:       # Auto mode was turned off in the middle
                self.phase = "idle"
                self._reset_auto()
        if self.auto and not manual:
            return self._auto_step(lm, w, h, now)
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

            if now >= self._next_keep:                       # keep ~15 frames per second
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
            pass                                             # so the app does not stop if the CSV is open (Excel)
        return result

    # ------------------------------------------------------------------ Auto mode
    def _speed(self, lm, w, h, now):
        """
        Hand speed: distance of the 21 points from ~0.2 second ago, in palm lengths, per second.
        (We compare with a frame 0.2 s old instead of consecutive frames, so that small MediaPipe jitter
        is not taken as movement.)
        """
        if lm is None:
            self._hist.clear()
            return None
        p = np.asarray(lm, dtype=float).reshape(21, 3)[:, :2] * (w, h)
        palm = float(np.linalg.norm(p[9] - p[0])) + 1e-6
        self._hist.append((now, p))
        ref = None
        for t_old, p_old in self._hist:                  # the oldest entry that is newer than 0.25 s
            if now - t_old <= 0.25:
                ref = (t_old, p_old)
                break
        if ref is None or now - ref[0] < 0.12:
            return None
        dt = now - ref[0]
        return float(np.linalg.norm(p - ref[1], axis=1).mean() / palm / dt)

    def _sample(self, lm, w, h, now):
        """One row at ~15 frames per second (NaN if there is no hand), otherwise None."""
        if now < self._next_keep:
            return None
        self._next_keep = max(self._next_keep + self.SAMPLE_DT, now - self.SAMPLE_DT)
        return self._fix(lm, w, h) if lm is not None else [np.nan] * 63

    def _auto_step(self, lm, w, h, now):
        v = self._speed(lm, w, h, now)
        done = None
        with self.lock:
            self.speed = 0.6 * self.speed + 0.4 * (v if v is not None else 0.0)
            sp = self.speed
            row = self._sample(lm, w, h, now)

            if self.phase == "idle":
                if row is not None:
                    self._pre.append(row)
                if now < self._cool_until:
                    return None
                if sp > self.v_hi and lm is not None:
                    if self._a_above is None:
                        self._a_above = now
                    if now - self._a_above >= 0.25:           # fast movement lasted more than 0.25 s: start
                        self.phase = "auto"
                        self._a_t0 = self._a_above
                        self._a_frames = list(self._pre)
                        self._a_last_move = now
                        self._a_tick = now
                        self._a_energy = 0.0
                        self._a_lost = None
                else:
                    self._a_above = None
                return None

            if self.phase != "auto":
                return None

            # --- gesture in progress
            if row is not None:
                self._a_frames.append(row)
            self._a_energy += sp * max(now - self._a_tick, 0.0)
            self._a_tick = now
            if sp > self.v_lo:
                self._a_last_move = now
            if lm is None:
                self._a_lost = self._a_lost or now
            else:
                self._a_lost = None
            ended = (now - self._a_last_move >= self.still_end
                     or now - self._a_t0 >= self.max_seg
                     or (self._a_lost is not None and now - self._a_lost >= 0.6))
            if not ended:
                return None

            frames, self._a_frames = self._a_frames, []
            drop = int(max(0.0, now - self._a_last_move - 0.1) / self.SAMPLE_DT)   # cut the last still part
            if 0 < drop < len(frames) - 5:
                frames = frames[:-drop]
            dur = self._a_last_move - self._a_t0
            energy = self._a_energy
            label = self.auto_label
            self.phase = "idle"
            self._reset_auto()
            self._cool_until = now + self.cooldown
            done = (frames, dur, energy, label)

        frames, dur, energy, label = done
        arr = np.array(frames, dtype=float).reshape(-1, 63)
        info = {"true": label, "aspect_fix": self.aspect_fix, "seconds": round(dur, 2)}
        why = "ok"
        if dur < self.min_dur:
            why = "short"
        elif energy < self.min_energy:
            why = "weak"
        if why == "ok":
            result = self._classify(arr, info)
            if not result["ok"]:
                why = "nohand"
        else:
            result = {**info, "n_frames": len(arr), "n_hand": int((~np.isnan(arr[:, 0])).sum()),
                      "ok": False, "pred": None, "conf": 0.0, "top2": []}
        result["ignored"] = why if why != "ok" else None
        result["why"] = why
        with self.lock:
            self.result = result
            self._result_t = now
        try:
            self._log_auto(result, dur, energy)
        except OSError:
            pass
        return result if why == "ok" else None

    def _log_auto(self, r, dur, energy):
        AUTO_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not AUTO_LOG.exists()
        second = r["top2"][1] if len(r["top2"]) > 1 else ("", float("nan"))
        with open(AUTO_LOG, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "true", "pred", "conf", "second", "second_conf", "n_frames", "n_hand",
                             "duration", "energy", "why", "v_hi", "v_lo"])
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), r["true"] or "", r["pred"] or "", f"{r['conf']:.3f}",
                         second[0], f"{second[1]:.3f}", r["n_frames"], r["n_hand"], f"{dur:.2f}", f"{energy:.2f}",
                         r["why"], self.v_hi, self.v_lo])

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
    """Draws the 21 points and their lines on the frame (fingertips are larger and red)."""
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
    Distances (in units of palm length = wrist->point 9):
      yellow lines = from the wrist to each fingertip (5)
      blue lines   = between neighbouring fingertips (4)
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
        seg(0, t, (0, 255, 255))            # yellow (BGR)
    for a, b in zip(TIPS[:-1], TIPS[1:]):
        seg(a, b, (255, 200, 0))            # blue
    return bgr


def _center_text(img, text, scale, color, thick):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 6, cv2.LINE_AA)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


# colors (BGR)
C_GREEN, C_ORANGE, C_RED, C_WHITE, C_GRAY = (94, 197, 34), (11, 158, 245), (68, 68, 239), (245, 245, 245), (184, 163, 148)


def _chip(img, text, x, y, fg=C_WHITE, scale=0.6, thick=1, pad=8, alpha=0.62):
    """Text in a half-transparent black box (easy to read on the video). Returns the right edge of the box."""
    H, W = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x0, y0 = x, y
    x1, y1 = min(x + tw + 2 * pad, W - 1), min(y + th + base + 2 * pad - 2, H - 1)
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, (20, 20, 20)), alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
    return x1


def _bar(img, frac, color, h_bar=8, margin=12):
    """A thin bar at the bottom (track + fill)."""
    H, W = img.shape[:2]
    y1, y0 = H - margin, H - margin - h_bar
    x0, x1 = margin, W - margin
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, (20, 20, 20)), 0.6, roi, 0.4, 0, dst=roi)
    if frac > 0:
        cv2.rectangle(img, (x0, y0), (x0 + int((x1 - x0) * min(frac, 1.0)), y1), color, -1)


# --------------------------------------------------------------------------- webrtc callback
def make_callback(state, tracker, predictor, recorder):
    import av

    fps = {"t": time.time()}

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        h, w = img.shape[:2]

        lm = tracker.detect(img)                       # on the original (not mirrored) frame, as in training
        state.paused = recorder.blocks_static(now)     # static adds nothing during a (confirmed) recording
        probs = predictor.proba(lm) if lm is not None else None
        state.update(probs, now)

        res = recorder.step(lm, w, h, now)
        if res is not None:
            accepted = res["ok"] and res["conf"] >= recorder.min_conf
            state.after_dynamic(res["pred"] if accepted else None)

        img = cv2.flip(img, 1)                         # mirror view, for display only
        if lm is not None:
            lm_view = [1.0 - v if i % 3 == 0 else v for i, v in enumerate(lm)]   # flip x
            if state.show_points:
                draw_hand(img, lm_view)
            if state.show_dist:
                draw_measurements(img, lm_view)

        # --- top: FPS and hand status
        state.fps = 0.9 * state.fps + 0.1 * (1.0 / max(now - fps["t"], 1e-6))
        fps["t"] = now
        x = _chip(img, f"FPS {state.fps:.0f}", 10, 10, C_GREEN if state.fps >= 10 else C_RED, scale=0.55)
        x = _chip(img, "HAND" if lm is not None else "no hand", x + 6, 10,
                  C_GREEN if lm is not None else C_GRAY, scale=0.55)
        if recorder.auto:                              # show the speed so you can set the thresholds
            fast = recorder.speed > recorder.v_hi
            _chip(img, f"AUTO  v {recorder.speed:.1f}", x + 6, 10, C_ORANGE if fast else C_GRAY, scale=0.55)

        phase, val = recorder.status(now)
        if phase == "countdown":
            _center_text(img, str(max(1, math.ceil(val))), 5.0, (0, 255, 255), 10)
            tag = f"  ->  {recorder.true_label}" if recorder.true_label else ""
            _chip(img, "Get ready" + tag, 10, 46, (0, 255, 255), scale=0.9, thick=2)
        elif phase == "recording":
            frac = min(val / recorder.record_seconds, 1.0)
            tag = f"  {recorder.true_label}" if recorder.true_label else ""
            _chip(img, f"REC {val:.1f} / {recorder.record_seconds:.1f}s{tag}", 10, 46, C_RED, scale=0.9, thick=2)
            _bar(img, frac, C_RED)
        elif phase == "auto":
            if recorder.blocks_static(now):            # only after the movement lasted a moment (no flicker)
                _chip(img, f"AUTO REC {val:.1f}s", 10, 46, C_RED, scale=0.9, thick=2)
                _bar(img, min(val / recorder.max_seg, 1.0), C_RED)
        elif phase == "result":
            r = recorder.result
            if r.get("ignored") in ("short", "weak"):
                _chip(img, "ignored: movement too short", 10, 46, C_GRAY, scale=0.8, thick=2)
            elif r["ok"]:
                ok = r["conf"] >= recorder.min_conf
                text = f"{r['pred']} {r['conf'] * 100:.0f}%" + ("" if ok else " (low confidence)")
                _chip(img, text, 10, 46, C_GREEN if ok else C_ORANGE, scale=0.9, thick=2)
            else:
                _chip(img, "Few hand frames, try again", 10, 46, C_ORANGE, scale=0.9, thick=2)
        else:
            with state.lock:
                label, conf, hold = state.top_label, state.top_conf, state.hold
                pend = list(state.pending) if state.pending else None
            if label:
                group = "?" in label
                ok = conf >= state.min_conf and not group
                text = label if group else f"{label} {conf * 100:.0f}%"
                _chip(img, text, 10, 46, C_GREEN if ok else C_ORANGE, scale=0.9, thick=2)
            if pend:
                a, b = pick_names(pend)
                _chip(img, f"1 = {a}      2 = {b}", 10, h - 62, C_ORANGE, scale=0.8, thick=2)
            _bar(img, hold, C_GREEN)                   # hold meter

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback