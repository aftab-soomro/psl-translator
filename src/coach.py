"""
coach.py  -  PSL Coach: tells a learner WHICH finger is wrong and HOW to fix it.

Idea (simple):
  1. From the training photos (data/processed/train.csv, 66 signers) we know, for every letter,
     how much each finger joint is bent and how far the fingers are spread.
     For each letter we keep the median and the normal spread between signers (robust standard deviation).
  2. Live, we measure the same 19 angles on the learner's hand.
  3. z = (learner - median) / spread. The finger with the biggest |z| is the one to fix.
     If the learner bends it more than the signers do -> "straighten it"; less -> "bend it more".
     Spread angles give "spread apart" / "bring together".
  4. The static model still decides whether the sign is right (probability of the target letter >= 0.60).
     The tips are shown when it is not; then they say what to change. When it is right but one finger is
     still very far from the photos (|z| > 4), an optional "even better" tip is shown.
     (Without this gate the coach complains on 27% of correct photos, with it only on 5%: results/coach_eval.csv.)

The 19 angles (all in degrees):
  - 15 bends: 3 joints per finger, 0 = straight, bigger = more bent (same joints as the 93 model features)
  - 4 spreads: angle between neighbouring fingers (thumb-index, index-middle, middle-ring, ring-pinky)
Angles do not change when the hand moves, gets bigger/smaller or is mirrored (left hand), so no extra
normalization is needed.
"""

import threading
from collections import deque

import numpy as np
import pandas as pd

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]          # same as src/features.py
JOINT_WORDS = ["at the knuckle", "at the middle joint", "at the tip joint"]
THUMB_JOINT_WORDS = ["at the base", "at the middle joint", "at the tip joint"]

# feature list: (kind, finger index or pair, joint index)
FEATURES = [("bend", f, j) for f in range(5) for j in range(3)] + \
           [("spread", (f, f + 1), None) for f in range(4)]
FEATURE_NAMES = [f"bend_{FINGERS[f]}_{j + 1}" for f in range(5) for j in range(3)] + \
                [f"spread_{FINGERS[f]}_{FINGERS[f + 1]}" for f in range(4)]

MIN_SPREAD_BEND = 10.0       # degrees: the smallest "normal variation" we allow for a bend
MIN_SPREAD_SPREAD = 6.0      # degrees: the same for a spread angle
Z_TIP = 2.5                  # |z| above this -> the coach gives a tip for that finger
Z_POLISH = 4.0               # when the model already accepts the sign, a tip is shown only above this |z|


def _angle(a, b):
    """Angle in degrees between vectors a and b (arrays of shape (..., 3))."""
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def hand_angles(landmarks):
    """
    63 numbers (raw or normalized) or an array (N, 63) -> 19 angles per hand, shape (19,) or (N, 19).
    """
    X = np.asarray(landmarks, dtype=float)
    single = X.ndim == 1
    P = X.reshape(-1, 21, 3)
    out = []
    for chain in CHAINS:                                    # 15 bends
        for k in range(1, 4):
            a = P[:, chain[k - 1]] - P[:, chain[k]]
            b = P[:, chain[k + 1]] - P[:, chain[k]]
            out.append(180.0 - _angle(a, b))                # 0 = straight
    for f in range(4):                                      # 4 spreads (base -> tip direction of each finger)
        c1, c2 = CHAINS[f], CHAINS[f + 1]
        d1 = P[:, c1[4]] - P[:, c1[1]]
        d2 = P[:, c2[4]] - P[:, c2[1]]
        out.append(_angle(d1, d2))
    A = np.column_stack(out)
    return A[0] if single else A


class CoachReference:
    """Median and normal spread of the 19 angles for every letter (from the training photos)."""

    def __init__(self, median, spread, counts):
        self.median = median        # {letter: (19,)}
        self.spread = spread        # {letter: (19,)}
        self.counts = counts        # {letter: number of photos}

    @classmethod
    def fit(cls, X63, labels):
        A = hand_angles(X63)
        labels = np.asarray(labels)
        floor = np.array([MIN_SPREAD_BEND] * 15 + [MIN_SPREAD_SPREAD] * 4)
        median, spread, counts = {}, {}, {}
        for letter in sorted(set(labels)):
            a = A[labels == letter]
            med = np.median(a, axis=0)
            mad = 1.4826 * np.median(np.abs(a - med), axis=0)   # robust standard deviation
            median[letter], spread[letter], counts[letter] = med, np.maximum(mad, floor), len(a)
        return cls(median, spread, counts)

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path)
        cols = [f"f{i}" for i in range(63)]
        return cls.fit(df[cols].values, df["label"].values)

    @property
    def letters(self):
        return list(self.median)


def advise(ref, angles, target, z_tip=Z_TIP):
    """
    angles: 19 angles of the learner's hand; target: letter name.
    Returns a dict:
      z            : (19,) signed z-scores
      finger_z     : [5 floats] worst |z| of each finger (its 3 bends and the spreads it is part of)
      worst        : index of the finger to fix, or None if every |z| <= z_tip
      tips         : list of up to 2 short English tips (worst first)
      ok           : True when no tip is needed
      max_z        : the biggest |z| of all 19 angles
    """
    z = (np.asarray(angles) - ref.median[target]) / ref.spread[target]
    items = []                                              # (|z|, finger, text)
    for i, (kind, f, j) in enumerate(FEATURES):
        if abs(z[i]) <= z_tip:
            continue
        if kind == "bend":
            where = (THUMB_JOINT_WORDS if f == 0 else JOINT_WORDS)[j]
            verb = "Straighten" if z[i] > 0 else "Bend"
            more = "" if z[i] > 0 else " more"
            text = f"{verb} your {FINGERS[f]}{more} {where}"
            items.append((abs(z[i]), f, text))
        else:
            a, b = f
            text = (f"Bring your {FINGERS[a]} and {FINGERS[b]} closer together" if z[i] > 0
                    else f"Spread your {FINGERS[a]} and {FINGERS[b]} apart")
            # the spread belongs to the finger that differs more from the reference in its own bends
            fa = np.abs(z[3 * a:3 * a + 3]).max()
            fb = np.abs(z[3 * b:3 * b + 3]).max()
            items.append((abs(z[i]), a if fa >= fb else b, text))

    finger_z = [float(np.abs(z[3 * f:3 * f + 3]).max()) for f in range(5)]
    for k in range(4):                                       # spreads count for both fingers
        finger_z[k] = max(finger_z[k], float(abs(z[15 + k])))
        finger_z[k + 1] = max(finger_z[k + 1], float(abs(z[15 + k])))

    # one tip per finger: a bend tip is easier to follow, so it wins over a spread tip for the same finger
    items.sort(key=lambda t: (-t[0]))
    per_finger = {}
    for score, f, text in items:
        is_bend = not text.startswith(("Bring", "Spread"))
        best = per_finger.get(f)
        if best is None or (is_bend and not best[2]):
            per_finger[f] = (score if best is None else max(score, best[0]), text, is_bend)
    order = sorted(per_finger.items(), key=lambda kv: -kv[1][0])
    tips = [v[1] for _, v in order[:2]]
    worst = order[0][0] if order else None
    return {"z": z, "finger_z": finger_z, "worst": worst, "tips": tips, "ok": not items,
            "max_z": float(np.abs(z).max())}


class Coach:
    """
    Practice mode state, shared by the camera thread (writes) and the page (reads), so it uses a lock.
    The angles are averaged over the last `smooth_seconds` so the tips do not jump around.
    """

    SUCCESS_HOLD = 1.0          # seconds the sign must stay correct to count as "done"

    def __init__(self, ref):
        self.ref = ref
        self.lock = threading.Lock()
        self.target = None          # letter name or None (practice off)
        self.smooth_seconds = 0.5
        self.min_conf = 0.60
        self._win = deque()
        self._good_since = None
        self.reset()

    def reset(self):
        self._win = deque() if not hasattr(self, "_win") else self._win
        self._win.clear()
        self._good_since = None
        self.result = None          # last advise() result
        self.match = 0.0            # model probability for the target letter (averaged)
        self.has_hand = False
        self.done = False           # the model accepted the sign for SUCCESS_HOLD seconds
        self.good_frac = 0.0        # 0..1 progress of the success hold

    def set_target(self, target):
        with self.lock:
            if target != self.target:
                self.target = target
                self.reset()

    def update(self, landmarks, probs, now, group_of=None):
        """landmarks: 63 raw numbers or None; probs: {letter: prob} from the static model or None."""
        with self.lock:
            if self.target is None:
                return
            if landmarks is None:
                self.has_hand = False
                self._win.clear()
                self._good_since = None
                self.good_frac = 0.0
                return
            self.has_hand = True
            p = 0.0
            if probs:
                members = group_of(self.target) if group_of else (self.target,)
                p = sum(probs.get(m, 0.0) for m in members)     # Tuey/Daal: the model sees them as one group
            self._win.append((now, hand_angles(landmarks), p))
            while self._win and now - self._win[0][0] > self.smooth_seconds:
                self._win.popleft()
            ang = np.mean([a for _, a, _ in self._win], axis=0)
            self.match = float(np.mean([q for _, _, q in self._win]))
            self.result = advise(self.ref, ang, self.target)
            good = self.match >= self.min_conf                  # the static model decides if the sign is right
            if good:
                if self._good_since is None:
                    self._good_since = now
                self.good_frac = min((now - self._good_since) / self.SUCCESS_HOLD, 1.0)
                if self.good_frac >= 1.0:
                    self.done = True
            else:
                self._good_since = None
                self.good_frac = 0.0

    def snapshot(self):
        with self.lock:
            r = self.result
            return {
                "target": self.target,
                "has_hand": self.has_hand,
                "match": self.match,
                "tips": list(r["tips"]) if r else [],
                "finger_z": list(r["finger_z"]) if r else [],
                "worst": r["worst"] if r else None,
                "ok": bool(r["ok"]) if r else False,
                "accepted": self.has_hand and self.match >= self.min_conf,
                # accepted, but one finger is still far from the photos: an optional "even better" tip
                "polish": bool(r and self.has_hand and self.match >= self.min_conf and r["max_z"] > Z_POLISH),
                "done": self.done,
                "good_frac": self.good_frac,
            }
