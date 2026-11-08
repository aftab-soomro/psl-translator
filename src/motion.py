"""
motion.py

Haath ki raftaar (motion meter): 21 points ki average harkat, palm-lengths per second mein.
Palm = wrist (0) se point 9 ki doori, isliye haath camera se door ho ya qareeb, number nahi badalta.
"""

from collections import deque

import numpy as np


class MotionMeter:
    def __init__(self, smooth=5, peak_window=2.0):
        self._prev = None
        self._prev_t = None
        self._recent = deque(maxlen=smooth)    # aakhri 5 frames ka average (jitter kam)
        self._history = deque()                # (waqt, speed) pichle peak_window second ke
        self.peak_window = peak_window

    def _peak(self, now):
        while self._history and now - self._history[0][0] > self.peak_window:
            self._history.popleft()
        return max((s for _, s in self._history), default=0.0)

    def update(self, lm, w, h, now):
        """lm = 63 raw numbers ya None. Wapis: (abhi ki speed, pichle 2 second ka peak)."""
        if lm is None:
            self._prev = None
            self._recent.clear()
            return 0.0, self._peak(now)

        pts = np.array([(lm[3 * i] * w, lm[3 * i + 1] * h) for i in range(21)])
        palm = np.linalg.norm(pts[9] - pts[0])
        speed = 0.0
        if self._prev is not None and palm > 1:
            dt = max(now - self._prev_t, 1e-3)
            speed = float(np.linalg.norm(pts - self._prev, axis=1).mean() / palm / dt)
        self._prev, self._prev_t = pts, now

        self._recent.append(speed)
        smooth = float(np.mean(self._recent))
        self._history.append((now, smooth))
        return smooth, self._peak(now)