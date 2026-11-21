"""
coach_eval.py  -  checks the PSL Coach without a camera, signer-wise (no person in both parts).
Used by notebooks/17_coach.ipynb.

Test idea:
  - 5-fold GroupKFold over the 66 training signers (the sealed test set is not used).
  - The coach's reference angles AND the static model (same pipeline as the deployed one, with mirror
    augmentation) are rebuilt on the training folds only.
  - For every photo of a held-out signer we make "mistakes" on purpose: we bend ONE finger more,
    or straighten it, by `theta` degrees at its middle joint (the rest of the hand stays the same).
  - The app logic is copied: the model decides "correct" (target probability >= 0.60) or "not right yet";
    only then the coach's tips are shown (plus an optional "even better" tip when the model accepts).
  - We measure: does the tip name the finger we changed? Is the direction right (bend more / straighten)?
    How often does a correct photo get a wrong tip?

Run from the project root:  python -m src.coach_eval
Writes results/coach_eval.csv and results/coach_eval_by_finger.csv
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.coach import CHAINS, FINGERS, Z_POLISH, Z_TIP, CoachReference, advise, hand_angles
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.features import normalize_landmarks
from src.live import GROUPS, MEMBER_OF
from src.predict import MODEL_FILE

RESULTS_DIR = PROJECT_ROOT / "results"


def _rotate(v, axis, theta):
    """Rodrigues rotation of vectors v (n, 3) around a unit axis by theta radians."""
    k = axis / (np.linalg.norm(axis) + 1e-12)
    return (v * np.cos(theta) + np.cross(k, v) * np.sin(theta)
            + np.outer(v @ k, k) * (1 - np.cos(theta)))


def bend_finger(x63, finger, joint, delta_deg):
    """
    Returns a copy of the hand where `finger` is bent by `delta_deg` more (positive) or less (negative)
    at `joint` (0, 1, 2 = first, middle, last joint of that finger). Returns None if it is not possible.
    """
    P = np.asarray(x63, dtype=float).reshape(21, 3).copy()
    chain = CHAINS[finger]
    k = joint + 1
    j, prev, nxt = chain[k], chain[k - 1], chain[k + 1]
    a, b = P[prev] - P[j], P[nxt] - P[j]
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-6:                     # finger perfectly straight: bend across the palm
        axis = np.cross(a, P[17] - P[5])
    before = hand_angles(P.reshape(-1))[3 * finger + joint]
    if delta_deg < 0 and before < abs(delta_deg) + 5:   # cannot straighten an already straight joint
        return None
    best = None
    for sign in (1, -1):                                # pick the rotation direction that does what we want
        Q = P.copy()
        down = list(chain[k + 1:])
        Q[down] = P[j] + _rotate(P[down] - P[j], axis, np.radians(sign * abs(delta_deg)))
        after = hand_angles(Q.reshape(-1))[3 * finger + joint]
        change = after - before
        if np.sign(change) == np.sign(delta_deg) and (best is None or abs(change - delta_deg) < best[0]):
            best = (abs(change - delta_deg), Q)
    return None if best is None else best[1].reshape(-1)


def tip_direction_ok(result, finger, delta_deg):
    """The tip about this finger must say 'Straighten' if we bent it more, 'Bend ... more' if we straightened it."""
    want = "Straighten" if delta_deg > 0 else "Bend"
    return any(t.startswith(want) and f" {FINGERS[finger]}" in t for t in result["tips"])


def _mirror(X):
    M = np.array(X, dtype=float, copy=True)
    M[:, 0::3] *= -1                                    # left hand: flip every x (as in notebooks 09-16)
    return M


def _target_prob(model, X, letters):
    """Model probability of the target letter for each row (Tuey/Daal: the two are one group, as in the app)."""
    P = model.predict_proba(X)
    names = [str(c) for c in model.classes_]
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros(len(X))
    for r, t in enumerate(letters):
        members = GROUPS.get(MEMBER_OF.get(t, ""), (t,))
        out[r] = sum(P[r, idx[m]] for m in members if m in idx)
    return out


def run(theta=40.0, z_tip=Z_TIP, min_conf=0.60, n_splits=5, joint=1):
    """
    For every held-out photo: the unchanged photo + one changed copy per finger and direction.
    The app logic is copied: the static model decides whether the sign is right (probability of the
    target >= min_conf); only if it is not, the coach's tips are shown.
    """
    df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    cols = [f"f{i}" for i in range(63)]
    X, y, g = df[cols].values, df["label"].values, df["signer_id"].values
    base_model = joblib.load(MODELS_DIR / MODEL_FILE)

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, g)):
        ref = CoachReference.fit(X[tr], y[tr])
        model = clone(base_model).fit(np.vstack([X[tr], _mirror(X[tr])]), np.concatenate([y[tr], y[tr]]))
        cases, hands = [], []
        for i in te:
            cases.append((i, "unchanged", None, 0.0))
            hands.append(X[i])
            for f in range(5):
                for d in (theta, -theta):
                    x2 = bend_finger(X[i], f, joint, d)
                    if x2 is not None:
                        cases.append((i, "bent more" if d > 0 else "straightened", f, d))
                        hands.append(normalize_landmarks(x2))
        hands = np.array(hands)
        probs = _target_prob(model, hands, [y[c[0]] for c in cases])
        for (i, case, f, d), x2, p in zip(cases, hands, probs):
            r = advise(ref, hand_angles(x2), y[i], z_tip=z_tip)
            accepted = p >= min_conf
            row = {"fold": fold, "letter": y[i], "case": case, "finger": "" if f is None else FINGERS[f],
                   "delta": d, "target_prob": p, "accepted": accepted,
                   "tip_given": not r["ok"], "shown_tip": (not accepted) and (not r["ok"]),
                   "polish_tip": accepted and r["max_z"] > Z_POLISH}
            if f is not None:
                row["finger_ok"] = r["worst"] == f
                row["direction_ok"] = tip_direction_ok(r, f, d)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(res):
    un = res[res["case"] == "unchanged"]
    ch = res[res["case"] != "unchanged"]
    caught = ch[~ch["accepted"]]                       # the app says "not right yet" -> the coach speaks
    polished = ch[ch["polish_tip"]]                     # accepted, but an "even better" tip is shown
    summary = pd.DataFrame([
        {"measure": "correct photo: app says correct (model accepts)", "value": un["accepted"].mean(), "n": len(un)},
        {"measure": "correct photo: a wrong tip is shown", "value": un["shown_tip"].mean(), "n": len(un)},
        {"measure": "changed finger: app says not right yet", "value": 1 - ch["accepted"].mean(), "n": len(ch)},
        {"measure": "caught: coach names the changed finger", "value": caught["finger_ok"].mean(), "n": len(caught)},
        {"measure": "caught: tip has the right direction", "value": caught["direction_ok"].mean(), "n": len(caught)},
        {"measure": "correct photo: an 'even better' tip is shown", "value": un["polish_tip"].mean(), "n": len(un)},
        {"measure": "changed but accepted: 'even better' tip is shown",
         "value": ch[ch["accepted"]]["polish_tip"].mean(), "n": int(ch["accepted"].sum())},
        {"measure": "'even better' tip: names the changed finger", "value": polished["finger_ok"].mean(),
         "n": len(polished)},
        {"measure": "all changed: coach names the changed finger (no model gate)", "value": ch["finger_ok"].mean(),
         "n": len(ch)},
    ])
    by_finger = (caught.groupby(["finger", "case"])
                 .agg(n=("finger_ok", "size"), finger_ok=("finger_ok", "mean"), direction_ok=("direction_ok", "mean"))
                 .reset_index())
    return summary, by_finger


if __name__ == "__main__":
    out = []
    for theta in (25.0, 40.0, 60.0):
        res = run(theta=theta)
        s, b = summarize(res)
        s.insert(0, "theta_deg", theta)
        b.insert(0, "theta_deg", theta)
        out.append((s, b))
        print(f"\n=== theta = {theta:.0f} degrees ===")
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat([s for s, _ in out]).to_csv(RESULTS_DIR / "coach_eval.csv", index=False)
    pd.concat([b for _, b in out]).to_csv(RESULTS_DIR / "coach_eval_by_finger.csv", index=False)
    print("\nSaved results/coach_eval.csv and results/coach_eval_by_finger.csv")
