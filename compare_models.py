"""
compare_models.py  -  trains the 4 models again, live, and compares them on the sealed test set.

Run from the project root (venv active):
    python compare_models.py                 # same setting as the results table in the slides (raw 63 numbers)
    python compare_models.py --final         # setting of the real app: 93 features + mirror training
    python compare_models.py --cv            # also signer-wise 5-fold cross-validation on the training signers (slower)
    python compare_models.py --plot          # also show a bar chart

It does NOT change any file in the project. The hyper-parameters are the tuned ones from results/tuning_results.csv.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED
from src.features import add_engineered_features

COLS = [f"f{i}" for i in range(63)]


def make_models():
    def pipe(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    return {
        "Baseline (always the same letter)": DummyClassifier(strategy="most_frequent"),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=3)),
        "SVM (RBF)": pipe(SVC(kernel="rbf", C=100, gamma=0.01)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, max_depth=20, random_state=RANDOM_SEED)),
        "MLP": pipe(MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001, max_iter=1000,
                                  random_state=RANDOM_SEED)),
    }


def mirror(X):
    """A left hand is the mirror image of a right hand: flip every x."""
    M = np.array(X, dtype=float, copy=True)
    M[:, 0::3] *= -1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="93 features + mirror training (like the app)")
    ap.add_argument("--cv", action="store_true", help="also run signer-wise 5-fold cross-validation")
    ap.add_argument("--plot", action="store_true", help="show a bar chart")
    args = ap.parse_args()

    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_tr, y_tr, g_tr = train[COLS].to_numpy(float), train["label"].astype(str).to_numpy(), train["signer_id"].to_numpy()
    X_te, y_te = test[COLS].to_numpy(float), test["label"].astype(str).to_numpy()

    if args.final:
        # same as the app: training set = photos + mirrored photos, then 63 -> 93 features
        X_fit = add_engineered_features(np.vstack([X_tr, mirror(X_tr)]))
        y_fit = np.concatenate([y_tr, y_tr])
        X_eval = add_engineered_features(X_te)
        X_eval_left = add_engineered_features(mirror(X_te))
        setting = "FINAL setting: 93 features + mirror training"
    else:
        X_fit, y_fit, X_eval, X_eval_left = X_tr, y_tr, X_te, mirror(X_te)
        setting = "SLIDE setting: raw 63 numbers, no mirror training"

    print(f"\n{setting}")
    print(f"train: {len(X_tr)} photos ({len(set(g_tr))} signers)   test: {len(X_te)} photos (16 other signers)\n")

    DummyClassifier().fit(X_fit[:20], y_fit[:20])     # warm-up, so the first model's timing is not unfair

    rows = []
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(X_fit, y_fit)
        t_fit = time.time() - t0

        t0 = time.time()
        pred = model.predict(X_eval)
        t_pred = (time.time() - t0) / len(X_eval) * 1000          # milliseconds for ONE hand
        pred_left = model.predict(X_eval_left)

        row = {
            "Model": name,
            "Test acc (right hand)": accuracy_score(y_te, pred),
            "Test acc (left hand)": accuracy_score(y_te, pred_left),
            "F1 macro": f1_score(y_te, pred, average="macro", zero_division=0),
            "Train time (s)": t_fit,
            "Predict (ms/hand)": t_pred,
        }
        if args.cv:
            # signer-wise: no signer is in both parts of a fold. Mirror copies (final setting) are made
            # from the training part of each fold only, so nothing leaks into the validation part.
            accs = []
            for tr, va in GroupKFold(n_splits=5).split(X_tr, y_tr, g_tr):
                if args.final:
                    Xa = add_engineered_features(np.vstack([X_tr[tr], mirror(X_tr[tr])]))
                    ya = np.concatenate([y_tr[tr], y_tr[tr]])
                    Xv = add_engineered_features(X_tr[va])
                else:
                    Xa, ya, Xv = X_tr[tr], y_tr[tr], X_tr[va]
                accs.append(accuracy_score(y_tr[va], make_models()[name].fit(Xa, ya).predict(Xv)))
            row["CV acc (5 folds)"] = float(np.mean(accs))
        rows.append(row)
        print(f"  done: {name:<35} test acc {row['Test acc (right hand)']:.4f}")

    df = pd.DataFrame(rows)
    df = df.sort_values("Test acc (right hand)", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nBest model on the test set: {df.iloc[0]['Model']}  ({df.iloc[0]['Test acc (right hand)']:.1%})")
    if args.cv:
        best_cv = df.sort_values("CV acc (5 folds)", ascending=False).iloc[0]
        print(f"Best model in cross-validation (training signers only): {best_cv['Model']}  ({best_cv['CV acc (5 folds)']:.1%})")
        if best_cv["Model"] != df.iloc[0]["Model"]:
            gap = (df.iloc[0]["Test acc (right hand)"] - df[df["Model"] == best_cv["Model"]].iloc[0]["Test acc (right hand)"])
            print(f"Note: CV and test disagree. The gap on the test set is {gap * len(X_te):.0f} photos out of {len(X_te)}.")
            print("      We choose the model by cross-validation; the test set is opened once, only to check (not to choose).")

    if args.plot:
        import matplotlib.pyplot as plt
        d = df[~df["Model"].str.startswith("Baseline")]
        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(d["Model"], d["Test acc (right hand)"] * 100, color=["#F5A524"] + ["#12857C"] * (len(d) - 1))
        for b, v in zip(bars, d["Test acc (right hand)"] * 100):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center")
        plt.ylim(0, 100)
        plt.ylabel("Test accuracy (%)")
        plt.title(setting)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()