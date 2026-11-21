"""
make_dynamic_guide.py  -  makes one small animated GIF per dynamic letter for the app (Letter guide + Practice).

Needs the UAlpha40 zip (the same one notebook 15 uses) and the project's own model, so it picks a CLEAR example:
for each letter it scores up to 12 videos with the dynamic model and keeps the one the model is most sure about.

Run from the project root (the app must be closed or restarted afterwards):
    python make_dynamic_guide.py                       # finds D:\\Downloads\\*UAlpha40*.zip
    python make_dynamic_guide.py "D:\\path\\to\\UAlpha40.zip"

Writes app/assets/dynamic/<letter>.gif (about 14 frames, 300 px wide). The app works without them (it shows a placeholder).
"""

import glob
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.live import DYNAMIC_MAP, DYNAMIC_MODEL_PATH          # noqa: E402
from src.video_features import video_features                 # noqa: E402

OUT = ROOT / "app" / "assets" / "dynamic"
N_CANDIDATES = 12      # videos scored per letter
N_FRAMES = 14          # frames in the GIF
WIDTH = 300            # GIF width in pixels
FRAME_MS = 130         # time per GIF frame


def find_zip(arg):
    if arg:
        return Path(arg)
    found = [Path(p) for p in glob.glob(r"D:\Downloads\*.zip") + glob.glob(str(Path.home() / "Downloads" / "*.zip"))
             if "UAlpha40" in Path(p).name]
    if not found:
        sys.exit("UAlpha40 zip not found. Give its path:  python make_dynamic_guide.py \"D:\\path\\UAlpha40.zip\"")
    return found[0]


def score_video(model, path, letter):
    """Model probability of the true letter for one video (0 if no hand was found)."""
    try:
        X = video_features(path)
    except Exception as e:                                       # a broken video must not stop the script
        print("   skipped:", Path(path).name, e)
        return 0.0
    if X is None or np.isnan(X).any():
        return 0.0
    classes = [str(c) for c in model.classes_]
    return float(model.predict_proba(X)[0][classes.index(letter)])


def make_gif(video_path, out_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError("video has no frames")
    idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
    pics = []
    for i in idx:
        f = frames[i]
        h, w = f.shape[:2]
        f = cv2.resize(f, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        pics.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=96))
    OUT.mkdir(parents=True, exist_ok=True)
    pics[0].save(out_path, save_all=True, append_images=pics[1:], duration=FRAME_MS, loop=0, optimize=True)


def main():
    zpath = find_zip(sys.argv[1] if len(sys.argv) > 1 else None)
    print("zip:", zpath)
    model = joblib.load(DYNAMIC_MODEL_PATH)
    zf = zipfile.ZipFile(zpath)
    names = [n for n in zf.namelist() if not n.endswith("/")]
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    with tempfile.TemporaryDirectory() as tmp:
        for letter in DYNAMIC_MAP:
            vids = sorted(n for n in names if f"/{letter}/" in n and Path(n).suffix.lower() in exts)
            if not vids:
                print(f"{letter}: no videos found in the zip, skipped")
                continue
            step = max(1, len(vids) // N_CANDIDATES)             # spread over many signers
            best = (-1.0, None)
            for n in vids[::step][:N_CANDIDATES]:
                p = Path(tmp) / ("v" + Path(n).suffix)
                p.write_bytes(zf.read(n))
                sc = score_video(model, p, letter)
                if sc > best[0]:
                    best = (sc, n)
            score, name = best
            p = Path(tmp) / ("best" + Path(name).suffix)
            p.write_bytes(zf.read(name))
            make_gif(p, OUT / f"{letter}.gif")
            kb = (OUT / f"{letter}.gif").stat().st_size / 1024
            print(f"{letter}: used {Path(name).name} (model sure {score:.2f}) -> {letter}.gif ({kb:.0f} KB)")
    print("Done. Restart the app to see the GIFs.")


if __name__ == "__main__":
    main()
