import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "app" / "assets" / "letters"
OUT.mkdir(parents=True, exist_ok=True)
SKIP = {".venv", "venv", ".git", "__pycache__", "node_modules"}


def walk(pattern):
    for p in ROOT.rglob(pattern):
        if not (set(p.parts) & SKIP):
            yield p


# 1) find train.csv
csvs = list(walk("train.csv"))
print("train.csv found:", csvs)
df = pd.read_csv(csvs[0])
feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
print("rows:", len(df), "| feature columns:", len(feat_cols))

# 2) index of all images (name -> path)
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
index = {}
for p in walk("*"):
    if p.is_file() and p.suffix.lower() in exts and "assets" not in p.parts:
        index.setdefault(p.name, p)
print("images found:", len(index))


def read_img(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def square_thumb(img, size=220):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


saved, missing = [], []
for label, g in df.groupby("label"):
    X = g[feat_cols].to_numpy(dtype=float)
    d = np.linalg.norm(X - X.mean(axis=0), axis=1)
    img = None
    for i in np.argsort(d):                       # most typical first
        name = Path(str(g.iloc[i]["filename"])).name
        if name in index:
            img = read_img(index[name])
            if img is not None:
                break
    if img is None:
        missing.append(label)
        continue
    cv2.imwrite(str(OUT / f"{label}.jpg"), square_thumb(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
    saved.append(label)

print("\nCreated:", len(saved), "->", OUT)
print("Missing:", missing)