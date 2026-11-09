# PSL Live Translator

Real-time **Pakistan Sign Language (PSL) Urdu-alphabet recognizer**. The webcam sees a hand, the app recognises the letter and builds an Urdu word on screen (and can read it aloud).

Machine Learning semester project, Sukkur IBA University. Authors: Aftab and Meraj.

## How it works

```
Webcam frame -> MediaPipe HandLandmarker (21 points, 63 numbers)
             -> normalisation (wrist = origin, scaled by palm size)
             -> classical ML model -> letter + confidence
             -> hold the sign for 1 second -> letter is added to the word
```

- **36 static letters:** KNN, SVM, Random Forest and MLP were compared with signer-wise cross-validation. The chosen model uses engineered landmark features (93) + StandardScaler + SVC (C=100, gamma=0.01), trained with mirror augmentation.
- **4 dynamic letters** (Jeem, Aray, Alifmad, 2-Hay): these need movement, so the user presses *Record*, a 3-2-1 countdown runs, the gesture is recorded for about 3 seconds and a Random Forest on 8 resampled time steps classifies it.
- **Tuey / Daal** look almost identical in landmarks. When the model cannot separate them, the app shows two numbered buttons (or press keyboard `1` / `2`) and the user chooses.
- The **Harf guide** in the sidebar shows one photo, the Urdu letter and the Roman name of every letter, so a new user can find a sign.

## Setup

Use the same Python version the project was developed with (`python 3.12.10`), in a fresh virtual environment.

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the app

```
python -m streamlit run app/live_app.py
```

Open the link Streamlit prints, press **START** and allow the camera in the browser. The first "Sunao" (speak) needs internet, because the audio is made with gTTS; it is cached afterwards. The camera stream also uses a public STUN server.

## Project layout

| Folder / file | What is in it |
|---|---|
| `app/live_app.py` | The single live page (camera, word builder, dynamic recording, Harf guide) |
| `app/assets/letters/` | One reference photo per letter for the sidebar guide |
| `src/` | Model code: features, prediction, live tracking (`live.py`), dynamic letters (`video_features.py`), letter map (`letters.py`) |
| `models/` | Trained models (static v3, dynamic RF) |
| `notebooks/` | Experiments, cross-validation, confusion-pair analysis |
| `results/` | Result tables and the live dynamic-letter test log |
| `reports/figures/` | Graphs and figures for the report |
| `notes/` | Project log, dataset notes, leakage audit, AI-use log, viva Q&A |
| `legacy/` | Old multi-page version of the app (kept for reference, not used) |
| `make_letter_guide.py` | Rebuilds the sidebar photos from the dataset (needs the dataset in `data/`) |
| `data/` | `raw/` dataset images (not in Git), `processed/` extracted landmarks, `custom_test/` own recorded test images (not in Git) |

## Dataset

The UAlpha40 dataset is **not** stored in this repository. Put it under `data/` to retrain or to run `make_letter_guide.py`. The app itself only needs the saved models and `app/assets/letters/`.

## Results

- Static letters, 36 classes: **95.2%** signer-wise cross-validation accuracy (the test split was kept sealed and was not used to choose the model).
- Dynamic letters (studio videos): 99.1% cross-validation, 96.7% on the held-out test videos.
- Live dynamic-letter accuracy from a real webcam is logged in `results/live_dynamic_log.csv` and shown in the app under *Hisaab*.

## Known limitations

- Tuey and Daal cannot be told apart from landmarks alone (pair accuracy about 62%); they account for about 45% of cross-validation errors. The app asks the user to choose.
- About 5.8% of dataset hands are not detected by MediaPipe.
- All training data was recorded in a studio, so other lighting, backgrounds and signers may lower accuracy.
- The test set is small (about +/-1.8 percentage points) and there is no external test set.
- There is no "no letter" class; a non-sign hand shape can still be predicted as some letter, which is why the app uses a confidence threshold and a one-second hold.
- Only 36 static + 4 dynamic letters are covered, not words or sentences.