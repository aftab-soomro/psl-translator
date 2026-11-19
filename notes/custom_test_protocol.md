# Custom Test Protocol (Phase 11: unseen signers)

**Goal:** The signers in the dataset test set (95.02% accuracy) were also from the dataset (different people, but the same camera and light). Here we check the model on photos of our classmates, who are completely new people with a new camera and a new background. We will write the difference between the two scores in the report.

**Golden rule:** Never train, tune or select the model with the photos in `data/custom_test/`. They are only the final exam.

## 1. Consent (before taking any photo)
Tell each classmate (verbally or by WhatsApp message, and write down here who said yes):
- The photos are only for this ML project.
- **Only the hand will be in the frame, not the face.** The photos will not go to GitHub, the report or the slides.
- Their photos will be deleted whenever they ask.

| Classmate ID | Consent given (yes/no) | Date | Right/left handed | Note |
|---|---|---|---|---|
| c01 | | | | |
| c02 | | | | |
| c03 | | | | |
| c04 | | | | |
| c05 | | | | |
| c06 | | | | |

(Do **not** write real names here, only c01, c02... Keep a separate private note of the names.)

## 2. How many photos
- 5-6 classmates, **2 photos** of each letter (3 is better if possible).
- 36 letters x 2 photos = **72 photos per classmate** (about 10-15 minutes).
- About 360-430 photos in total.

## 3. How to take the photos (checklist)
- [ ] **Right hand**, one hand, the other hand out of the frame.
- [ ] Plain background (wall, curtain), few things behind.
- [ ] Light **from the front**, no window/light behind the hand.
- [ ] The whole hand and the wrist in the frame, the hand in the middle of the frame.
- [ ] Camera at the level of the hand, 30-50 cm away (the distance of a laptop webcam).
- [ ] The face is **not** in the frame.
- [ ] Look at each letter first in the dataset picture (`data/raw/<Letter>-Original/`) or the FESF dictionary (psl.org.pk) and then make the sign.
- [ ] 2 different photos of each letter (move the hand a little and make it again), not a copy of the same photo.
- [ ] Take blurry or dark photos again.
- [ ] If a classmate is left-handed, still take the photos with the right hand first; the left-hand test comes later (Phase 12, C2).

## 4. Where to put the photos and what to name them
Folder: `data/custom_test/<Letter>/` (the folders already exist, see the table below).
File name: `c<ID>_<Letter>_<number>.jpg`, for example `c01_Lam_1.jpg`, `c01_Lam_2.jpg`, `c03_1-Hay_1.jpg`.
- The extension must be lowercase `.jpg` (Windows treats `.jpg` and `.JPG` as the same, so the count would be doubled).
- If the phone gives `.jpeg` / `.heic`, convert it to `.jpg`.
- `data/custom_test/` is gitignored, so the photos do not go to GitHub. **Never use `git add -f` on it.**

## 5. List of letters (folder name, Urdu, FESF name)
| # | Folder | Urdu | FESF name | c01 complete | Note |
|---|---|---|---|---|---|
| 1 | Alif | ا | Alif | | |
| 2 | Bay | ب | Bay | | |
| 3 | Pay | پ | Pay | | |
| 4 | Tay | ت | Tay | | |
| 5 | Taay | ٹ | Ttay | | |
| 6 | Say | ث | Say | | |
| 7 | Chay | چ | Chay | | |
| 8 | 1-Hay | ح | Hay | | |
| 9 | Khay | خ | Khay | | |
| 10 | Daal | د | Daal | | |
| 11 | Dal | ڈ | Ddaal | | |
| 12 | Zaal | ذ | Zaal | | |
| 13 | Ray | ر | Ray | | |
| 14 | Zay | ز | Zay | | |
| 15 | Zaey | ژ | Zzay | | |
| 16 | Seen | س | Seen | | |
| 17 | Sheen | ش | Sheen | | |
| 18 | Suad | ص | Suad | | |
| 19 | Zuad | ض | Zuad | | |
| 20 | Tuey | ط | Toay | | |
| 21 | Zuey | ظ | Zoay | | |
| 22 | Ain | ع | Aain | | |
| 23 | Ghain | غ | Ghain | | |
| 24 | Fay | ف | Fay | | |
| 25 | Kaf | ق | Qaaf | | |
| 26 | Kiaf | ک | Kaaf | | |
| 27 | Gaaf | گ | Gaaf | | |
| 28 | Lam | ل | Laam | | |
| 29 | Meem | م | Meem | | |
| 30 | Nuun | ن | Noon | | |
| 31 | Nuungh | ں | Noon Ghunna | | |
| 32 | Wao | و | Wow | | |
| 33 | Dochahay | ھ | Do Chashmi Hay | | |
| 34 | Hamza | ء | Hamza | | |
| 35 | Cyeh | ی | Choti Yay | | |
| 36 | Byeh | ے | Bari Yay | | |

## 6. When the photos are ready
When the photos are ready: check every letter, the count per classmate and the name format, then extract the landmarks and run the model (Steps 112-114).
