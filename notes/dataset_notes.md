# Dataset Notes

Source: UAlpha40 (Mendeley, DOI: 10.17632/3pvnnckxyb.3)
Paper: Sajid, Baloch & Javaid, Data in Brief, 2025, DOI: 10.1016/j.dib.2025.111342
Authors: Sameena Javaid, Shahood Sajid, Yusra Khan Baloch (Dept. of Computer Science, Bahria University, Karachi)

- 22,280 images total: 2,896 original + 19,383 augmented (36 static classes)
- 393 videos for 4 dynamic classes (2-Hay, Alifmad, Aray, Jeem) — out of project scope
- Collected with 3 DSLR cameras + LED lighting, volunteers aged 20-45
- Signs follow the FESF standard chart for Urdu Sign Language
- No related published research on this dataset yet (first ML application)
- File naming: sXXXX = signer/volunteer ID (confirmed by our own inspection — same
  sXXXX appears with the same hand/sleeve across different letter folders)
- Some images use .jpg, some .JPG — case must be handled in code


## Dynamic letters (excluded, for future reference if ever needed)

2-Hay → ہ (Choti Hay), Jeem → ج (Jeem), Alifmad → آ (Alif Madda), Aray → ڑ (Rray)


## Decision (confirmed)

- Use only the 36 folders ending in "-Original". Do NOT use "-Augmented" folders.
- Exclude these 4 folders entirely (dynamic/video gestures, out of scope): 2-Hay, Alifmad, Aray, Jeem
- The 36 static classes (folder name without "-Original"):
  1-Hay, Ain, Alif, Bay, Byeh, Chay, Cyeh, Daal, Dal, Dochahay, Fay, Gaaf, Ghain,
  Hamza, Kaf, Khay, Kiaf, Lam, Meem, Nuun, Nuungh, Pay, Ray, Say, Seen, Sheen,
  Suad, Taay, Tay, Tuey, Wao, Zaal, Zaey, Zay, Zuad, Zuey
- Signer ID is embedded in the filename as "sXXXX" (e.g. s0251). This will be used
  for signer-wise train/test splitting (Phase 4) to avoid leakage.

  
## Verified counts (Step 22-23)

- Paper reports 2,897 original images across 36 static classes.
- Our own count (after removing augmented/dynamic folders): 2,896 images.
- Difference of 1 image is negligible and does not affect the project.
- Note: Windows treats .jpg and .JPG as the same extension (case-insensitive
  filesystem), so counting code must normalize extensions to avoid double-counting.

  
## Visual inspection (Step 24)

- Hand is clear, sharp and fully visible in all sampled images.
- Backgrounds vary a lot between signers (different clothing/rooms) — not a
  concern since we use hand landmarks, not raw pixels.
- "Kaf" sign has a difficult, folded-hand angle with overlapping fingers —
  may cause more landmark-detection failures in Phase 2, worth watching.

  
## Duplicate check (Step 25)

- Checked each class folder for a signer ID (sXXXX) appearing more than once.
- Result: 0 folders had a repeated signer. Each signer appears at most once
  per letter, so there is no within-class duplication to worry about.

  
## Label mapping source (Step 26)

Mapped using the official FESF PSL Dictionary (https://psl.org.pk/dictionary/8-alphabet-urdu),
matching each dataset folder name to FESF's own alphabet naming.


## Landmark extraction results (Step 36-39)

- 2,896 original images -> 2,727 successful landmark extractions, 169 failed (5.8%)
- "Tay" is a clear outlier: 43/81 images failed (46.9% fail rate), leaving only
  43 usable images -- far below the ~75-82 that other classes have.
- Next highest fail rates: Ain (16.5%), Ghain (16.2%), Dochahay (13.8%)
- All other classes have 0-10% fail rate, comfortably above 70 images
- This imbalance (Tay having ~half the data of other classes) is a known
  limitation and will be revisited in error analysis and the report.

  
## Tay failure analysis (Step 40)

Visually inspected 6 failed "Tay" images. All show a closed-fist hand shape
with fingers curled and overlapping, viewed knuckles-first toward the camera.
Lighting, background and framing are all fine -- the failure is specifically
because MediaPipe's landmark model struggles to resolve individual finger
joints when they are folded and occluding each other. This is a known
limitation of landmark-based (vs. pixel-based) hand recognition for fist-like
signs, and explains why "Tay" has a much higher failure rate (46.9%) than
other classes.

**Nateeja:** PCA se pata chalta hai k zyada tar classes 2D me kaafi overlap
karti hain, jo tawaqqo ke mutabiq hai kyun k saare signs ek hi haath ki
banawat se bante hain. Ek chhota cluster (dayen taraf) baqi sab se saaf
alag hai, jo darust karta hai k kam se kam kuch harf shape ke lihaaz se
kaafi mukhtalif hain. In 2 dimensions me sirf X% variance capture hoti hai,
is liye ye sirf ek jhalak hai — asal model 63 dimensions par kaam karega,
jahan farq aur bhi saaf hoga.

## Best model (Step 77)

SVM (RBF kernel, C=100, gamma=0.01) chosen as best model:
- Highest tuned CV accuracy: 92.19%
- Largest improvement from tuning: +5.26 percentage points
- Simpler than MLP (2 hyperparameters vs. multiple), lower overfitting risk
- MLP is a close second (92.01%) and will be mentioned as a strong alternative
  in the report

  ## Final test results (Step 81)

Test set: 562 images, 36 classes, from 16-17 signers never seen in training.
- Baseline: 2.85%
- KNN: 88.26%
- Random Forest: 91.46%
- MLP: 93.77%
- SVM (RBF): 95.02% <- best model

All models score HIGHER on the held-out test set than in cross-validation,
because final models were trained on 100% of train.csv vs ~80% per CV fold.
No signs of overfitting.

## Precision/Recall/F1 (Step 82)

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| SVM (RBF) | 95.02% | 95.24% | 94.85% | 94.82% |
| MLP | 93.77% | 93.96% | 93.58% | 93.61% |
| Random Forest | 91.46% | 91.65% | 91.28% | 91.29% |
| KNN | 88.26% | 88.59% | 88.11% | 88.10% |
| Baseline | 2.85% | 0.08% | 2.78% | 0.15% |

Ranking stays identical between accuracy and macro F1 -- confirms SVM's high
accuracy is NOT just from majority classes; it performs well even on
underrepresented classes like Tay.