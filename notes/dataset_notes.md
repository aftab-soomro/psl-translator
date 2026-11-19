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

**Result:** PCA shows that most classes overlap a lot in 2D,
which is expected because all the signs are made with the same hand
structure. One small cluster (on the right) is clearly separate from all the others,
which confirms that at least some letters are very different in shape.
In 2 dimensions only X% of the variance is captured,
so this is only a glimpse — the real model works in 63 dimensions,
where the difference will be even clearer.

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

## Top confusing pairs (Step 85)

1. Tuey <-> Daal: 12 total confusions (biggest error source)
2. Ain <-> Ghain: 4 total confusions
3. Zuad <-> Zuey: 2 total confusions

Note: Ain/Ghain and Zuad/Zuey are letter pairs that are also phonetically
similar/identical in spoken Urdu (Ain-Ghain are both guttural sounds; Zuad-Zuey
are both pronounced as "z" in Urdu despite being distinct in Arabic). This
suggests their PSL hand-signs may have been designed with visually related
shapes, consistent with the model's confusion pattern.

Our earlier prediction (Kaf/Tay, based on similar "closed fist" appearance in
Step 46) did NOT turn out to be a major confusion pair -- a reminder that
visual intuition from a few sample images doesn't always match what the
trained model actually struggles with.

## Weakest 5 classes by F1 (Step 86)

| Letter | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Tuey | 0.700 | 0.438 | 0.538 | 16 |
| Daal | 0.591 | 0.813 | 0.684 | 16 |
| Ain | 0.800 | 0.857 | 0.828 | 14 |
| Ghain | 0.857 | 0.857 | 0.857 | 14 |
| Tay | 1.000 | 0.750 | 0.857 | 12 |

Consistent with Step 85: Tuey and Daal are confused with EACH OTHER (Tuey's
low recall = Daal's low precision), confirming this is a genuine, mutual
confusion pattern, not a coincidence.

Interesting: despite having the fewest training images (43), Tay has PERFECT
precision (100%) -- whenever the model predicts Tay, it's always correct. Its
recall (75%) is lower, meaning some true Tay signs get missed and called
something else, but the shape itself is distinctive enough not to be a major
confusion source once detected.

## Error analysis (Step 92-93)

8 diverse misclassifications reviewed. Two clear failure patterns:

1. **Closed-fist occlusion** (3/8 errors: Alif, Nuun, Taay): when the hand
   forms a closed fist with knuckles facing the camera, overlapping fingers
   make individual joints hard to distinguish, leading to misclassification.
   This is the SAME root cause identified in Step 40 for Tay's high landmark-
   extraction failure rate -- here it shows up as prediction errors instead of
   extraction failures.

2. **Genuinely similar hand shapes** (4/8 errors: Ain/Ghain, Daal/Tuey,
   Ray/Zaal): these letter pairs have naturally similar finger configurations
   in PSL itself, independent of any data quality issue.

One case (Nuungh) involved an unusual hand angle/rotation.

Conclusion: model errors are not random -- they cluster around (a) fist-shaped
signs with finger occlusion, and (b) letter pairs with inherently similar PSL
hand-shapes.

## Advanced extras

### B4: Top-3 predictions + low-confidence warning (Step 116)
- App shows top-3 letters with probabilities, using SVM predict_proba.
- Warning shown below 40% confidence. Threshold chosen from observation: some
  correct predictions had 32-42% confidence (e.g. Tuey), so low confidence is
  a hint to retry, not proof of a wrong answer.
- No retraining needed; core model unchanged (best_model.joblib).