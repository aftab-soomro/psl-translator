# Leakage Audit

Every decision we made to prevent data leakage, and the reason for it.

## 1. Augmented images were not used at all

Besides the 2,897 original images, the UAlpha40 dataset also had 19,383 "augmented"
copies (made with noise, rotation and brightness changes). We used only the
2,896 original images (1 image mismatch, the difference is negligible).

Reason: augmented images are slightly changed copies of the original images. If
an original went to train and its copy went to test, the model would "remember" the
same hand it saw in train, and the test accuracy would be false (inflated).

## 2. Signer-wise train/test split (not a random split)

In the dataset, the filename of every image contains the signer ID (sXXXX) - 82 different
people. Instead of a row-level random split we made a SIGNER-level split: 65-66
signers in train, 16-17 signers in test, no signer in both.

Reason: if one sign of a person were in train and another sign in test, the
model would recognise the shape of that person's hand (size, finger length),
not the sign. The signer-wise split makes sure the model sees the hands of
COMPLETELY NEW people in the test set, which is like real-world use.

Verified (Step 55-56):
- All 36 classes are present in both train and test
- No file is in both sets (overlap = 0)
- No signer is in both sets (signer overlap = 0)
- Random seed 42 is fixed, so the split can be reproduced

## 3. Feature-level leakage check

Our 63 features (21 landmarks x x,y,z) come from inside a single image only --
MediaPipe processes each image SEPARATELY, unaware of the others. There is no feature that:
- uses information from another image
- is not available at prediction time (like a hint of the "true answer")
- uses future data (the dataset is not a time series, so this does not apply)

## 4. Preprocessing is fitted on the training data only

When we use scaling (StandardScaler) in Phases 5-6, it will always be inside a
scikit-learn Pipeline and fitted only on the TRAINING fold -- never on the whole
dataset (train+test). This way no information of the test set (like its
average/spread) can "leak" at training time.

## 5. Test set discipline

test.csv was created in Phase 4 (Step 54), and it will not be opened until
Phase 7 (Step 81) -- not for EDA, not for model choice, not for
hyperparameter tuning. All decisions will be made with cross-validation on
train.csv only.

## Extras: re-use of test.csv (from Step 117)
The numbers of every variant were reported on test.csv (core v1 and v2 mirror), but the variant was chosen only by 5-fold CV (train.csv). No decision was changed after looking at the test numbers. Step 117: CV chose (a) mirror-train; on the test set the right hand was 0.9377 (core 0.9502), and we still kept (a).

## Step 118: signer-wise CV was run after seeing the test numbers
The test score of v3 did not match the CV, so we suspected that the StratifiedKFold CV was too optimistic because of signer leakage. So StratifiedGroupKFold(signer_id) was run on train.csv (this decision was made after seeing the test numbers). Result: the signer-wise CV gave the same ranking, so the suspicion was wrong. The variants reported on test.csv so far: core v1, v2, v3, v3b. The choice was always made by CV.

## Step 124: augmented images
Augmented copies were taken only for the signers of train.csv (matched with the signer ID sXXXX from the filename). The 5,516 copies of the test signers were excluded. In CV, inside every fold only the copies of that fold's train signers were used; an assert checks that no copy of a validation signer is present. Validation is on original images only. Both D and E were reported on test.csv, and the choice was made by CV.

Step 124b: the CV of v4 (augmented + mirror) is signer-wise, with the same asserts. Both v3 and v4 were reported on test.csv (v3 0.9324, v4 0.9448), but the choice was not made by the test set. CV is a tie, and the decision was left to separate custom photos (low/high light).

## Step 126: videos
The split is signer-wise: test = the same 16 test signers that were in the static letters, everyone else is train. In CV groups=signer_id. All variants (T=8/16, 4 models) were reported on the test set, and the choice was made by CV (RF, T=8).
