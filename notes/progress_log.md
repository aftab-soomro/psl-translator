# Progress Log

Chronological record: for each step, what was done, why, how, the files, the numbers, and a plain "explain it simply" paragraph.
Numbers always come from `results/*.csv`. Old numbers are never deleted.

> **Phase 0-9 backfill note (2026-11-01):** The Phase 0-9 entries below were written later from the git history
> (`git log`) and `results/*.csv`, not at the time the work was done.
> Aftab and Meraj must read the "Explain it simply" paragraphs and correct them in their own words.

---

## Phase 0-1: Setup and dataset (2026-09-26 / 27)
- **What:** .venv, requirements, GitHub repo, `ai_use_log.md`; downloaded and studied the UAlpha40 dataset (Mendeley, CC BY 4.0).
- **Decision:** only the 36 `-Original` folders; the augmented folders and the 4 dynamic folders (2-Hay, Alifmad, Aray, Jeem) are left out. The signer ID is in the filename (sXXXX).
- **Numbers:** 2,896 original images (paper: 2,897), 82 signers.
- **Commits:** 769e8ae, 00b4ebd, b7ab6a0, 703cd70, 4107ebb, 05bad52.
- **Simply:** Besides the real photos, the dataset also had copies of them (augmented). If we kept the copies, the same picture would show up again in the test set, so we used only the real photos.

## Phase 2: Landmarks (2026-09-27)
- **What:** `src/landmarks.py` (MediaPipe Tasks API, 63 numbers or None), `src/features.py` (wrist as origin, divide by size).
- **Numbers:** 2,727 succeeded, 169 failed (5.8%). Tay: 43/81 failed (46.9%), only 43 usable.
- **Reason (Tay):** closed fist, fingers overlap, MediaPipe cannot separate the joints.
- **Commits:** 3871da7, 69ec89b, dd0b5c4.
- **Simply:** Instead of learning straight from the photo, we took the 21 points of the hand. We measured the points from the wrist, so a small hand and a big hand look the same.

## Phase 3: EDA (2026-09-27)
- 4 plots `reports/figures/01-04`: class distribution, average hand shapes, overlay of similar letters, PCA.
- **Commit:** b2dd7ee.

## Phase 4: Split and leakage audit (2026-09-27)
- **What:** signer-wise split, seed 42, 20% of the signers go to the test set.
- **Numbers:** train 2,165 (66 signers), test 562 (16 signers). Overlap 0. Tay 31/12.
- **Commit:** 6a5055c. Details: `notes/leakage_audit.md`.
- **Simply:** All photos of one person go either to train or to test. Otherwise the model would remember the person's hand, not the sign.

## Phase 5-6: Baseline, models, tuning (2026-09-27)
- All models are `Pipeline(StandardScaler + model)`, 5-fold CV on the training data only.
- **CV (default -> tuned):** Baseline 3.05; KNN 82.49 -> 82.77 (k=3); SVM 86.93 -> 92.19 (C=100, gamma=0.01); RF 88.50 -> 88.68 (depth 20, 300 trees); MLP 91.36 -> 92.01 ((100,50), lr 0.001).
- **Commits:** d72830c, d790d5c. Files: `cv_results.csv`, `tuning_results.csv`.
- **Simply:** The scaler is inside the pipeline so it learns only from the training part. Tuning helped the SVM the most.

## Phase 7: Final test (2026-09-27)
- The test set was opened only here. Test: SVM 95.02% acc / 94.82% macro F1, MLP 93.77, RF 91.46, KNN 88.26, Baseline 2.85.
- **Best:** SVM (RBF). Files: `final_results.csv`, `confusion_matrix.csv`, `per_class_report.csv`, figure 05.
- Biggest confusion: Tuey<->Daal (9+3). Weakest F1: Tuey 0.538, Daal 0.684, Ain 0.828, Ghain 0.857, Tay 0.857.
- **Commits:** 6199ede, 7984d93.
- **Simply:** The test set is the exam you take only once. We chose the model with CV and used the test set only to see the result.

## Phase 8: Error analysis (2026-09-27)
- 8 errors, 2 patterns: (1) closed-fist occlusion (Alif, Nuun, Taay), (2) similar shapes (Ain/Ghain, Daal/Tuey, Ray/Zaal). Nuungh = angle.
- **Commit:** e2715c0. Figure 06.

## Phase 9: Save the model and predict() (2026-09-27, core freeze tag 2026-10-25)
- `models/best_model.joblib`, `class_names.joblib`, `src/predict.py`. Notebook 08 checked 10 test images (roadmap: 10/10).
- **Commits:** 52f9875, 6f7c190 (tag `core-v1`; `best_model_core_v1.joblib` backup, same hash).
- **Note (2026-11-01):** In 6f7c190 only 50 lines (model-save cells) changed in `06_evaluation.ipynb`; the test numbers are the same.

---

## 2026-11-01 | Phase 10 | Step 116 (code) B4: top-3 + low-confidence warning
- **What:** `app.py` shows the top-3 guesses and a warning when confidence is below 40%. It uses `predict_proba`; the model did not change.
- **Commit:** 22bfc00 ("tests pending").
- **Simply:** The model gives not just one answer but its top 3. When it is not sure, it says "try again".

## 2026-11-01 | Phase 10 | Step 109 (core freeze) complete
- **What / why:**
  - `tempfile.TemporaryDirectory` (the photo goes into a temporary folder that deletes itself, so no `temp_snapshot.jpg` is created in the project folder).
  - `temp_snapshot.jpg` in `.gitignore` (so a face does not go into the repo).
  - `use_container_width=True` -> `width="stretch"` (Streamlit deprecated the old way).
  - `notes/01_dataset_overview.ipynb` -> `notebooks/` (it was in the wrong folder, moved with `git mv`).
- **Test (Streamlit AppTest, a fake camera image was sent):**
  - Empty white image: warning "No hand found...", 0 exceptions (Step 106 verified).
  - Lam: Lam 85.9%. Alif: Alif 87.5%. Tuey: **Daal 46.9% (wrong)**, no warning because 46.9% is above the 40% threshold.
  - No `temp_snapshot.jpg` was created in the root folder.
- **Result / lesson:** Tuey->Daal is the same known confusion as in Phase 7. The low-confidence warning does not catch every error; this is a limitation for the report.
- **Simply:** We cleaned up the app: the photo deletes itself and old warnings were removed. We also saw that the app can sometimes give a wrong answer with confidence.
- **Windows note:** `streamlit.exe` is blocked by an Application Control policy on Aftab's laptop. The way to run it: `python -m streamlit run app/app.py`.

## 2026-11-01 | Phase 10 | Step 107 (manual test of 10 letters) and Step 108 (Meraj)
- **Step 107:** Aftab said verbally that 10 letters were tested with the webcam (suggested list: Alif, Lam, Meem, Wao, Kaf, Tuey, Daal, Ain, Tay, Nuun). **The per-letter results are not recorded here.** If the report mentions app testing, this table must be filled in later.
- **Open palm test:** the result was not recorded. This is the one of the three Step 116 tests that was still open.
- **Step 108:** The test on Meraj's laptop will be done later. Status: open.
- **Push:** Phase 10 is not complete (Step 108 is open), so the commits up to `ccaac18` have not been pushed yet.

## 2026-11-01 | Phase 11 | Steps 110-111 preparation (classmates' photos)
- **What:** wrote `notes/custom_test_protocol.md` (consent, how many photos, photo checklist, file naming `c01_Lam_1.jpg`, table of the 36 letters). Created the 36 folders `data/custom_test/<Letter>/` (gitignored, verified with `git check-ignore`).
- **Why:** The signers in the dataset test set were also from the dataset. The score on new people, a new camera and a new background is closer to real life.
- **Decision:** Photos show only the hand, not the face (privacy). Classmates are written only by ID (c01...), not by name. We never train, tune or select with custom_test.
- **Simply:** This is the "exam at home". The model's score will differ between the class exam (dataset test) and the home exam with completely new people, and that difference tells the most honest story.
- **Status:** Taking the photos is still open (Aftab/Meraj's job).
## Step 116 (B4): Top-3 + low-confidence warning, 2026-11-01
- What: app/app.py shows the top-3 predictions and a warning when confidence is below 40% (the code was already there). Three manual tests were done.
- Results: Alif 87.5% correct; Tuey -> Daal 46.9% (wrong, no warning); open palm -> Bay 77.1% (no warning). Top-3: Bay 77.1, Say 7.6, Fay 4.2.
- Result: the warning catches only "the model is confused". The model picks only from the 36 letters, so it can give a confident answer for any hand.
- Explain it simply: the model has no "I don't know" option. It always picks one of the 36 letters, even if the hand is not any letter.
- Limitation (for the report): on out-of-distribution input the confidence is not reliable.

## Step 117 (C2): Left-hand support, 2026-11-01
- What: notebooks/09_left_hand.ipynb. A left hand is the mirror of a right hand; in the normalized landmarks x is flipped (x *= -1).
- CV (5-fold, SVM C=100 gamma=0.01, train.csv): baseline right 0.9187, left (no fix) 0.7760, (b) flip 0.9187, (a) mirror-train right 0.9316, left 0.9312. (a) is better than the baseline in every fold (+0.5 to +2.3 points).
- Decision (by CV): (a) mirror-train. Model: models/best_model_v2_mirror.joblib. The core model was not touched.
- Test (results/advanced_left_hand_test.csv): core right 0.9502, core left 0.7972, (a) right 0.9377, (a) left 0.9377.
- Explain it simply: a left hand is a mirror image of a right hand. We showed the model both kinds of hands, so the accuracy on the left hand went from 80% to 94%, and on the right hand it dropped a little (95% to 93.8%).

- App test: predict.py now uses best_model_v2_mirror.joblib (MODEL_FILE constant, the core v1 backup is kept safe). Notebook 08: 10/10 correct (Tuey correct but at 38.69%, below the threshold). Webcam: Lam recognised with both hands, 75.5% and 83.5%.

## Step 118 (B3): Angles and fingertip distances, 2026-11-01
- What: add_engineered_features() in src/features.py (30 new features: 5 tip-to-wrist, 10 tip-to-tip, 15 joint angles). Notebook 10_features_angles.ipynb.
- CV (StratifiedKFold, 5-fold): raw63+mirror 0.9316; eng93+mirror (gamma 0.01) 0.9483, gamma 0.0068: 0.9460. Better in every fold.
- Test (results/advanced_features_test.csv): core v1 0.9502/left 0.7972; v2 0.9377; v3 (chosen) 0.9324; v3b 0.9342. The ranking differs between CV and test.
- We suspected the CV was leaky; signer-wise CV (StratifiedGroupKFold, results/advanced_grouped_cv.csv) gave the same ranking (A 0.9252, B 0.9311, C 0.9499, D 0.9458 right hand). The suspicion was wrong.
- Decision (by CV): v3 = models/best_model_v3_eng.joblib. Core and v2 are kept safe.
- Explain it simply: instead of giving the model only the points, we also gave it simple facts like "how much is the finger bent" and "is the thumb touching the index finger". It was 1.7 points better in CV and slightly lower on the test set; the difference is small.
- App test: predict.py now uses v3. Notebook 08: 9/10 correct (Tuey -> Daal 44.54%, no warning; Suad correct but at 66.8%).

## Step 119 (A1): Word Builder page, 2026-11-01
- What: app/pages/1_Word_Builder.py (new page) and src/letters.py (a copy of LETTER_MAP; it is still in app.py too).
- How: a list of letters in st.session_state; webcam photo -> predict() -> choose the right one from the top 3 -> "Add letter". Backspace, Space and Clear buttons. After each letter the camera key changes so the camera is empty.
- Test: manual, added 3 letters, backspace and clear worked fine.
- Explain it simply: Streamlit runs the page again on every click, so a normal variable is forgotten. session_state is a memory that stays. Choosing from the top 3 can also fix a model mistake.
- Cleanup left: LETTER_MAP is now in two places (app.py and src/letters.py); in Phase 14 make app.py import it.

## Step 120 (A2): Speaking the word in Urdu (gTTS), 2026-11-01
- What: a "Speak word" button on the Word Builder page. gTTS (lang="ur") makes an mp3 and st.audio plays it.
- Fallback: the mp3 of every word is cached in app/audio_cache/<md5>.mp3. A cached word works without internet. A new word gives a warning when there is no internet.
- Test: the voice was in Urdu, the cache file was created (~7 KB each).
- Explain it simply: gTTS sends the word to Google and gets the voice back (needs internet). Once a voice is made we save it, so the second time it works offline too.
- .gitignore: app/audio_cache/*.mp3. Before the demo, commit only the files of the demo words with git add -f.

## Step 121 (B2): Clustering of 36 average hand shapes, 2026-11-01
- What: notebooks/11_clustering.ipynb. From train.csv, the average shape of each letter (36 x 63). Hierarchical (Ward) dendrogram, pairwise distance rank, k-means (elbow, silhouette).
- Files: reports/figures/07_dendrogram.png, 08_kmeans_elbow_silhouette.png; results/advanced_cluster_pairs.csv, advanced_kmeans_k.csv.
- Pair ranks (out of 630 pairs, 1 = closest): Tuey-Daal 1, Zuad-Zuey 6, Ain-Ghain 26, Ray-Zaal 100.
- k-means: k=2 silhouette 0.635 (only Kaf and Pay separate); k=3..12 silhouette 0.305-0.382 (k=5: 0.382). The elbow has no clear bend. This means the shapes do not form clearly separate groups.
- k=5: Tuey/Daal in one cluster, Ain/Ghain in their own cluster, Zuad/Zuey in one cluster, Ray and Zaal in separate clusters.
- Explain it simply: in clustering we do not give the letter names; we ask it to make groups from the shapes only. The letters the model confuses were also close in shape, so the reason for the confusion is the real similarity of the shapes.
- Limit: Meem-Nuun (rank 2) are also very close, but the model did not confuse them much. Being close is necessary, not enough.

## Step 122 (B1): Random Forest feature importance on the hand skeleton, 2026-11-01
- What: notebooks/12_feature_importance.ipynb. RF (300 trees, depth 20, seed 42) on the 63 features of train.csv. The importance of a landmark = the sum of x+y+z. Shown in color on the average hand skeleton.
- Files: reports/figures/09_feature_importance_skeleton.png, results/advanced_feature_importance.csv.
- Top: middle_tip 0.0896, thumb_tip 0.0873, pinky_tip 0.0654, ring_tip 0.0640, middle_dip 0.0626, index_tip 0.0622. Wrist 0 (it is the origin after normalization).
- Explain it simply: to tell the letters apart, the RF trees used the finger tips (and the joints near them) the most. The points near the bottom of the palm were used less.
- Limit: RF importance is split between similar features, and the tips move more, so read it as "what the RF used", not as "the truth".

## Step 124: Augmented-image experiment, 2026-11-01
- What: notebooks/13_augmented_experiment.ipynb. From the zip (without extracting it), 13,795 augmented images that belong only to train signers; landmarks extracted -> 11,821 succeeded (data/processed/aug_landmarks.csv, gitignored). The 5,516 copies of the test signers (16) were not used at all.
- Checks: train/test signer overlap 0; all augmented signers are among the train signers; no copies of the validation-fold signers are in training (assert).
- Signer-wise CV (5-fold, eng93, SVM C=100 gamma=0.01, no mirror): D originals only 0.9458; E originals+augmented 0.9468 (+0.1 point, mixed across folds).
- Test (results/advanced_augmented_test.csv): D 0.9413; E 0.9520.
- Decision (by CV): no reason to adopt augmented data (6.5x data, almost the same score). We kept model v3.
- Only 166 augmented copies of Tay had a hand found (the others 278-420), the same closed-fist problem.
- Explain it simply: an augmented copy changes the light/zoom, but landmarks and normalization already remove that difference, so the copy gives the same numbers. New signers (new hands) would help, copies do not.

### Step 124b: v4 = eng93 + mirror + augmented, 2026-11-01
- We looked at the augmented copies (next to the original): slight difference in light/color, slight shift/zoom/rotation, no flip, nothing dark. The pictures look like clean studio shots.
- Signer-wise CV (results/advanced_v4_cv.csv): v3 0.9499; v4 0.9509 (+0.10, mixed across folds).
- Test (results/advanced_v4_test.csv): v3 0.9324; v4 0.9448 (+1.24). Left/right are equal for both (mirror).
- Decision: CV is a tie, and we do not break the tie with the test set. v3 is in the app. v4 was saved as models/best_model_v4_aug.joblib (5.5 MB); the v3 vs v4 decision will be made with Aftab's own photos (low/high light) (Phase 11).
- Explain it simply: the light difference in the augmented copies is slight, so the benefit "better in the dark" does not show up in CV/test. We will measure it with our own webcam photos.

## Step 125: Summary tables and re-check of the app, 2026-11-01
- What: notebooks/14_summary_tables.ipynb. The numbers of all variants come from results/*.csv: results/summary_test_variants.csv and results/summary_signer_cv_variants.csv.
- Test (right/left): core v1 0.9502/0.7972; v2 0.9377/0.9377; v3 0.9324/0.9324; v3b 0.9342/0.9342; v4 0.9448/0.9448.
- Signer-wise CV (right/left): A 0.9252/0.7725; B 0.9311/0.9311; C (v3) 0.9499/0.9499; D 0.9458/0.8871; F (v4) 0.9509/0.9509.
- Result: mirroring is needed for the left hand (a -15 point gap in CV without mirror), the new features give ~+1.9 points in CV, augmented gives ~+0.1 in CV.
- App check (v3): main page + top-3, left hand, Word Builder + voice, open palm, all fine.
- Phase 12 is complete, except Step 123 (quiz, dropped) and Step 126 (videos, stretch).
- Pending decision: v3 vs v4 (own photos in low/high light, Phase 11).

## Step 126 (a-c): From the videos of the 4 dynamic letters, 2026-11-01
- Videos: 393 mp4 (2-Hay 87, Alifmad 105, Aray 102, Jeem 99), 1920x1080, 30 fps, 43-64 frames. The file s316.mp4 had a 3-digit name (taken as signer 0316).
- Split: test = the same 16 test signers (61 videos); everyone else (train and "none" signers) is train (332 videos). Group = signer_id.
- Landmarks: every second frame (step=2), at 640 px width, with the same extract_landmarks(). 11,439 frames, a hand was found in 96.5%. data/processed/video_landmarks.csv (gitignored).
- Features: missing frames filled by interpolation, each frame normalized + wrist movement (2 numbers), the video resampled to T time steps. T=8: 520 features.
- Signer-wise CV (5-fold, train videos): Baseline 0.271; KNN 0.967; SVM 0.976; RF 0.991; MLP 0.985 (T=8). T=16 is slightly lower for all. Choice: RF, T=8.
- Test (61 videos): RF T=8 0.9672 (2 errors, both 2-Hay -> Aray); SVM/MLP 0.9836; KNN 0.9672; baseline 0.2459. Files: results/advanced_video_cv.csv, advanced_video_test.csv, models/dynamic_rf_T8.joblib.
- Explain it simply: static letters are recognised from one picture; in dynamic letters the hand moves. We made the landmarks of the video frames into one long list (fixed length) and gave it to the model.
- Limit: only 4 letters, and the movement of each is very different, so it is an easy problem. The videos are 1080p and studio-like.

Phase 12 live app: Steps 1-2 done (FPS ~27), Step 3 started.

Phase 12 live app: Steps 1-3 code complete (details in notes/live_app.md). Live testing is still open.

## Step 127: Tuey/Daal pair analysis, 2026-11-09
- What: notebooks/16_tuey_daal_pair.ipynb. In the live app Tuey and Daal got mixed up; we checked whether this is a problem of the model or of the data.
- Signer-wise CV (5-fold, train.csv, 2,165 rows, 36 classes): 0.9524. Total errors 103.
- The biggest error is Tuey <-> Daal: 46 / 103 (45%). The accuracy of this pair is only 0.617.
- A specialist model (only Tuey vs Daal) is also close to chance. Result: the landmarks do not show a difference between the two; this is a limit of the data, not of the model.
- If we treat the two as one, CV is ~97.4%.
- Next pairs: Ain <-> Ghain 8, Zuad <-> Zuey 7, Seen <-> Taay 5, Alif <-> Suad 4, Meem <-> Taay 3. Their error is below 30%, so we did not make a group.
- Taay is weak because the hand is not detected (closed fist), not because of confusion; CV recall for Taay is 0.92.
- Files: results/advanced_confusion_pairs_cv.csv, results/advanced_per_letter_recall_cv.csv.
- Decision: in the live app Tuey/Daal became one group; when the hold is complete the user chooses 1 or 2 (the buttons "ٹ or د?" in the app).
- Explain it simply: the two signs look so alike in landmarks that no model can separate them; so the app asks.
- Limit: write in the report that Tuey/Daal is not automatic, the user chooses.

## Step 128: Live app: smoothing and dynamic letters (Method B), 2026-11-09
- What: src/live.py, app/live_app.py (details in notes/live_app.md).
- Static: average of the probabilities over 0.4 second, hold 1 second, confidence >= 60%, Tuey/Daal group.
- Dynamic: button -> 3-2-1 -> 3 seconds of recording -> dynamic_rf_T8 (T=8, the same features as in training). The live camera is 4:3 while the training videos are 16:9, so we use the y aspect_fix.
- The live test log is written automatically to results/live_dynamic_log.csv (true, pred, conf, frames, frames with a hand).
- Method A (self-detection from motion) not for now; first look at the live accuracy of B.
- Live accuracy (own webcam, one signer): 4 dynamic letters 56/56 correct; 70% accepted at confidence >= 0.60. On 'no gesture' 4/10 false letters (the model has no none class). Details in notes/live_app.md and results/live_dynamic_threshold.csv.
- Explain it simply: a static letter needs you to hold still for a moment; for a dynamic letter you press a button so the app knows the movement is about to start.
- Limit: the dynamic model has no "no gesture" class, so a wrong movement can also turn into a letter; if confidence is below 60% no answer is given.
