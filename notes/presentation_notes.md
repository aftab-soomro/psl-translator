# Presentation notes (running list for the PPT and the speaking script)

Keep adding to this file whenever something is built or decided. At the end, the whole project zip is given to Claude, who reads all the code and these notes and builds the slides and the script from them.
Numbers must always come from `results/*.csv` or the logs. Items marked (fill) are not known yet.

## 1. The project in one sentence
A live webcam app that recognises the Pakistan Sign Language Urdu alphabet (36 static letters + 4 dynamic letters), builds a word on the screen and can read it aloud. Classical ML only (no deep learning), Python 3.12.

## 2. Story of the work (one slide each)
1. **Problem and data.** UAlpha40 dataset (CC BY 4.0), 36 static letters as photos, 4 dynamic letters as videos. Many signers, studio recordings.
2. **Hand landmarks.** MediaPipe HandLandmarker gives 21 points (63 numbers). About 5.8% of dataset hands are not detected (for example Tay: 46.9% of its images failed because the fist hides the fingers).
3. **Normalisation.** Wrist = origin, scale by palm size, so hand size and position do not matter.
4. **No leakage.** Signer-wise split: the same person is never in train and test. A random split would let the model recognise the person, not the sign. The test set was kept sealed and was not used to choose the model.
5. **Models.** KNN, SVM, Random Forest, MLP compared with signer-wise 5-fold cross-validation. Chosen: 93 engineered features (angles and distances) + StandardScaler + SVC (C=100, gamma=0.01), trained with mirror augmentation (left hand).
6. **Result.** About 95.2% signer-wise cross-validation accuracy on 36 classes (2,165 training rows). Test accuracy: (fill from results/final_results.csv).
7. **The hard pair: Tuey / Daal.** 46 of 103 CV errors (45%); pair accuracy only 0.617; even a specialist model is at chance. The landmarks do not contain the difference. Decision: the app asks the user (buttons 1 / 2). If the two were one class, CV would be about 97.4%.
8. **Dynamic letters.** Jeem, Aray, Alifmad, 2-Hay need movement. Random Forest on 8 resampled time steps of the landmark sequence: 99.1% CV, 96.7% on held-out test videos.
9. **Live test on my own webcam.** 56 of 56 dynamic gestures correct. At confidence >= 0.60: 39 of 56 accepted, all correct. "No gesture" tests: 4 of 10 added a false letter, so the confidence threshold 0.60 is used.
10. **Live app.** One screen: video with the 21 points, word box, live recognition with top-3 bars, Harf guide (photo of every letter), Settings, Stats.
11. **Auto mode (no button).** Motion-triggered gesture detection (details below).
12. **Limits and future work.** See section 5.

## 3. How the live app works (for the demo)
- Static letter: hold the sign about 1 second; the letter is added. Probabilities are averaged over 0.4 s so the label does not flicker. Same letter again needs a new letter or the hand removed and brought back.
- Tuey / Daal: two numbered buttons appear (or keys 1 / 2).
- Dynamic letter: press Record, 3-2-1, do the gesture for 3 seconds. Or switch on Auto mode and just do the gesture.
- Stats popup: live accuracy from the logs (`results/live_dynamic_log.csv`).
- Demo idea: spell a short name with static letters, show a Tuey / Daal question, then one dynamic letter (Jeem) with the button, then with Auto mode.

## 4. Auto mode: the engineering story (good for a "what went wrong and how I fixed it" slide)
- Why: a button is awkward for a signer.
- How: hand speed = average distance moved by the 21 points over ~0.2 s, divided by palm length, per second. Fast for 0.25 s starts a segment (with ~0.5 s of earlier frames); 0.4 s of stillness ends it; the still tail is cut; the same dynamic model classifies it.
- I recorded my screen and checked it frame by frame. Problems found:
  1. Start threshold 0.40 was too low: resting hand 0.0-0.4, slightly moving hand 1.3-1.6, so it triggered all the time.
  2. After a correct Jeem the end pose of the hand was read by the static model as Wao and Alif and added to the word. Cause: motion blur made the hand vanish for a moment, which unlocked the pose.
  3. Tiny triggers paused the static letters and flickered the red REC chip.
  4. A gesture that lasted the full 3.5 s got a low confidence (Jeem 31%) and was rejected (a safe failure).
- Fixes (2026-11-20): static paused only after movement lasted 0.9 s; the locked end pose is unlocked only after the hand is away for 1 s; new defaults (Start 0.80, End 0.35, Min duration 0.9, Min motion 1.3); maximum segment 3.0 s; result card clears after 8 s; warning when End >= Start.
- Tests: synthetic hand movements (old code added the wrong letter, new code does not; a tremor is ignored); the real app with a fake camera. Live accuracy of Auto mode after the fixes: (fill, not measured yet).
- Lesson to tell: look at the real behaviour (screen recording), not only at the numbers; separate "detect the gesture" from "classify the gesture".

## 5. Limits and future work (be honest)
- All training data is from a studio; other lighting, backgrounds and signers may lower accuracy. No external test set; the test set is small (about +/- 1.8 points).
- Tuey / Daal cannot be separated from landmarks alone.
- No "no letter" class: a non-sign hand shape can still be predicted as a letter (confidence threshold and 1 s hold reduce this).
- Dynamic letters: right hand, one signer for the live test.
- Only letters, not words or sentences. Future: more signers, a "no gesture" class for dynamic letters, depth or video-based features for Tuey / Daal.

## 6. Questions the audience may ask (short answers are in notes/viva_qa.md)
- Why classical ML and not a CNN? Why landmarks and not images? Why signer-wise split? Why 0.60 threshold? Why does Tuey / Daal fail? How does Auto mode know a gesture started? What does the app do with no hand?

## 7. Log of things added after the submission (newest last)
- 2026-11-20: Auto mode fixes (branch auto-fixes). Tests passed on synthetic data and in the real app with a fake camera.