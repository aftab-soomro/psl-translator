# Live single-page app (Phase 12, live)

## Why
- The old 3 pages (snapshot, Word Builder, dynamic video upload) needed a photo or a video. They are now in legacy/.
- Goal: one page, live camera, the 21 hand points shown live, letters added to the word by themselves.

## Step 1: live camera + 21 points (legacy/live_test.py)
- Library: streamlit-webrtc (frames of the browser camera reach a Python callback).
- MediaPipe HandLandmarker in VIDEO mode (detect_for_video), the timestamp always increasing.
- Result: it works, FPS ~27 on my laptop (no hand). With a hand and prediction, FPS ~11-12.

## Step 2: prediction + automatic word (src/live.py, app/live_app.py)
- Model: the one that MODEL_FILE in src/predict.py names (currently v3).
- Detection runs on the original (not mirrored) frame, because the training images were not mirrored either; the screen shows a mirror view for display only.
- Probability averaging: the average of the probabilities over the last 0.4 second, then the top letter is chosen (less flicker).
- A letter is added when the top letter stays the same for 1 second with >= 60% confidence (can be changed in Settings).
- The same letter is added again only when another letter has been added or the hand is removed and brought back.
- Against flicker: the candidate changes only if another letter appears for 3 frames in a row; if no hand is found for 8 frames it counts as "hand removed".
- Distances on screen: wrist->tip (5) and neighbouring tips (4), in units of palm length.
- No video is saved anywhere; an mp3 is made in app/audio_cache only on "Speak".
- Live test of the static letters: (still open)

## Tuey / Daal group (src/live.py, GROUPS)
- Reason: notebooks/16_tuey_daal_pair.ipynb. Tuey and Daal cannot be separated from landmarks (pair accuracy 0.617, even a specialist model is at chance).
- So when the model reaches the hold on one of these two, the letter is not added directly; it becomes "pending".
- Two buttons appear above the camera (1 = Tuey, 2 = Daal); you can also choose with keyboard 1 / 2. On choosing, that letter is added.
- Rule: only a pair with more than 30% errors becomes a group. For now only Tuey-Daal.

## Step 3: dynamic letters live (Method B, with a button)
- Problem: the static model needs you to hold still for 1 second, but the dynamic letters (2-Hay, Alifmad, Aray, Jeem) involve movement.
- Method B: "Record" button -> 3-2-1 countdown -> 3 seconds of recording (~15 frames/second) -> dynamic model.
- Model: models/dynamic_rf_T8.joblib, features src/video_features.video_features(T=8), the same as in training (in training every second frame of a 30 FPS video, i.e. ~15 Hz).
- aspect_fix: the training videos were 16:9 and the live camera is 4:3; the difference is reduced by scaling y (can be turned off/on in Settings).
- A hand must be found in at least 5 frames, otherwise "no hand found". If confidence is < 60%, no answer is added, it is only shown.
- Live test log: results/live_dynamic_log.csv (time, true, pred, conf, second, second_conf, n_frames, n_hand, aspect_fix, seconds). "Stats" shows the accuracy and the errors; the "no gesture" tests are counted separately.
- Live accuracy of dynamic (2026-11-09, my own webcam, one signer, 3 s recording, default settings): 4 letters 56/56 correct (2-Hay 10, Alifmad 15, Aray 10, Jeem 21). At confidence >= 0.60, 39/56 (70%) accepted, and all accepted ones were correct. Of the 10 'no gesture' tries, 4 added a false letter (avg confidence 0.55).
- Threshold: results/live_dynamic_threshold.csv (0.5: true gestures accepted 86% / false letters added 50%; 0.6: 70% / 40%; 0.7: 59% / 10%). There are only 10 'no gesture' tries, so I kept 0.60 (not tuned on the test).
- Mistakes in the log: I forgot to change the dropdown; I removed 1 try (I did the wrong gesture) and corrected the label of 5 tries based on the time.

## Auto mode (Method A, motion-triggered, no button): built 2026-11-16, fixed 2026-11-20
- Why: the Record button is awkward; a signer should just make the gesture. Settings -> "Auto mode on" (off by default).
- Idea: speed = average distance moved by the 21 points over the last ~0.2 s, divided by palm length (wrist -> point 9), per second.
  Fast movement for >= 0.25 s starts a segment (with ~0.5 s of earlier frames, so the start is not cut off); the segment ends when the hand is
  still for 0.4 s, after 3 s at most, or when the hand is lost for 0.6 s. The still part at the end is cut. Then the same dynamic model as the button.
- A segment is ignored if the movement is shorter than "Min duration" (0.9 s) or too small ("Min motion" 1.3 palm lengths). After a segment there is a 1.5 s cooldown.
- Defaults: Start threshold 0.80, End threshold 0.35 (must be lower than Start), Min duration 0.9, Min motion 1.3. The video shows "AUTO v x.x" so you can see the hand speed.
- Log: results/live_dynamic_auto_log.csv (own file, not mixed with the button log); "Stats" shows it separately.
- What the screen recording of 2026-11-16 showed (hand near the face):
  1. A resting hand has speed 0.0-0.4 and a slightly moving hand 1.3-1.6, so a Start threshold of 0.40 triggered all the time.
  2. After the dynamic letter (Jeem) the hand stayed in the end pose and the static model added a wrong letter (Wao) and later another one (Alif).
     Reason: motion blur made the hand disappear for a moment, which unlocked the end pose.
  3. Every tiny trigger paused the static letters and showed the red REC chip, so it flickered.
  4. A gesture that went on for the full 3.5 s (hand moving to the face, then the gesture, then back) got a low confidence (Jeem 31%) and was rejected (a safe failure).
- Fixes (branch auto-fixes): the static letters are paused only after the movement has lasted "Min duration" (a tremor does not block them); the REC chip appears only then;
  after a dynamic letter the locked end pose is unlocked only when the hand is away for 1 s (a short blur drop-out does not unlock it) or another letter is added;
  new defaults (above); maximum segment 3.0 s (same as a button recording); the result in the Dynamic card disappears after 8 s; a warning if End >= Start.
- Tests: synthetic hand movements (tremor is ignored and does not block static, a real gesture is recognised and blocks static; the old code added the wrong letter and the new code does not);
  the real app with a fake camera (Record, Auto mode switch, result timeout, no errors). NOT measured yet: the accuracy of Auto mode with a real hand after the fixes.
- Limitation: the dynamic model has no "no gesture" class, so a movement that is not a dynamic letter can still become a letter; the confidence threshold 0.60 and the checks above reduce this but do not remove it.