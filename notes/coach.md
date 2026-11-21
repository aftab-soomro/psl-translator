# PSL Coach (practice mode): which finger is wrong?

Added 2026-11-21 on branch auto-fixes. Code: `src/coach.py`, test: `src/coach_eval.py` + `notebooks/17_coach.ipynb`.

## Why
The app can say which letter a hand shows. A person who is learning PSL (for example a hearing parent or teacher of a deaf child)
also needs to know what to change when the sign is not right yet. PSL learning apps we found (PSL by Deaf Reach, PSL Isharay)
only show videos and do not use the camera; an ASL app (fingerspelling.xyz) gives right/wrong feedback but does not name the finger.
We did not find a PSL tool that names the finger to fix (this is "not found in our search", not "nobody did it").

## How it works
1. 19 angles per hand, all easy to explain: 15 finger bends (3 joints per finger, 0 = straight, the same joints as the 93
   model features) and 4 spreads (angle between neighbouring fingers). Angles do not change with hand position, size or
   mirroring (left hand).
2. Reference: for every letter, the median of each angle over the training photos (66 signers) and the normal variation
   between signers (robust standard deviation = 1.4826 x MAD, at least 10 degrees for a bend and 6 for a spread).
3. Live: angles are averaged over 0.5 s, then z = (learner - median) / variation for every angle.
   |z| > 2.5 gives a tip for that finger: "Straighten your index at the middle joint", "Bend your thumb more at the base",
   "Spread your index and middle apart". At most 2 tips, one per finger, the biggest first; a bend tip wins over a spread tip.
4. The static model still decides if the sign is right (probability of the target letter >= 0.60; for Tuey/Daal the
   group probability). If it is not, the tips are shown and the worst finger is red on the video. If it is right but one
   finger is still very far from the photos (|z| > 4), an optional "Even better: ..." tip is shown in orange.
5. After the model accepts the sign for 1 second: "Done! Pick the next letter". Letters are not added to the word in
   practice mode, and Auto mode is paused.

UI: sidebar -> "Practice (Coach)" -> pick a letter (or "Next letter"). The Live recognition card shows the photo of the
letter, the match (model probability), five finger chips (green / orange / red) and the tips. On the video each finger is
coloured the same way.

## Test (signer-wise, no leakage)
We cannot ask 66 signers to make mistakes, so we made them on purpose in the landmark data: on every held-out photo one
finger was bent more or straightened by theta degrees at its middle joint. 5-fold GroupKFold over the training signers;
reference and model rebuilt on the training folds only; the sealed test set was not used. Chance for naming the finger: 20%.

| theta | correct photo: wrong tip shown | model says "not right yet" | coach names the finger (when not right) | right direction | model accepts anyway: "even better" tip names the finger |
|---|---|---|---|---|---|
| 25 deg | 5.3% | 11.5% | 34.7% | 35.4% | 63.0% |
| 40 deg | 5.3% | 16.0% | 63.0% | 62.0% | 87.1% |
| 60 deg | 5.3% | 25.3% | 84.8% | 82.3% | 93.7% |

(results/coach_eval.csv, results/coach_eval_by_finger.csv, reports/figures/coach_eval_by_finger.png)

- The "even better" tip appears on only 4.9% of correct photos.
- Finding: the static model still accepts 84% of hands with one finger 40 degrees off (it is trained to be tolerant).
  The coach notices these: at 40 degrees its "even better" tip appears in 40% of them and names the right finger 87% of the time.
- By finger (40 deg): thumb, index and middle are best (about 70%); ring and pinky are harder (about 50-60%);
  "straighten your thumb" is the weakest tip (39%).

Design choices made with this test (cross-validation only): 3D angles were better than 2D (x, y only);
median + MAD was better than mean + std; threshold 2.5 (lower floors) named the finger best. Without the model gate the coach
would complain on 27% of correct photos; with it only 5%.

## Limits (say them honestly)
- The mistakes in the test are simulated (one finger rotated), not made by real learners. A live test with real learners is
  still needed.
- The reference comes from studio photos; MediaPipe's depth (z) is noisy; angles cannot tell hyperextension from bending.
- Palm direction (where the palm faces) is not checked; if every finger looks fine but the model still says no, the card
  says to compare the palm direction with the photo.
- The finger-by-finger tips work for the 36 static letters only. The 4 moving letters have a simpler practice mode (below).

## Practice for the 4 moving letters (Jeem, Aray, Alifmad, 2-Hay)
- The same Practice list also has the 4 dynamic letters (marked "moving"). A movement is not a pose, so there are no finger tips.
- Press Record (3-2-1) or switch Auto mode on, then do the gesture. The app runs the normal dynamic model and says:
  "Correct! Jeem 91%", "Right letter but not clear (45%): do it slower and bigger", "That looked like Aray (70%), try again",
  "Movement too short" or "Hand not seen enough". A small counter shows "2 of 3 attempts correct".
- Nothing is added to the word and practice attempts are NOT written to `results/live_dynamic_log.csv` or the Auto log,
  so the numbers in Stats (the live test) stay clean.
- `make_dynamic_guide.py` makes one small GIF per moving letter from the UAlpha40 videos (it picks the video the model is
  most sure about). The GIFs appear in the Letter guide and in the practice card. Without them the app shows a placeholder.

## Also fixed
- 2-Hay was shown as do-chashmi he; it is now Choti Hay, as written in notes/dataset_notes.md (the dynamic letter card title
  and the word use the right letter now). The four dynamic glyphs in the card title are now large and clear (before they were
  tiny and faint, so the Choti Hay was hard to see).
