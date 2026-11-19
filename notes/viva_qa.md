# Viva Q&A

Likely questions with short, simple answers. More are added after every step.

## Dataset and split
**Q: Why did you not use the augmented images?**
A: Augmented images are copies of the real images. If the real image went to train and its copy went to test, the model would see the same thing again and the test score would be inflated.

**Q: Why a signer-wise split and not a random split?**
A: The shape of a person's hand (size, finger length) is the same in all their photos. With a random split the model would recognise the person, not the sign. With a signer-wise split the test set has completely new people, as in real use.

## Landmarks
**Q: Why did 46.9% of the Tay images fail?**
A: Tay is a closed fist and the fingers overlap. MediaPipe cannot see the joints separately, so the hand is not detected. For this reason only 43 Tay samples were left.

**Q: Why normalization?**
A: We put the wrist at the origin and divide by the distance from the wrist to point 9. This removes the effect of the hand's position and size. A small hand and a big hand give the same features.

## Models and testing
**Q: Why is the scaler inside the pipeline?**
A: So the scaler is fitted only on the training part in every CV fold. Otherwise information from the validation data would leak.

**Q: Why did you choose the SVM as the best model?**
A: After tuning it had a CV accuracy of 92.19% (SVM) and on the test set 95.02% accuracy / 94.82% macro F1, the best of the four models. The MLP got 93.77% on the test set.

**Q: When was the test set opened?**
A: Only once, in Phase 7. The models were chosen with CV.

**Q: What is the use of the baseline?**
A: The baseline (always the most frequent class) gives only ~3%. This shows that our 95% comes from real learning, not luck.

## Custom test (unseen signers)
**Q: Why do you have your own test set besides the dataset test set?**
A: The signers of the dataset test set were also from the same dataset (the same camera and light). The classmates' photos are new people, a new camera and a new background. The difference between the two scores tells how well the model works in the real world.

**Q: Why did you not tune the model with the custom test?**
A: If we had, this test set would also become part of the learning and the score would be false. It is only the final exam.

**Q: What about the classmates' privacy?**
A: The photos show only the hand, not the face. The folder is gitignored and does not go to GitHub. Classmates are written only by the IDs c01, c02.

## App
**Q: Does the low-confidence warning catch every error?**
A: No. In the test Tuey was called Daal with 46.9% confidence, while the threshold is 40%, so no warning appeared. The warning is only a hint, not a proof.

**Q: Why did you not keep the photo in the project folder?**
A: A photo contains a face. So it goes into a `tempfile` temporary folder and deletes itself.

**Question:** If someone shows an open palm without any sign, what will the model do?
**Answer:** It will still pick one of the 36 letters; in our test it was Bay 77%. This is because there was no "no letter" class in training. The confidence threshold catches only the cases where the model itself is confused.

**Question:** What did you do for the left hand?
**Answer:** We flipped the x of the training landmarks to make mirror copies and added them to training. A left hand is the mirror image of a right hand, so the model learned both. The accuracy on the left hand went from 79.7% to 93.8%.

**Question:** Why did the right-hand score fall on the test set when it had risen in CV?
**Answer:** The difference is only about 7 images, and the test set has only 16 signers, so it is probably noise. The decision was made with CV and not changed after looking at the test set. Both results are in the report.

**Question:** Why did you add the new features (angles/distances)?
**Answer:** With raw coordinates the model has to find the shape by itself. Angles and distances tell it directly whether a finger is bent, which can help with similar letters (Tuey/Daal). They were better in every CV fold.

**Question:** Better in CV but lower on the test set, what do you do?
**Answer:** The test set has only 562 images/16 signers, so +/-1 point is noise. We chose with CV (and signer-wise CV) and reported the test numbers clearly. Changing the decision after looking at the test set would be cheating.

**Question:** How did the Word Builder remember the letters?
**Answer:** With st.session_state. Streamlit runs the script again on every click, and things in session_state stay. The list of letters is kept there.

**Question:** What if the model gives a wrong letter?
**Answer:** The page shows the top-3 predictions, and the user can pick the right one and add it. This is needed because the model gets confused on similar pairs like Tuey/Daal.

**Question:** What happens to the voice if there is no internet during the demo?
**Answer:** Every spoken word is saved as an mp3 in app/audio_cache. We speak the demo words once beforehand to cache them, and then they work without internet.

**Question:** How is clustering related to your project?
**Answer:** We grouped the average shapes of the 36 letters without their names. The letters that our classifier confuses (Tuey/Daal, Zuad/Zuey, Ain/Ghain) were also close in the clustering. So the reason for the confusion is the similarity of the shapes.

**Question:** How did you choose k in k-means?
**Answer:** We looked at the elbow and the silhouette. The elbow had no clear bend, and the silhouette was high at k=2 (only Pay and Kaf separate) and about 0.3 for the rest, which means weak groups. So there is no single "right k".

**Question:** Which part of the hand turned out to be the most important, and why?
**Answer:** According to the Random Forest, the finger tips, especially the middle and thumb tips. The reason may be that in the signs of the letters the fingers differ the most. The importance of the wrist is 0 because it is the origin in the normalization.

**Question:** Why did you not use the augmented images at the start, and what happened later?
**Answer:** If the original of a signer were in train and a copy in test, there would be leakage. Later we added only the copies of train signers. In CV the difference was +0.1 point, so almost no benefit, because landmarks and normalization already remove the effect of light/zoom.

**Question:** Why a separate model for the dynamic letters?
**Answer:** Static letters are recognised from one picture, but a dynamic letter needs the movement of the hand. We took the landmarks of the video frames, split them into a fixed number of time steps, and made one long feature list.

**Question:** Why is the CV accuracy on the videos as high as 99%?
**Answer:** There are only 4 letters and their movements are very different. The CV was signer-wise (new signers). On the test set it was also 96.7%, so there is no leakage.
