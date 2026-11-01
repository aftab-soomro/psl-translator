# Viva Q&A

Mumkin sawal aur chhote saaf jawab. Har step ke baad izafa hota hai.

## Dataset aur split
**Q: Augmented images kyun use nahi kin?**
A: Augmented images asli images ki nakal hain. Agar asli train me aur uski nakal test me jati, to model wahi cheez dobara dekhta aur test score jhoota (inflated) hota.

**Q: Signer-wise split kyun, random kyun nahi?**
A: Ek insaan ke haath ki shape (size, ungliyon ki lambai) uski sari photos me same hoti hai. Random split me model insaan pehchan leta, sign nahi. Signer-wise me test par bilkul naye log hote hain, jaise asli use me.

## Landmarks
**Q: Tay ke 46.9% images fail kyun hue?**
A: Tay band mutthi hai, ungliyan overlap karti hain. MediaPipe ko joints alag dikhai nahi dete, to haath detect nahi hota. Isi wajah se Tay ke sirf 43 samples bache.

**Q: Normalization kyun?**
A: Wrist ko origin bana kar aur wrist se point 9 tak ki doori se divide kar ke haath ki position aur size ka asar khatam hota hai. Chhota aur bara haath same features dete hain.

## Models aur testing
**Q: Pipeline me scaler kyun?**
A: Taake scaler har CV fold me sirf training hisse par fit ho. Warna validation ki maloomat leak hoti.

**Q: SVM best kyun chuna?**
A: Tuning ke baad CV 92.19% (SVM) aur test par 95.02% accuracy / 94.82% macro F1, jo chaaron models me sabse behtar. MLP 93.77% test par.

**Q: Test set kab khula?**
A: Sirf ek dafa, Phase 7 me. Models CV se chune gaye.

**Q: Baseline ka kya faida?**
A: Baseline (sab se zyada wali class) sirf ~3% deta hai. Is se pata chalta hai ke hamare 95% asli seekhne ka nateeja hain, qismat ka nahi.

## Custom test (unseen signers)
**Q: Dataset test ke ilawa apna test set kyun?**
A: Dataset test ke signers bhi usi dataset ke the (wahi camera aur roshni). Classmates ki photos naye log, naya camera aur naya background hain. Dono scores ka farq batata hai ke model asli duniya me kitna chalega.

**Q: Custom test se model tune kyun nahi kiya?**
A: Agar tune karte to ye test set bhi seekhne ka hissa ban jata aur score jhoota hota. Ye sirf aakhri imtihan hai.

**Q: Classmates ki privacy ka kya?**
A: Photos me sirf haath, chehra nahi. Folder gitignored hai, GitHub par nahi jata. Classmates sirf c01, c02 ID se likhe jate hain.

## App
**Q: Low-confidence warning har galti pakadti hai?**
A: Nahi. Test me Tuey ko Daal 46.9% confidence se bata diya gaya, jabke threshold 40% hai, to warning nahi aayi. Warning sirf ishara hai, saboot nahi.

**Q: Photo project folder me kyun nahi rakhi?**
A: Photo me chehra hota hai. Isliye `tempfile` ke temporary folder me jati hai aur khud delete ho jati hai.

**Sawal:** Agar koi bina sign ke khuli hatheli dikhaye to model kya karega?
**Jawab:** Wo phir bhi 36 harf mein se ek chunega, hamare test mein Bay 77%. Kyunki training mein "koi harf nahi" class nahi thi. Confidence threshold sirf unhe pakadta hai jahan model khud confuse ho.

**Sawal:** Left hand ke liye kya kiya?
**Jawab:** Train ke landmarks ke x ko ulta kar ke mirror copies bana ke training mein daali. Left hand right ka mirror image hai, is liye model dono seekh gaya. Left hand par accuracy 79.7% se 93.8% hui.

**Sawal:** Test par right-hand score kyun gira jab CV mein barha tha?
**Jawab:** Farq sirf lagbhag 7 images ka hai aur test mein sirf 16 signers hain, shayad shor. Faisla CV se hua tha, test dekh kar nahi badla. Dono nateeje report mein hain.

**Sawal:** Naye features (angles/distances) kyun jode?
**Jawab:** Raw coordinates model ko khud shape nikalni parti hai. Angles aur faasle seedhe batate hain ungli moori hai ya nahi, ye milte julte harfon (Tuey/Daal) mein madad kar sakta hai. CV mein har fold mein behtar aaye.

**Sawal:** CV mein behtar, test mein kam, to kya karein?
**Jawab:** Test mein sirf 562 images/16 signers hain, to ±1 point shor hai. Humne CV (aur signer-wise CV) se chunao kiya aur test ke numbers saaf report kiye. Test dekh kar faisla badalna cheating hoti.

**Sawal:** Word Builder mein harfon ko yaad kaise rakha?
**Jawab:** st.session_state se. Streamlit har click par script dobara chalata hai, session_state ki cheezein bachi rehti hain. Harfon ki list wahan rakhi hai.

**Sawal:** Agar model galat harf bataye to?
**Jawab:** Page top-3 predictions dikhata hai, user sahi wala chun kar jod sakta hai. Ye is liye zaroori hai ke Tuey/Daal jaisi milti julti jodiyon mein model confuse hota hai.

**Sawal:** Agar demo ke waqt internet na ho to awaaz ka kya hoga?
**Jawab:** Har bola hua lafz mp3 ki shakal mein app/audio_cache mein save hota hai. Demo ke lafz pehle bol kar cache kar lete hain, to bina internet bhi chalte hain.

**Sawal:** Clustering ka tumhare project se kya taalluq hai?
**Jawab:** Humne 36 harfon ke average shapes ko bina naam ke group kiya. Jin harfon ko hamara classifier confuse karta hai (Tuey/Daal, Zuad/Zuey, Ain/Ghain) wo clustering mein bhi qareeb nikle. To confusion ki wajah shapes ki samanta hai.

**Sawal:** k-means mein k kaise chuna?
**Jawab:** Elbow aur silhouette se dekha. Elbow mein saaf mor nahi tha, silhouette k=2 par oonchi (sirf Pay aur Kaf alag) aur baaqi ~0.3, yaani groups kamzor. Isliye koi ek "sahi k" nahi.

**Sawal:** Haath ka kaun sa hissa sabse ahem nikla aur kyun?
**Jawab:** Random Forest ke hisaab se ungliyon ke sire (fingertips), khaaskar middle aur thumb tip. Wajah ye ho sakti hai ke harf ke sign mein ungliyan sabse zyada alag hoti hain. Wrist ki importance 0 hai kyunki normalization mein wo origin hai.

**Sawal:** Augmented images kyun nahi use ki shuru mein, aur baad mein kya hua?
**Jawab:** Agar ek signer ki original train mein aur copy test mein ho to leakage hota. Baad mein sirf train signers ki copies jodi. CV mein farq +0.1 point tha, yaani lagbhag koi faida nahi, kyunki landmarks aur normalization roshni/zoom ka asar pehle hi hata dete hain.