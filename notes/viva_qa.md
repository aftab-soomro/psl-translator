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