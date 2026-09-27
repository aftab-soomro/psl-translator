# Leakage Audit

Har wo faisla jo humne data leakage rokne ke liye liya, aur uski wajah.

## 1. Augmented images bilkul use nahi kin

UAlpha40 dataset me 2,897 original images ke ilawa 19,383 "augmented" copies
bhi thin (noise, rotation, brightness ke saath banai gayi). Humne sirf
2,896 original images use kin (1 image mismatch, farq negligible).

Wajah: augmented images asal images ki hi halki-badli hui copies hain. Agar
ek original train me jata aur uski copy test me, to model wahi haath "yaad"
kar leta jo usne train me dekha tha, aur test accuracy jhooti (inflated) hoti.

## 2. Signer-wise train/test split (random split nahi)

Dataset me har image ke filename me signer ID (sXXXX) maujood hai — 82 alag
log. Humne row-level random split ki bajaye SIGNER-level split kiya: 65-66
signers train me, 16-17 signers test me, koi signer dono me nahi.

Wajah: agar ek hi insaan ka ek sign train me hota aur dusra sign test me, to
model us insaan ke haath ki shape (size, ungliyon ki lambai) pehchan leta,
sign ko nahi. Signer-wise split ensure karta hai ke model test set me
BILKUL NAYE logon ke haath dekhta hai, jo real-world use jaisa hai.

Verified (Step 55-56):
- Saari 36 classes dono train aur test me maujood hain
- Koi file dono sets me nahi (overlap = 0)
- Koi signer dono sets me nahi (signer overlap = 0)
- Random seed 42 fix hai, taake split reproduce ho sake

## 3. Feature-level leakage check

Humare 63 features (21 landmarks x x,y,z) sirf ek hi image ke andar se
nikalte hain -- MediaPipe har image ko ALAG, ek-dusre se bekhabar process
karta hai. Koi feature aisa nahi jo:
- kisi doosri image ki maloomat use kare
- prediction ke waqt maujood na ho (jaise "asal jawab" ka koi hint)
- future data use kare (dataset time-series nahi hai, is liye ye lagoo nahi)

## 4. Preprocessing sirf training data par fit hoga

Jab Phase 5-6 me hum scaling (StandardScaler) use karenge, wo hamesha
scikit-learn ke Pipeline ke andar hoga, aur sirf TRAINING fold par fit
hoga -- kabhi poore dataset (train+test) par nahi. Isse test set ki
koi bhi maloomat (jaise uska average/spread) training ke waqt "leak"
nahi hogi.

## 5. Test set discipline

test.csv Phase 4 (Step 54) me bana, aur usko Phase 7 (Step 81) tak
nahi khola jayega -- na EDA ke liye, na model choice ke liye, na
hyperparameter tuning ke liye. Sirf train.csv par cross-validation
se sab faisle honge.