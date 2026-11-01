# Custom Test Protocol (Phase 11: unseen signers)

**Maqsad:** Dataset ke test (95.02% accuracy) me bhi signers dataset ke hi the (alag log, magar usi camera/roshni me). Yahan hum apne classmates ki photos par model check karte hain, jo bilkul naye log, naya camera aur naya background hain. Dono scores ka farq report me likhenge.

**Sunehri rule:** `data/custom_test/` ki photos se model ko **kabhi train, tune ya select nahi karna**. Ye sirf aakhri imtihan hai.

## 1. Ijazat (pehle, photo se pehle)
Har classmate se kehna (zubani ya WhatsApp message, aur yahan likh lena kis ne haan kaha):
- Photos sirf is ML project ke liye hain.
- **Sirf haath frame me aayega, chehra nahi.** Photos GitHub, report ya slides me nahi jayengi.
- Jab chahein, unki photos delete ho jayengi.

| Classmate ID | Ijazat mili (haan/nahi) | Date | Right/left hand wale | Note |
|---|---|---|---|---|
| c01 | | | | |
| c02 | | | | |
| c03 | | | | |
| c04 | | | | |
| c05 | | | | |
| c06 | | | | |

(Asli naam yahan **mat** likhein, sirf c01, c02... Naam ka alag private note apne paas rakhein.)

## 2. Kitni photos
- 5-6 classmates, har harf ki **2 photos** (3 ho sakein to behtar).
- 36 harf x 2 photos = **72 photos per classmate** (kareeban 10-15 minute).
- Total kareeban 360-430 photos.

## 3. Photo kaise lein (checklist)
- [ ] **Seedha (right) haath**, ek haath, dusra haath frame se bahar.
- [ ] Sada background (deewar, parda), peeche ki cheezein kam.
- [ ] Roshni **saamne se**, haath ke peeche khidki/light na ho.
- [ ] Poora haath aur kalai (wrist) frame me, haath frame ke beech me.
- [ ] Camera haath ke level par, 30-50 cm door (laptop webcam jitni doori).
- [ ] Chehra frame me **nahi**.
- [ ] Har harf pehle dataset ki tasveer (`data/raw/<Harf>-Original/`) ya FESF dictionary (psl.org.pk) se dekh kar banayen.
- [ ] Har harf ki 2 alag photos (haath thora hila kar dobara banayen), ek hi photo copy nahi.
- [ ] Blur ya andheri photo dobara lein.
- [ ] Agar classmate left-handed hai, tab bhi pehle right hand se lein; left-hand test baad me (Phase 12, C2).

## 4. Photos kahan aur kis naam se
Folder: `data/custom_test/<Harf>/` (folder pehle se bane hain, neeche table dekhein).
File ka naam: `c<ID>_<Harf>_<number>.jpg`, jaise `c01_Lam_1.jpg`, `c01_Lam_2.jpg`, `c03_1-Hay_1.jpg`.
- Extension chhota `.jpg` hona chahiye (Windows `.jpg` aur `.JPG` ko same maanta hai, ginti double hoti hai).
- Phone ki `.jpeg` / `.heic` ho to `.jpg` me badal lein.
- `data/custom_test/` gitignored hai, photos GitHub par nahi jayengi. **Kabhi `git add -f` na karein.**

## 5. Harf ki list (folder naam, Urdu, FESF naam)
| # | Folder | Urdu | FESF naam | c01 mukammal | Note |
|---|---|---|---|---|---|
| 1 | Alif | ا | Alif | | |
| 2 | Bay | ب | Bay | | |
| 3 | Pay | پ | Pay | | |
| 4 | Tay | ت | Tay | | |
| 5 | Taay | ٹ | Ttay | | |
| 6 | Say | ث | Say | | |
| 7 | Chay | چ | Chay | | |
| 8 | 1-Hay | ح | Hay | | |
| 9 | Khay | خ | Khay | | |
| 10 | Daal | د | Daal | | |
| 11 | Dal | ڈ | Ddaal | | |
| 12 | Zaal | ذ | Zaal | | |
| 13 | Ray | ر | Ray | | |
| 14 | Zay | ز | Zay | | |
| 15 | Zaey | ژ | Zzay | | |
| 16 | Seen | س | Seen | | |
| 17 | Sheen | ش | Sheen | | |
| 18 | Suad | ص | Suad | | |
| 19 | Zuad | ض | Zuad | | |
| 20 | Tuey | ط | Toay | | |
| 21 | Zuey | ظ | Zoay | | |
| 22 | Ain | ع | Aain | | |
| 23 | Ghain | غ | Ghain | | |
| 24 | Fay | ف | Fay | | |
| 25 | Kaf | ق | Qaaf | | |
| 26 | Kiaf | ک | Kaaf | | |
| 27 | Gaaf | گ | Gaaf | | |
| 28 | Lam | ل | Laam | | |
| 29 | Meem | م | Meem | | |
| 30 | Nuun | ن | Noon | | |
| 31 | Nuungh | ں | Noon Ghunna | | |
| 32 | Wao | و | Wow | | |
| 33 | Dochahay | ھ | Do Chashmi Hay | | |
| 34 | Hamza | ء | Hamza | | |
| 35 | Cyeh | ی | Choti Yay | | |
| 36 | Byeh | ے | Bari Yay | | |

## 6. Jab photos tayar hon
Mujhe batayein kitne classmates ki photos hain. Main check karunga (har harf, har classmate ki ginti, naam ka format), phir landmarks nikal kar model chalaunga (Steps 112-114).
