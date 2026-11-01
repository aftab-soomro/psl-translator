# Progress Log

Chronological record: har step ka kya/kyun/kaise, files, numbers, aur "simply explain".
Numbers hamesha `results/*.csv` se. Purane numbers kabhi delete nahi hote.

> **Phase 0-9 backfill note (2026-11-01):** Neeche Phase 0-9 ki entries git history
> (`git log`) aur `results/*.csv` se baad me likhi gayi hain, us waqt ke waqt par nahi.
> "Explain it simply" paragraphs Aftab/Meraj ko parh kar apne alfaaz me theek karne hain.

---

## Phase 0-1: Setup aur dataset (2026-09-26 / 27)
- **Kya:** .venv, requirements, GitHub repo, `ai_use_log.md`; UAlpha40 dataset (Mendeley, CC BY 4.0) download aur samjha.
- **Faisla:** sirf 36 `-Original` folders; augmented aur 4 dynamic folders (2-Hay, Alifmad, Aray, Jeem) bahar. Signer ID filename me (sXXXX).
- **Numbers:** 2,896 original images (paper: 2,897), 82 signers.
- **Commits:** 769e8ae, 00b4ebd, b7ab6a0, 703cd70, 4107ebb, 05bad52.
- **Simply:** Dataset me asli photos ke ilawa unki nakal (augmented) bhi thi. Nakal rakhte to test me wahi cheez dobara aati, isliye sirf asli photos li.

## Phase 2: Landmarks (2026-09-27)
- **Kya:** `src/landmarks.py` (MediaPipe Tasks API, 63 numbers ya None), `src/features.py` (wrist origin, size se divide).
- **Numbers:** 2,727 kamyab, 169 fail (5.8%). Tay: 43/81 fail (46.9%), sirf 43 usable.
- **Wajah (Tay):** band mutthi, ungliyan overlap, MediaPipe joints alag nahi kar pata.
- **Commits:** 3871da7, 69ec89b, dd0b5c4.
- **Simply:** Photo se seedha seekhne ke bajaye haath ke 21 points nikale. Points ko wrist se naap kar chhota-bara haath same kar diya.

## Phase 3: EDA (2026-09-27)
- 4 plots `reports/figures/01-04`: class distribution, average hand shapes, milte julte harf overlay, PCA.
- **Commit:** b2dd7ee.

## Phase 4: Split aur leakage audit (2026-09-27)
- **Kya:** signer-wise split, seed 42, 20% signers test me.
- **Numbers:** train 2,165 (66 signers), test 562 (16 signers). Overlap 0. Tay 31/12.
- **Commit:** 6a5055c. Detail: `notes/leakage_audit.md`.
- **Simply:** Ek insaan ki saari photos ya train me ya test me. Warna model insaan ka haath yaad kar leta, sign nahi.

## Phase 5-6: Baseline, models, tuning (2026-09-27)
- Sab models `Pipeline(StandardScaler + model)`, 5-fold CV sirf train par.
- **CV (default -> tuned):** Baseline 3.05; KNN 82.49 -> 82.77 (k=3); SVM 86.93 -> 92.19 (C=100, gamma=0.01); RF 88.50 -> 88.68 (depth 20, 300 trees); MLP 91.36 -> 92.01 ((100,50), lr 0.001).
- **Commits:** d72830c, d790d5c. Files: `cv_results.csv`, `tuning_results.csv`.
- **Simply:** Scaler pipeline ke andar hai taake wo sirf training hisse se seekhe. Tuning ne SVM ko sabse zyada faida diya.

## Phase 7: Final test (2026-09-27)
- Test set sirf yahan khula. Test: SVM 95.02% acc / 94.82% macro F1, MLP 93.77, RF 91.46, KNN 88.26, Baseline 2.85.
- **Best:** SVM (RBF). Files: `final_results.csv`, `confusion_matrix.csv`, `per_class_report.csv`, figure 05.
- Sabse bari confusion: Tuey<->Daal (9+3). Kamzor F1: Tuey 0.538, Daal 0.684, Ain 0.828, Ghain 0.857, Tay 0.857.
- **Commits:** 6199ede, 7984d93.
- **Simply:** Test set woh imtihan hai jo ek hi dafa diya. CV se model chuna, test se sirf nateeja dekha.

## Phase 8: Error analysis (2026-09-27)
- 8 galtiyan, 2 pattern: (1) closed-fist occlusion (Alif, Nuun, Taay), (2) milti julti shapes (Ain/Ghain, Daal/Tuey, Ray/Zaal). Nuungh = angle.
- **Commit:** e2715c0. Figure 06.

## Phase 9: Model save aur predict() (2026-09-27, core freeze tag 2026-10-25)
- `models/best_model.joblib`, `class_names.joblib`, `src/predict.py`. Notebook 08 ne 10 test images par check kiya (roadmap: 10/10).
- **Commits:** 52f9875, 6f7c190 (tag `core-v1`; `best_model_core_v1.joblib` backup, hash same).
- **Note (2026-11-01):** 6f7c190 me `06_evaluation.ipynb` me sirf 50 lines (model save cells) badli; test numbers wahi hain.

---

## 2026-11-01 | Phase 10 | Step 116 (code) B4: top-3 + low-confidence warning
- **Kya:** `app.py` me top-3 andaze aur 40% se kam confidence par warning. `predict_proba` use hua, model nahi badla.
- **Commit:** 22bfc00 ("tests pending").
- **Simply:** Model sirf ek jawab nahi, top 3 batata hai. Kam yaqeen par kehta hai "dobara try karo".

## 2026-11-01 | Phase 10 | Step 109 (core freeze) mukammal
- **Kya/kyun:**
  - `tempfile.TemporaryDirectory` (photo ka temporary folder khud delete hota hai, project folder me `temp_snapshot.jpg` nahi banti).
  - `.gitignore` me `temp_snapshot.jpg` (chehra repo me na jaye).
  - `use_container_width=True` -> `width="stretch"` (Streamlit ne purana tareeqa deprecate kiya).
  - `notes/01_dataset_overview.ipynb` -> `notebooks/` (galat folder me tha, `git mv`).
- **Test (Streamlit AppTest, camera nakli bheja):**
  - Khali safed image: warning "Haath nahi mila...", exception 0 (Step 106 verified).
  - Lam: Lam 85.9%. Alif: Alif 87.5%. Tuey: **Daal 46.9% (galat)**, warning nahi aayi kyunki 46.9% > 40% threshold.
  - Root folder me `temp_snapshot.jpg` nahi bani.
- **Nateeja/sabaq:** Tuey->Daal wohi known confusion hai (Phase 7). Low-confidence warning har galti nahi pakadti; ye report ki limitation hai.
- **Simply:** App ko safai di: photo apne aap delete hoti hai, purani warnings hati. Aur dekha ke app ghalat jawab bhi kabhi kabhi bharose se deti hai.
- **Windows note:** `streamlit.exe` Aftab ke laptop par Application Control policy se block hai. Chalane ka tareeqa: `python -m streamlit run app/app.py`.

## 2026-11-01 | Phase 10 | Step 107 (10 harf manual test) aur Step 108 (Meraj)
- **Step 107:** Aftab ne zubani bataya ke 10 harf webcam se test kar liye (suggested list: Alif, Lam, Meem, Wao, Kaf, Tuey, Daal, Ain, Tay, Nuun). **Harf-wise nateeje yahan record nahi hue.** Agar report me app testing ka zikr karna ho to ye table baad me bharni hogi.
- **Khuli hatheli (open palm) test:** nateeja record nahi hua. Step 116 ke teen tests me se ye baaqi hai.
- **Step 108:** Meraj ke laptop par test baad me hoga. Status: baaqi.
- **Push:** Phase 10 poora nahi hua (Step 108 baaqi), isliye `ccaac18` tak ke commits abhi push nahi hue.

## 2026-11-01 | Phase 11 | Step 110-111 tayari (classmates ki photos)
- **Kya:** `notes/custom_test_protocol.md` likha (ijazat, kitni photos, photo checklist, file naming `c01_Lam_1.jpg`, 36 harf ki table). `data/custom_test/<Harf>/` ke 36 folder bana diye (gitignored, `git check-ignore` se verify).
- **Kyun:** Dataset test me bhi signers dataset wale the. Naye log, naya camera aur background par score kya aata hai, ye real-world ke qareeb hai.
- **Faisla:** Photos me sirf haath, chehra nahi (privacy). Classmates ko sirf ID (c01...) se likhenge, naam nahi. custom_test se kabhi train/tune/select nahi.
- **Simply:** Ye "ghar ka imtihan" hai. Class ke imtihan (dataset test) me aur ghar me bilkul naye logon ke saamne model ka score farq hoga, aur wo farq hi sab se sachi baat batata hai.
- **Status:** Photos lena abhi baaqi (Aftab/Meraj ka kaam).
## Step 116 (B4): Top-3 + low-confidence warning, 2026-11-01
- Kya: app/app.py mein top-3 predictions aur 40% se kam confidence par warning (code pehle se tha). Teen manual tests kiye.
- Nateeje: Alif 87.5% sahi; Tuey -> Daal 46.9% (galat, warning nahi aayi); khuli hatheli -> Bay 77.1% (warning nahi aayi). Top-3: Bay 77.1, Say 7.6, Fay 4.2.
- Nateeja: warning sirf "model confuse" ko pakadti hai. Model 36 harf mein se hi chunta hai, is liye kisi bhi haath par confident jawab de sakta hai.
- Explain it simply: model ke paas "pata nahi" ka option nahi hai. Wo hamesha 36 mein se ek harf chun leta hai, chahe haath kisi harf ka na ho.
- Limitation (report ke liye): out-of-distribution input par confidence bharosemand nahi.

## Step 117 (C2): Left-hand support, 2026-11-01
- Kya: notebooks/09_left_hand.ipynb. Left hand = right hand ka mirror, normalized landmarks mein x ulta (x *= -1).
- CV (5-fold, SVM C=100 gamma=0.01, train.csv): baseline right 0.9187, left (bina fix) 0.7760, (b) flip 0.9187, (a) mirror-train right 0.9316, left 0.9312. (a) har fold mein baseline se behtar (+0.5 se +2.3 points).
- Faisla (CV se): (a) mirror-train. Model: models/best_model_v2_mirror.joblib. Core model chhua nahi.
- Test (results/advanced_left_hand_test.csv): core right 0.9502, core left 0.7972, (a) right 0.9377, (a) left 0.9377.
- Explain it simply: left hand right hand ka shishe wala aks hai. Model ko dono tarah ke haath dikha kar sikhaya, to left hand par accuracy 80% se 94% ho gayi, right hand par thora kam (95% se 93.8%).

- App test: predict.py ab best_model_v2_mirror.joblib use karta hai (MODEL_FILE constant, core v1 backup mehfooz). Notebook 08: 10/10 sahi (Tuey sahi par 38.69%, threshold se neeche). Webcam: Lam dono haathon se pehchana, 75.5% aur 83.5%.

## Step 118 (B3): Angles aur fingertip distances, 2026-11-01
- Kya: src/features.py mein add_engineered_features() (30 naye features: 5 tip-to-wrist, 10 tip-to-tip, 15 joint angles). Notebook 10_features_angles.ipynb.
- CV (StratifiedKFold, 5-fold): raw63+mirror 0.9316; eng93+mirror (gamma 0.01) 0.9483, gamma 0.0068: 0.9460. Har fold mein behtar.
- Test (results/advanced_features_test.csv): core v1 0.9502/left 0.7972; v2 0.9377; v3 (chosen) 0.9324; v3b 0.9342. CV aur test ki ranking alag.
- Shak tha ke CV leaky hai; signer-wise CV (StratifiedGroupKFold, results/advanced_grouped_cv.csv) ne wahi ranking di (A 0.9252, B 0.9311, C 0.9499, D 0.9458 right-hand). Shak ghalat nikla.
- Faisla (CV se): v3 = models/best_model_v3_eng.joblib. Core aur v2 mehfooz.
- Explain it simply: model ko sirf points dene ke bajaye "ungli kitni moori hai" aur "angootha index ko chhoo raha hai" jaisi seedhi baatein bhi di. CV mein 1.7 points behtar, test mein thora kam, farq chhota hai.
- App test: predict.py ab v3 use karta hai. Notebook 08: 9/10 sahi (Tuey -> Daal 44.54%, warning nahi aayi; Suad sahi par 66.8%).

## Step 119 (A1): Word Builder page, 2026-11-01
- Kya: app/pages/1_Word_Builder.py (nayi page) aur src/letters.py (LETTER_MAP ki copy, abhi app.py mein bhi hai).
- Kaise: st.session_state mein harfon ki list; webcam photo -> predict() -> top-3 mein se sahi harf chuno -> "Harf jodo". Backspace, Space, Clear buttons. Har harf ke baad camera key badal kar khali.
- Test: manual, 3 harf jode, backspace aur clear theek chale.
- Explain it simply: Streamlit har click par page dobara chalata hai, to normal variable bhool jata hai. session_state ek yaaddasht hai jo bachi rehti hai. Top-3 se chunne se model ki galti bhi theek ho sakti hai.
- Baaqi safai: LETTER_MAP ab do jagah hai (app.py aur src/letters.py), Phase 14 mein app.py ko import par laga dena.

## Step 120 (A2): Lafz ko Urdu mein bolna (gTTS), 2026-11-01
- Kya: Word Builder page mein "Lafz sunao" button. gTTS (lang="ur") se mp3 banti hai, st.audio se chalti hai.
- Fallback: har lafz ki mp3 app/audio_cache/<md5>.mp3 mein cache hoti hai. Cache wala lafz bina internet chalta hai. Naya lafz bina internet par warning aati hai.
- Test: awaaz Urdu mein aayi, cache file ban gayi (~7 KB har ek).
- Explain it simply: gTTS lafz Google ko bhejta hai aur awaaz wapas milti hai (internet chahiye). Ek baar bani awaaz hum save rakhte hain, is liye doosri dafa offline bhi chalti hai.
- .gitignore: app/audio_cache/*.mp3. Demo se pehle sirf demo ke lafz ki files git add -f se commit karni hain.

## Step 121 (B2): Clustering of 36 average hand shapes, 2026-11-01
- Kya: notebooks/11_clustering.ipynb. train.csv se har harf ka average shape (36 x 63). Hierarchical (Ward) dendrogram, pairwise distance rank, k-means (elbow, silhouette).
- Files: reports/figures/07_dendrogram.png, 08_kmeans_elbow_silhouette.png; results/advanced_cluster_pairs.csv, advanced_kmeans_k.csv.
- Pair ranks (630 jodiyon mein, 1 = sabse qareeb): Tuey-Daal 1, Zuad-Zuey 6, Ain-Ghain 26, Ray-Zaal 100.
- k-means: k=2 silhouette 0.635 (sirf Kaf, Pay alag); k=3..12 silhouette 0.305-0.382 (k=5: 0.382). Elbow mein saaf mor nahi. Matlab shapes saaf alag groups nahi banate.
- k=5: Tuey/Daal ek cluster mein, Ain/Ghain apne cluster mein, Zuad/Zuey ek cluster mein, Ray aur Zaal alag clusters mein.
- Explain it simply: clustering mein harf ke naam nahi batate, sirf shapes se groups banwate hain. Jo harf model confuse karta hai wo shapes mein bhi qareeb nikle, to confusion ki wajah shapes ki asli samanta hai.
- Limit: Meem-Nuun (rank 2) bhi bohat qareeb hain, par model ne unko zyada confuse nahi kiya. Qareeb hona zaroori hai, kaafi nahi.

## Step 122 (B1): Random Forest feature importance on hand skeleton, 2026-11-01
- Kya: notebooks/12_feature_importance.ipynb. RF (300 trees, depth 20, seed 42) train.csv ke 63 features par. Har landmark ki importance = x+y+z ka jod. Average haath ke skeleton par rang se.
- Files: reports/figures/09_feature_importance_skeleton.png, results/advanced_feature_importance.csv.
- Top: middle_tip 0.0896, thumb_tip 0.0873, pinky_tip 0.0654, ring_tip 0.0640, middle_dip 0.0626, index_tip 0.0622. Wrist 0 (normalization se origin).
- Explain it simply: RF ke darakhton ne harf alag karne ke liye sabse zyada ungliyon ke sire (aur unke qareeb ke joints) istemal kiye. Hatheli ke neeche ke points kam kaam aaye.
- Limit: RF ki importance milte julte features mein baant jati hai, aur tips zyada hilte hain, is liye ise "sach" nahi, "RF ne kya istemal kiya" samjhein.

## Step 124: Augmented-image experiment, 2026-11-01
- Kya: notebooks/13_augmented_experiment.ipynb. Zip se (extract kiye bina) 13,795 augmented images jo sirf train signers ki hain; landmarks nikale -> 11,821 kamyab (data/processed/aug_landmarks.csv, gitignored). Test signers (16) ki 5,516 copies bilkul istemal nahi hui.
- Checks: train/test signers overlap 0; augmented signers sab train signers mein se; validation fold ke signers ki copies training mein nahi (assert).
- Signer-wise CV (5-fold, eng93, SVM C=100 gamma=0.01, bina mirror): D originals only 0.9458; E originals+augmented 0.9468 (+0.1 point, folds mila-jula).
- Test (results/advanced_augmented_test.csv): D 0.9413; E 0.9520.
- Faisla (CV se): augmented apnane ki wajah nahi (data 6.5x, score lagbhag barabar). Model v3 hi rakha.
- Tay ki sirf 166 augmented copies mein haath mila (baaqi 278-420), wahi band mutthi wala masla.
- Explain it simply: augmented copy mein roshni/zoom badalta hai, magar landmarks aur normalization ye farq pehle hi hata dete hain, to copy se wahi numbers nikalte hain. Naye signers (naye haath) se faida hota, copies se nahi.