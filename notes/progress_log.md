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
