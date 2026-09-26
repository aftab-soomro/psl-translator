# Dataset Notes

Source: UAlpha40 (Mendeley, DOI: 10.17632/3pvnnckxyb.3)
Paper: Sajid, Baloch & Javaid, Data in Brief, 2025, DOI: 10.1016/j.dib.2025.111342
Authors: Sameena Javaid, Shahood Sajid, Yusra Khan Baloch (Dept. of Computer Science, Bahria University, Karachi)

- 22,280 images total: 2,896 original + 19,383 augmented (36 static classes)
- 393 videos for 4 dynamic classes (2-Hay, Alifmad, Aray, Jeem) — out of project scope
- Collected with 3 DSLR cameras + LED lighting, volunteers aged 20-45
- Signs follow the FESF standard chart for Urdu Sign Language
- No related published research on this dataset yet (first ML application)
- File naming: sXXXX = signer/volunteer ID (confirmed by our own inspection — same
  sXXXX appears with the same hand/sleeve across different letter folders)
- Some images use .jpg, some .JPG — case must be handled in code


## Decision (confirmed)

- Use only the 36 folders ending in "-Original". Do NOT use "-Augmented" folders.
- Exclude these 4 folders entirely (dynamic/video gestures, out of scope): 2-Hay, Alifmad, Aray, Jeem
- The 36 static classes (folder name without "-Original"):
  1-Hay, Ain, Alif, Bay, Byeh, Chay, Cyeh, Daal, Dal, Dochahay, Fay, Gaaf, Ghain,
  Hamza, Kaf, Khay, Kiaf, Lam, Meem, Nuun, Nuungh, Pay, Ray, Say, Seen, Sheen,
  Suad, Taay, Tay, Tuey, Wao, Zaal, Zaey, Zay, Zuad, Zuey
- Signer ID is embedded in the filename as "sXXXX" (e.g. s0251). This will be used
  for signer-wise train/test splitting (Phase 4) to avoid leakage.