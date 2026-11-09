# Live single-page app (Phase 12, live)

## Kyun
- Purane 3 pages (snapshot, Word Builder, Dynamic video upload) mein photo/video dena parta tha. Ab wo legacy/ mein hain.
- Maqsad: ek hi page, live camera, haath ke 21 points live dikhein, harf khud lafz mein jurein.

## Qadam 1: live camera + 21 points (legacy/live_test.py)
- Library: streamlit-webrtc (browser camera ke frames Python callback tak).
- MediaPipe HandLandmarker VIDEO mode (detect_for_video), timestamp hamesha barhta hua.
- Nateeja: chal gaya, mere laptop par FPS ~27 (haath na ho). Haath aur prediction ke saath FPS ~11-12.

## Qadam 2: prediction + auto lafz (src/live.py, app/live_app.py)
- Model: wahi jo src/predict.py ka MODEL_FILE kehta hai (abhi v3).
- Detection asli (bina ulte) frame par, kyunke training images bhi bina ulti thin; screen par sirf dikhane ke liye ulta.
- Probability averaging: pichhle 0.4 second ki probabilities ka average, phir top harf chuna (flicker kam).
- Harf tab judta hai jab top harf >= 60% yaqeen ke saath 1 second tak wahi rahe (Settings se badal sakte hain).
- Wahi harf dobara tab judega jab koi aur harf jur jaye ya haath hata kar wapas aaye.
- Flicker se bachao: 3 frame lagataar dusra harf aaye tabhi candidate badalta hai; 8 frame haath na mile to "haath hata" maana.
- Faasle screen par: wrist->tip (5) aur paas ki tips (4), palm length ki unit mein.
- Video kahin save nahi hota; sirf "Sunao" par app/audio_cache mein mp3 banti hai.
- Static letters ka live test: (abhi baaki)

## Tuey / Daal ka group (src/live.py, GROUPS)
- Wajah: notebooks/16_tuey_daal_pair.ipynb. Landmarks se Tuey aur Daal alag nahi hote (pair accuracy 0.617, specialist model bhi chance par).
- Isliye jab model in dono mein se ek ko hold tak pohnchata hai, harf seedha nahi judta; "pending" ban jata hai.
- Camera ke upar do button aate hain (1 = Tuey, 2 = Daal); keyboard 1 / 2 se bhi chun sakte hain. Chunne par wahi harf judta hai.
- Rule: sirf wahi pair group banta hai jis mein ghalti 30% se zyada ho. Abhi sirf Tuey-Daal.

## Qadam 3: dynamic letters live (Tareeqa B, button se)
- Masla: static model ko 1 second rukna chahiye, dynamic letters (2-Hay, Alifmad, Aray, Jeem) mein harkat hoti hai.
- Tareeqa B: "Record" button -> 3-2-1 countdown -> 3 second recording (~15 frame/second) -> dynamic model.
- Model: models/dynamic_rf_T8.joblib, features src/video_features.video_features(T=8) wahi jo training mein thi (training mein har doosra frame, 30 FPS ka, yaani ~15 Hz).
- aspect_fix: training videos 16:9 thin, live camera 4:3; y ko scale karke farq kam kiya (Settings mein band/on).
- Kam az kam 5 frames mein haath mile, warna "haath nahi mila". Yaqeen < 60% ho to jawab nahi, sirf dikhaya jata hai.
- Live test log: results/live_dynamic_log.csv (time, true, pred, conf, second, second_conf, n_frames, n_hand, aspect_fix, seconds). "Hisaab" mein accuracy aur galtiyan dikhti hain, "kuch nahi" wale test alag ginte hain.
- Dynamic ka live accuracy (2026-11-09, mera apna webcam, ek signer, 3 s recording, default settings): 4 harf 56/56 sahi (2-Hay 10, Alifmad 15, Aray 10, Jeem 21). Yaqeen >= 0.60 par 39/56 (70%) qubool, qubool hue sab sahi. 'kuch nahi' ki 10 tries mein se 4 par fazool harf jura (avg yaqeen 0.55).
- Threshold: results/live_dynamic_threshold.csv (0.5: asli 86% / fazool 50%; 0.6: 70% / 40%; 0.7: 59% / 10%). 'kuch nahi' ki sirf 10 tries hain, is liye 0.60 hi rakha (test par tune nahi kiya).
- Log ki ghaltiyan: dropdown badalna bhool gaya; 1 try hata di (galat gesture kiya) aur 5 tries ka label time ke hisab se theek kiya.

## Tareeqa A (motion se khud pehchan): abhi nahi
- Plan tha: motion meter -> haath ruka = static, hila = recording -> ruka = dynamic model (src/motion.py likha gaya, live app mein istemal nahi).
- Khatra: dynamic model ke paas "kuch nahi" class nahi; harf badalne ki harkat bhi Jeem/Aray ban sakti hai.
- Faisla: pehle B ka live accuracy dekhna; zaroorat par baad mein.