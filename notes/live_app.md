# Live single-page app (Phase 12, live)

## Kyun
- Purane 3 pages (snapshot, Word Builder, Dynamic video upload) mein photo/video dena parta tha.
- Maqsad: ek hi page, live camera, haath ke 21 points live dikhein, harf khud lafz mein jurein.

## Qadam 1: live camera + 21 points (app/live_test.py)
- Library: streamlit-webrtc (browser camera ke frames Python callback tak).
- MediaPipe HandLandmarker VIDEO mode (detect_for_video), timestamp hamesha barhta hua.
- Nateeja: chal gaya, mere laptop par FPS ~27.

## Qadam 2: prediction + auto lafz (src/live.py, app/live_app.py)
- Model: wahi jo src/predict.py ka MODEL_FILE kehta hai (abhi v3).
- Detection asli (bina ulte) frame par, kyunke training images bhi bina ulti thin; screen par sirf dikhane ke liye ulta.
- Harf tab judta hai jab top harf >= 60% yaqeen ke saath 1 second tak wahi rahe (sidebar se badal sakte hain).
- Wahi harf dobara tab judega jab koi aur harf jur jaye ya haath hata kar wapas aaye.
- Flicker se bachao: 3 frame lagataar dusra harf aaye tabhi candidate badalta hai; 8 frame haath na mile to "haath hata" maana.
- Faasle screen par: wrist->tip (5) aur paas ki tips (4), palm length ki unit mein.
- Video kahin save nahi hota; sirf "Sunao" par app/audio_cache mein mp3 banti hai.
- Static letters ka live test: (abhi baaki)

## Qadam 3: dynamic letters live (Tareeqa A, motion se)
- Masla: static model ko 1 second rukna chahiye, dynamic letters mein harkat hoti hai.
- Plan: motion meter -> haath ruka = static, hila = recording -> ruka = dynamic model.
- Khatra: dynamic model ke paas "kuch nahi" class nahi; harf badalne ki harkat bhi Jeem/Aray ban sakti hai.
- Agar A theek na chala to Tareeqa B (button + 2 second fixed recording).