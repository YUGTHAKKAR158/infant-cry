import os
import glob
import json
import numpy as np
import joblib

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.fractal_features import extract_frta_features
from src.utils.config import PROJECT_ROOT, PROCESSED_DIR, MODELS_DIR, CLASS_NAMES, SAMPLE_RATE

print("=== STEP 1: VERIFYING MODEL ARTIFACTS AND PIPELINE ===")

# Check required files
required_files = [
    os.path.join(MODELS_DIR, "deep_svm_model.joblib"),
    os.path.join(MODELS_DIR, "inca_selector.joblib"),
    os.path.join(MODELS_DIR, "frta_norm_stats.npz"),
    os.path.join(MODELS_DIR, "baby_cry_model_int8.tflite"),
    os.path.join(PROJECT_ROOT, "app", "server.py"),
    os.path.join(PROJECT_ROOT, "app", "templates", "index.html"),
    os.path.join(PROJECT_ROOT, "app", "static", "js", "dashboard.js"),
    os.path.join(PROJECT_ROOT, "app", "static", "css", "dashboard.css"),
]

all_ok = True
for f in required_files:
    exists = os.path.exists(f)
    size_str = f"({os.path.getsize(f):,} bytes)" if exists else "[MISSING]"
    status = "OK" if exists else "FAIL"
    print(f"[{status:4s}] {os.path.basename(f):35s} {size_str}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\nCRITICAL: Some required model or app files are missing!")
else:
    print("\nSUCCESS: All required model artifacts and web application files exist!")

print("\n=== STEP 2: VERIFYING PRESETS AND PREDICTIONS FOR ALL 6 CLASSES ===")

model = joblib.load(os.path.join(MODELS_DIR, "deep_svm_model.joblib"))
selector = joblib.load(os.path.join(MODELS_DIR, "inca_selector.joblib"))
stats = np.load(os.path.join(MODELS_DIR, "frta_norm_stats.npz"))
mean, std = stats["mean"], stats["std"]

# Representative target files from app/server.py
REPRESENTATIVE_SAMPLES = {
    "hungry": "hungry_0721_F4A0C710-AC2.wav",
    "belly_pain": "belly_pain_0032_643D64AD-B71.wav",
    "burping": "burping_0007_efd13da0-d40.wav",
    "discomfort": "discomfort_0007_a40b9b8d-e4b.wav",
    "tired": "tired_0005_999bf14b-e41.wav",
    "non_baby_cry": "115n.wav",
}

results = {}
for cls in CLASS_NAMES:
    target_fname = REPRESENTATIVE_SAMPLES.get(cls)
    folder = os.path.join(PROCESSED_DIR, cls)
    target_path = os.path.join(folder, target_fname) if target_fname else None
    
    if not target_path or not os.path.exists(target_path):
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        target_path = files[0] if files else None
        
    if target_path and os.path.exists(target_path):
        w, sr = load_audio(target_path)
        std_w = standardize_audio(w, sr)
        vec = extract_frta_features(std_w, SAMPLE_RATE)
        norm = (vec[np.newaxis, :] - mean) / std
        inca = selector.transform(norm)
        
        probs = model.predict_proba(inca)[0]
        pred_idx = int(np.argmax(probs))
        pred_cls = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx]) * 100
        
        match_symbol = "MATCH" if pred_cls == cls else "DIFF"
        print(f"[{match_symbol:5s}] Button '{cls:12s}' -> File: {os.path.basename(target_path):30s} | Output: {pred_cls:12s} ({conf:5.1f}%)")
        results[cls] = {"file": os.path.basename(target_path), "predicted": pred_cls, "confidence": round(conf, 1)}

print("\nFull System Verification Complete!")
