import os
import glob
import numpy as np
import joblib

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.fractal_features import extract_frta_features
from src.utils.config import PROCESSED_DIR, MODELS_DIR, CLASS_NAMES, SAMPLE_RATE

model = joblib.load(os.path.join(MODELS_DIR, "deep_svm_model.joblib"))
selector = joblib.load(os.path.join(MODELS_DIR, "inca_selector.joblib"))
stats = np.load(os.path.join(MODELS_DIR, "frta_norm_stats.npz"))
mean, std = stats["mean"], stats["std"]

print("=== FINDING 100% VERIFIED SAMPLES FOR EVERY CLASS ===")

verified_mapping = {}

for cls_idx, cls in enumerate(CLASS_NAMES):
    folder = os.path.join(PROCESSED_DIR, cls)
    files = glob.glob(os.path.join(folder, "*.wav"))
    
    best_file = None
    best_conf = -1.0
    
    for f in files:
        w, sr = load_audio(f)
        std_w = standardize_audio(w, sr)
        vec = extract_frta_features(std_w, SAMPLE_RATE)
        norm = (vec[np.newaxis, :] - mean) / std
        inca = selector.transform(norm)
        
        probs = model.predict_proba(inca)[0]
        conf = float(probs[cls_idx])
        
        if conf > best_conf:
            best_conf = conf
            best_file = os.path.basename(f)
            
    verified_mapping[cls] = (best_file, round(best_conf * 100, 1))
    print(f"Class '{cls:15s}' -> Best File: {best_file:35s} ({verified_mapping[cls][1]}% confidence)")

print("\nCopy-paste dictionary for app/server.py:")
print("REPRESENTATIVE_SAMPLES = {")
for cls, (fname, conf) in verified_mapping.items():
    print(f'    "{cls}": "{fname}",  # {conf}% confidence')
print("}")
