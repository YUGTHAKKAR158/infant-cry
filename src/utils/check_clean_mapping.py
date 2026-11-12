import os
import glob
import numpy as np
import joblib
from src.audio.load_audio import load_audio
from src.audio.fractal_features import extract_frta_features
from src.utils.config import PROJECT_ROOT, CLASS_NAMES, MODELS_DIR

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
model = joblib.load(os.path.join(MODELS_DIR, "deep_svm_model.joblib"))
selector = joblib.load(os.path.join(MODELS_DIR, "inca_selector.joblib"))
stats = np.load(os.path.join(MODELS_DIR, "frta_norm_stats.npz"))
mean, std = stats["mean"], stats["std"]

mapping = {}
for cls in CLASS_NAMES:
    folder = os.path.join(CLEANED_DIR, cls)
    files = glob.glob(os.path.join(folder, "*.wav"))
    best_file = None
    best_conf = -1.0
    for f in files:
        w, sr = load_audio(f)
        vec = extract_frta_features(w, sr)
        norm = (vec[np.newaxis, :] - mean) / std
        inca = selector.transform(norm)
        
        # Linear/RBF ensemble raw decision
        probs = model.predict_proba(inca)[0]
        cls_idx = CLASS_NAMES.index(cls)
        conf = probs[cls_idx]
        if conf > best_conf:
            best_conf = conf
            best_file = os.path.basename(f)
            
    mapping[cls] = (best_file, round(best_conf * 100, 1))
    print(f"Class '{cls:15s}' -> Best File: {best_file} ({best_conf*100:.1f}%)")
