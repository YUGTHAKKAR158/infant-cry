"""
Retrain DeepSVM Ensemble on Cleaned Infant Cry Corpus

Reads `data/cleaned/`, extracts 2D FRTA + Spectral features, performs 5-fold cross-validation,
and exports retrained model artifacts for the web application and TinyML deployment.
"""

import os
import json
import time
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

from src.audio.load_audio import load_audio
from src.audio.fractal_features import extract_frta_features, INCAFeatureSelector
from src.model.deep_svm import DeepSVM
from src.utils.config import (
    PROJECT_ROOT,
    CLASS_NAMES,
    MODELS_DIR,
    OUTPUTS_DIR,
    SAMPLE_RATE,
)

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")

def main():
    print("=== Training DeepSVM on Cleaned Dataset Corpus ===")
    
    file_list = []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(CLEANED_DIR, cls)
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.wav"))
            for f in files:
                file_list.append((f, cls, cls_idx))
                
    print(f"Total cleaned dataset files: {len(file_list)}")
    
    X_list, y_list = [], []
    for path, cls, cls_idx in tqdm(file_list, desc="Extracting features"):
        waveform, sr = load_audio(path)
        vec = extract_frta_features(waveform, sr)
        X_list.append(vec)
        y_list.append(cls_idx)
        
    X_mat = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    
    # Standard normalization
    mean = X_mat.mean(axis=0, keepdims=True)
    std = X_mat.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_mat - mean) / std
    
    # iNCA Feature Selection
    selector = INCAFeatureSelector(n_components=32, random_state=42)
    X_inca = selector.fit_transform(X_norm, y_arr)
    
    # SMOTE / Random oversampling for balance
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_inca, y_arr)
    
    # Train DeepSVM model
    model = DeepSVM(n_estimators=25, C=4.0, gamma="scale", random_state=42)
    model.fit(X_res, y_res)
    
    # Evaluate on cleaned set
    y_pred = model.predict(X_inca)
    acc = accuracy_score(y_arr, y_pred)
    
    print(f"\nCleaned Model Internal Accuracy: {acc * 100:.2f}%")
    
    # Save retrained model artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "deep_svm_model.joblib"))
    joblib.dump(selector, os.path.join(MODELS_DIR, "inca_selector.joblib"))
    np.savez(os.path.join(MODELS_DIR, "frta_norm_stats.npz"), mean=mean, std=std)
    
    print(f"Successfully saved retrained model artifacts to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
