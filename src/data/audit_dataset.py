"""
Dataset Audit & Cleaning Script for Infant Cry Corpus

Performs:
1. Signal quality audit (RMS, SNR, pitch stability) across all processed audio files.
2. 5-Fold Cross-Validation Out-Of-Fold (OOF) label disagreement detection to identify crowdsourced mislabeled files.
3. Generation of a cleaned dataset folder `data/cleaned/` with noisy/corrupt files pruned.
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.fractal_features import extract_frta_features, INCAFeatureSelector
from src.model.deep_svm import DeepSVM
from src.utils.config import (
    PROJECT_ROOT,
    PROCESSED_DIR,
    CLASS_NAMES,
    SAMPLE_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
)

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")

def audit_file_quality(file_path):
    waveform, sr = load_audio(file_path)
    
    # 1. Energy metrics
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    peak = float(np.max(np.abs(waveform)))
    
    # 2. Spectral metrics
    stft = np.abs(librosa.stft(waveform, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    cry_band = (freqs >= 250) & (freqs <= 2500)
    total_energy = float(np.sum(stft ** 2)) + 1e-9
    cry_energy = float(np.sum(stft[cry_band, :] ** 2))
    cry_ratio = cry_energy / total_energy
    
    # 3. Pitch presence
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr, fmin=200, fmax=800)
    valid_pitch = pitches[magnitudes > np.median(magnitudes)]
    has_pitch = len(valid_pitch) > 10
    
    # Determine quality flag
    is_silent = rms < 0.01 or peak < 0.02
    is_poor_quality = is_silent or (not has_pitch and cry_ratio < 0.25)
    
    return {
        "rms": rms,
        "peak": peak,
        "cry_ratio": cry_ratio,
        "has_pitch": has_pitch,
        "is_poor_quality": is_poor_quality
    }

def run_dataset_audit():
    print("=== Phase 1: Scanning & Extracting Features for Dataset Audit ===")
    file_list = []
    y_list = []
    
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(PROCESSED_DIR, cls)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        for f in files:
            file_list.append((f, cls, cls_idx))
            y_list.append(cls_idx)
            
    print(f"Total dataset files scanned: {len(file_list)}")
    
    X_features = []
    quality_stats = []
    
    for path, cls, cls_idx in tqdm(file_list, desc="Auditing audio files"):
        stats = audit_file_quality(path)
        waveform, sr = load_audio(path)
        vec = extract_frta_features(waveform, sr)
        X_features.append(vec)
        quality_stats.append(stats)
        
    X_mat = np.array(X_features, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    
    print("\n=== Phase 2: 5-Fold Out-Of-Fold (OOF) Label Noise Audit ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(file_list), len(CLASS_NAMES)))
    
    # Normalize
    mean = X_mat.mean(axis=0, keepdims=True)
    std = X_mat.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_mat - mean) / std
    
    from imblearn.over_sampling import RandomOverSampler

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_norm, y_arr)):
        X_tr, y_tr = X_norm[train_idx], y_arr[train_idx]
        X_va, y_va = X_norm[val_idx], y_arr[val_idx]
        
        ros = RandomOverSampler(random_state=42)
        X_tr_res, y_tr_res = ros.fit_resample(X_tr, y_tr)
        
        selector = INCAFeatureSelector(n_components=32, random_state=42)
        X_tr_inca = selector.fit_transform(X_tr_res, y_tr_res)
        X_va_inca = selector.transform(X_va)
        
        model = DeepSVM(n_estimators=15, C=4.0, gamma="scale", random_state=42)
        model.fit(X_tr_inca, y_tr_res)
        
        probs = model.predict_proba(X_va_inca)
        oof_probs[val_idx] = probs

    oof_preds = np.argmax(oof_probs, axis=1)
    raw_acc = accuracy_score(y_arr, oof_preds)
    print(f"Raw 5-Fold Cross-Validation Accuracy before cleaning: {raw_acc * 100:.2f}%")
    
    # Identify mislabeled / low quality files
    audit_results = []
    clean_files = []
    flagged_mislabeled = 0
    flagged_poor_quality = 0
    
    for i, (path, orig_cls, orig_idx) in enumerate(file_list):
        q = quality_stats[i]
        pred_idx = oof_preds[i]
        pred_cls = CLASS_NAMES[pred_idx]
        conf = float(oof_probs[i, pred_idx])
        orig_conf = float(oof_probs[i, orig_idx])
        
        # Only prune mislabeled files for majority classes (hungry/non_baby_cry) when confidence > 0.90
        # For minority classes (burping, discomfort, tired, belly_pain), retain all non-corrupt audio clips
        is_majority = orig_cls in ["hungry", "non_baby_cry"]
        is_mislabeled = is_majority and (pred_idx != orig_idx) and (conf > 0.90) and (orig_conf < 0.05)
        is_poor = q["is_poor_quality"]
        
        if is_mislabeled:
            flagged_mislabeled += 1
        if is_poor:
            flagged_poor_quality += 1
            
        keep = not (is_mislabeled or is_poor)
        if keep:
            clean_files.append((path, orig_cls))
            
        audit_results.append({
            "file_name": os.path.basename(path),
            "original_class": orig_cls,
            "predicted_class": pred_cls,
            "confidence": round(conf, 3),
            "original_class_confidence": round(orig_conf, 3),
            "is_poor_quality": bool(is_poor),
            "is_mislabeled": bool(is_mislabeled),
            "retained_in_cleaned_dataset": bool(keep)
        })

    print(f"\nAudit Summary:")
    print(f"  - Total files scanned: {len(file_list)}")
    print(f"  - Poor quality / silent clips flagged: {flagged_poor_quality}")
    print(f"  - Severe label noise / mislabeled clips flagged: {flagged_mislabeled}")
    print(f"  - Clean, high-confidence files retained: {len(clean_files)} ({len(clean_files)/len(file_list)*100:.1f}%)")
    
    # Save audit report JSON
    audit_path = os.path.join(OUTPUTS_DIR, "metrics", "dataset_audit_report.json")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Saved full audit report to: {audit_path}")
    
    # Populate cleaned dataset directory
    print("\n=== Phase 3: Building Cleaned Dataset Corpus ===")
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(CLEANED_DIR, cls), exist_ok=True)
        
    for path, cls in clean_files:
        dest = os.path.join(CLEANED_DIR, cls, os.path.basename(path))
        if not os.path.exists(dest):
            with open(path, "rb") as sf, open(dest, "wb") as df:
                df.write(sf.read())
                
    print(f"Successfully constructed cleaned dataset corpus at: {CLEANED_DIR}")

if __name__ == "__main__":
    run_dataset_audit()
