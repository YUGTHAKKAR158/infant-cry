"""
Train and Evaluate FRTA + iNCA + DeepSVM Pipeline

Complete implementation of the 5-Class / 6-Class Infant Cry Classification
system based on Rezaee et al. (ACM 2024).

Run with:
    .venv\Scripts\python.exe -m src.model.train_evaluate_deep_svm
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

from src.audio.load_audio import load_audio
from src.audio.fractal_features import extract_frta_features, INCAFeatureSelector
from src.data.dataset_loader import list_dataset_files
from src.data.split_dataset import stratified_split
from src.model.deep_svm import DeepSVM
from src.utils.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR


def extract_frta_for_dataset(file_label_pairs):
    X_list, y_list = [], []
    for path, label in tqdm(file_label_pairs, unit="file"):
        try:
            waveform, sr = load_audio(path)
            vec = extract_frta_features(waveform, sr)
            X_list.append(vec)
            y_list.append(label)
        except Exception as e:
            print(f"Failed {path}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics")
    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("--- Phase 1: Loading & Splitting Dataset ---")
    all_files = list_dataset_files()
    train_files, val_files, test_files = stratified_split(all_files)
    print(f"Dataset sizes -> train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    print("\n--- Phase 2: Extracting 2D FRTA Fractal & Spectral Features ---")
    X_train_raw, y_train = extract_frta_for_dataset(train_files)
    X_val_raw, y_val = extract_frta_for_dataset(val_files)
    X_test_raw, y_test = extract_frta_for_dataset(test_files)

    print(f"Extracted FRTA feature shape per sample: {X_train_raw.shape[1]}")

    # Standard normalization
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True) + 1e-8

    X_train_norm = (X_train_raw - mean) / std
    X_val_norm = (X_val_raw - mean) / std
    X_test_norm = (X_test_raw - mean) / std

    print("\n--- Phase 3: iNCA Feature Selection ---")
    nca_components = min(32, X_train_norm.shape[1])
    selector = INCAFeatureSelector(n_components=nca_components, random_state=42)

    X_train_inca = selector.fit_transform(X_train_norm, y_train)
    X_val_inca = selector.transform(X_val_norm)
    X_test_inca = selector.transform(X_test_norm)

    print(f"Selected feature dimension after iNCA: {X_train_inca.shape[1]}")

    print("\n--- Phase 4: Training DeepSVM Ensemble Model (with SMOTE Class Balancing) ---")
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_inca, y_train)
    print(f"Resampled training set shape: {X_train_resampled.shape}")

    model = DeepSVM(n_estimators=20, C=4.0, gamma="scale", random_state=42)
    start_train = time.perf_counter()
    model.fit(X_train_resampled, y_train_resampled)
    train_time = time.perf_counter() - start_train
    print(f"DeepSVM training completed in {train_time:.2f} seconds.")

    print("\n--- Phase 5: Test Set Evaluation ---")
    start_eval = time.perf_counter()
    y_pred = model.predict(X_test_inca)
    eval_time = time.perf_counter() - start_eval

    accuracy = accuracy_score(y_test, y_pred)
    num_cls = len(CLASS_NAMES)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(num_cls)), zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_cls)))

    print(f"\n================ DeepSVM Evaluation Results ================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Inference Time: {(eval_time / len(X_test_inca)) * 1000:.2f} ms/clip")
    print("\nPer-Class Breakdown:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s} -> Precision: {precision[i]*100:5.1f}% | Recall: {recall[i]*100:5.1f}% | F1: {f1[i]:.3f}")

    print(f"\nConfusion Matrix:\n{cm}")

    report = {
        "model_name": "DeepSVM (FRTA + iNCA)",
        "accuracy": float(accuracy),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(num_cls)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": CLASS_NAMES,
    }

    report_path = os.path.join(metrics_dir, "deep_svm_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save trained DeepSVM & selector artifacts
    joblib.dump(model, os.path.join(MODELS_DIR, "deep_svm_model.joblib"))
    joblib.dump(selector, os.path.join(MODELS_DIR, "inca_selector.joblib"))
    np.savez(os.path.join(MODELS_DIR, "frta_norm_stats.npz"), mean=mean, std=std)

    print(f"\nSaved metrics report to: {report_path}")
    print(f"Saved trained DeepSVM model to: {os.path.join(MODELS_DIR, 'deep_svm_model.joblib')}")


if __name__ == "__main__":
    main()
