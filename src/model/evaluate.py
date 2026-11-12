"""
Step 8: measure how well the trained model actually performs on data it has
never seen (the held-out test split saved by train.py).

Run with:
    .venv\\Scripts\\python.exe -m src.model.evaluate

Reports accuracy, precision, recall, F1, and a confusion matrix - not just
accuracy, because for a baby-cry detector a false negative (missing a real
cry) is a very different kind of mistake than a false positive (a false
alarm), and accuracy alone hides that distinction.
"""
import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report,
)
from tensorflow import keras

from src.data.dataset_loader import extract_features_for_files
from src.data.source_subgroup import get_cry_native_sample_rates, is_minority_subgroup_file
from src.utils.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR


def load_test_split():
    path = os.path.join(MODELS_DIR, "test_split.json")
    with open(path) as f:
        entries = json.load(f)
    return [(e["path"], e["label"]) for e in entries]


def load_norm_stats():
    stats = np.load(os.path.join(MODELS_DIR, "feature_norm_stats.npz"))
    return float(stats["mean"]), float(stats["std"])


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def subgroup_bias_check(test_files, model, mean, std):
    """
    Report accuracy separately for the two cry recording sources identified
    in docs/dataset.md, instead of trusting one overall accuracy number that
    could be dominated by the majority source. This is what caught the
    original generalization gap (45.5% vs 97.3%) before mitigation.
    """
    native_rates = get_cry_native_sample_rates()
    baby_cry_index = CLASS_NAMES.index("baby_cry")

    dominant, minority, non_cry = [], [], []
    for path, label in test_files:
        if label == baby_cry_index:
            (minority if is_minority_subgroup_file(path, native_rates) else dominant).append((path, label))
        else:
            non_cry.append((path, label))

    print("\n=== Source-subgroup bias check (see docs/dataset.md) ===")
    results = {}
    for name, entries in [("cry_dominant_source", dominant), ("cry_minority_source", minority), ("non_cry", non_cry)]:
        if not entries:
            print(f"{name}: no examples in test split")
            continue
        X, y = extract_features_for_files(entries)
        X = (X - mean) / std
        preds = np.argmax(model.predict(X, verbose=0), axis=1)
        acc = float(np.mean(preds == np.array(y)))
        print(f"{name}: n={len(entries)}  accuracy={acc:.3f}")
        results[name] = {"n": len(entries), "accuracy": acc}
    return results


def main():
    model_path = os.path.join(MODELS_DIR, "baby_cry_model.keras")
    model = keras.models.load_model(model_path)

    test_files = load_test_split()
    mean, std = load_norm_stats()

    print(f"Evaluating on {len(test_files)} held-out test files (never used in training or validation)")

    X_test, y_test = extract_features_for_files(test_files)
    X_test = (X_test - mean) / std

    start = time.perf_counter()
    y_probs = model.predict(X_test, verbose=0)
    elapsed = time.perf_counter() - start
    avg_inference_ms = (elapsed / len(X_test)) * 1000

    y_pred = np.argmax(y_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    num_cls = len(CLASS_NAMES)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(num_cls)), zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_cls)))

    print("\n=== Test set results ===")
    print(f"Accuracy: {accuracy:.4f}")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:15s}  precision={precision[i]:.3f}  recall={recall[i]:.3f}  f1={f1[i]:.3f}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted), order {CLASS_NAMES}:")
    print(cm)
    print(f"\nAverage inference time per clip: {avg_inference_ms:.2f} ms")

    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    plot_confusion_matrix(cm, os.path.join(plots_dir, "confusion_matrix.png"))

    report = {
        "test_set_size": len(test_files),
        "accuracy": float(accuracy),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_label_order": CLASS_NAMES,
        "avg_inference_time_ms_laptop_cpu": float(avg_inference_ms),
        "num_model_parameters": int(model.count_params()),
        "model_file_size_bytes": os.path.getsize(model_path),
    }
    with open(os.path.join(metrics_dir, "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nModel parameters: {report['num_model_parameters']}")
    print(f"Model file size: {report['model_file_size_bytes'] / 1024:.1f} KB")
    print(f"\nSaved confusion matrix plot to {os.path.join(plots_dir, 'confusion_matrix.png')}")
    print(f"Saved full metrics to {os.path.join(metrics_dir, 'test_metrics.json')}")


if __name__ == "__main__":
    main()
