"""
Step (Phase 4, continued): measure whether TFLite conversion and INT8
quantization actually cost us anything, using the SAME held-out test set
and the SAME metrics as src/model/evaluate.py - accuracy, per-class
precision/recall/F1, the source-subgroup bias check, model size, and
per-clip inference latency (still measured on this laptop's CPU, NOT the
ESP32 - the TFLite Micro interpreter on an actual microcontroller will be
slower; this only tells us "did quantization change the model's answers/
size", not "how fast will the ESP32 be").

Run with:
    .venv\\Scripts\\python.exe -m src.model.evaluate_tflite
"""
import os
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.data.dataset_loader import extract_features_for_files
from src.data.source_subgroup import get_cry_native_sample_rates, is_minority_subgroup_file
from src.utils.config import CLASS_NAMES, MODELS_DIR, OUTPUTS_DIR


def load_test_split():
    with open(os.path.join(MODELS_DIR, "test_split.json")) as f:
        entries = json.load(f)
    return [(e["path"], e["label"]) for e in entries]


def load_norm_stats():
    stats = np.load(os.path.join(MODELS_DIR, "feature_norm_stats.npz"))
    return float(stats["mean"]), float(stats["std"])


def run_tflite_inference(tflite_path, X):
    """
    Run every example in X through a .tflite model one at a time (matches
    how a real-time ESP32 pipeline would process one clip at a time, not a
    batch) and return (predicted_labels, avg_inference_ms).

    Handles both float32 and INT8 models transparently by checking the
    interpreter's expected input/output dtype and quantization parameters -
    an INT8 model needs its float input manually quantized before feeding
    it in, and its INT8 output manually dequantized back to probabilities.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    is_quantized_input = input_details["dtype"] == np.int8
    if is_quantized_input:
        input_scale, input_zero_point = input_details["quantization"]
        output_scale, output_zero_point = output_details["quantization"]

    predictions = []
    total_time = 0.0

    for i in range(len(X)):
        sample = X[i:i + 1]

        if is_quantized_input:
            sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        else:
            sample = sample.astype(np.float32)

        interpreter.set_tensor(input_details["index"], sample)

        start = time.perf_counter()
        interpreter.invoke()
        total_time += time.perf_counter() - start

        output = interpreter.get_tensor(output_details["index"])[0]
        if is_quantized_input:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(int(np.argmax(output)))

    avg_ms = (total_time / len(X)) * 1000
    return np.array(predictions), avg_ms


def evaluate_predictions(name, y_true, y_pred, test_files, avg_inference_ms, model_size_bytes):
    accuracy = accuracy_score(y_true, y_pred)
    num_cls = len(CLASS_NAMES)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_cls)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_cls)))

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}  |  Model size: {model_size_bytes / 1024:.2f} KB  |  "
          f"Avg inference: {avg_inference_ms:.3f} ms/clip (laptop CPU)")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:15s}  precision={precision[i]:.3f}  recall={recall[i]:.3f}  f1={f1[i]:.3f}")
    print(f"Confusion matrix {CLASS_NAMES}:\n{cm}")

    return {
        "accuracy": float(accuracy),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "avg_inference_time_ms_laptop_cpu": float(avg_inference_ms),
        "model_file_size_bytes": model_size_bytes,
    }


def main():
    test_files = load_test_split()
    mean, std = load_norm_stats()

    print(f"Evaluating on {len(test_files)} held-out test files")
    X_test, y_test = extract_features_for_files(test_files)
    X_test_normalized = (X_test - mean) / std

    report = {}

    for tag, filename in [("tflite_float32", "baby_cry_model_float32.tflite"),
                           ("tflite_dynamic_int8", "baby_cry_model_dynamic_int8.tflite"),
                           ("tflite_int8", "baby_cry_model_int8.tflite")]:
        path = os.path.join(MODELS_DIR, filename)
        y_pred, avg_ms = run_tflite_inference(path, X_test_normalized)
        size = os.path.getsize(path)
        report[tag] = evaluate_predictions(tag, y_test, y_pred, test_files, avg_ms, size)

    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    out_path = os.path.join(metrics_dir, "tflite_comparison.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
