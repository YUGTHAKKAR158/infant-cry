"""
Step (Phase 4): convert the trained Keras model into TensorFlow Lite format,
in two versions:

  1. Float32 TFLite - same numeric precision as the original Keras model,
     just repackaged into the lighter-weight TFLite format. Useful as a
     baseline to isolate "did the FORMAT change hurt accuracy" from
     "did QUANTIZATION hurt accuracy" in the comparison done in
     evaluate_tflite.py.

  2. INT8 quantized TFLite - every weight AND activation is converted from
     32-bit floating point to 8-bit integers. This is the format required by
     TensorFlow Lite Micro on a microcontroller like the ESP32, which has no
     hardware floating-point unit fast enough for real-time float32 inference
     and only a few hundred KB of RAM. Quantization typically shrinks the
     model ~4x and speeds up inference, at some (to be measured, not assumed)
     accuracy cost.

Full-integer quantization needs a "representative dataset": a handful of
real input examples the converter uses to figure out the right scale/offset
for turning floats into int8 without losing too much information. We reuse
actual (normalized) training examples for this - not validation/test, to
avoid any influence from data the model shouldn't be tuned around.

Run with:
    .venv\\Scripts\\python.exe -m src.model.convert_tflite
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.dataset_loader import list_dataset_files, extract_features_for_files
from src.data.split_dataset import stratified_split
from src.utils.config import MODELS_DIR

NUM_REPRESENTATIVE_SAMPLES = 622  # use the full (unaugmented) training split for calibration


def load_normalized_train_samples():
    all_files = list_dataset_files()
    train_files, _, _ = stratified_split(all_files)

    stats = np.load(os.path.join(MODELS_DIR, "feature_norm_stats.npz"))
    mean, std = float(stats["mean"]), float(stats["std"])

    rng = np.random.default_rng(seed=42)
    sample_files = list(rng.choice(len(train_files), size=min(NUM_REPRESENTATIVE_SAMPLES, len(train_files)), replace=False))
    sampled = [train_files[i] for i in sample_files]

    X, _ = extract_features_for_files(sampled)
    X = (X - mean) / std
    return X.astype(np.float32)


def convert_float32(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path


def convert_dynamic_range_int8(model, out_path):
    """
    "Dynamic range" quantization: only the WEIGHTS are converted to int8;
    activations are computed in float32 at runtime and requantized on the
    fly. No representative dataset needed (there's nothing to calibrate
    activation ranges for ahead of time). Usually a smaller accuracy hit than
    full-integer quantization, at a smaller (not zero) size/speed win, and
    still needs float32 arithmetic support at runtime - meaning it is NOT
    the format TensorFlow Lite Micro expects for a microcontroller with no
    hardware floating-point unit. Included here purely as a diagnostic
    middle point in the size-vs-accuracy comparison, not as a deployment
    candidate for the ESP32 itself.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path


def convert_int8(model, representative_samples, out_path):
    def representative_dataset_gen():
        for i in range(len(representative_samples)):
            yield [representative_samples[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path


def main():
    model_path = os.path.join(MODELS_DIR, "baby_cry_model.keras")
    model = keras.models.load_model(model_path)

    print("Loading representative samples from training data (for INT8 calibration)...")
    representative_samples = load_normalized_train_samples()
    print(f"Using {len(representative_samples)} representative samples")

    float32_path = os.path.join(MODELS_DIR, "baby_cry_model_float32.tflite")
    dynamic_int8_path = os.path.join(MODELS_DIR, "baby_cry_model_dynamic_int8.tflite")
    int8_path = os.path.join(MODELS_DIR, "baby_cry_model_int8.tflite")

    print("Converting to float32 TFLite...")
    convert_float32(model, float32_path)

    print("Converting to dynamic-range INT8 TFLite (weights only, diagnostic)...")
    convert_dynamic_range_int8(model, dynamic_int8_path)

    print("Converting to full-integer INT8 quantized TFLite...")
    convert_int8(model, representative_samples, int8_path)

    keras_size = os.path.getsize(model_path)
    float32_size = os.path.getsize(float32_path)
    dynamic_int8_size = os.path.getsize(dynamic_int8_path)
    int8_size = os.path.getsize(int8_path)

    print(f"\nKeras (.keras) model size:        {keras_size / 1024:.1f} KB")
    print(f"TFLite float32 model size:        {float32_size / 1024:.1f} KB")
    print(f"TFLite dynamic-range INT8 size:   {dynamic_int8_size / 1024:.1f} KB")
    print(f"TFLite full-integer INT8 size:    {int8_size / 1024:.1f} KB  "
          f"({keras_size / int8_size:.1f}x smaller than .keras)")
    print(f"\nSaved: {float32_path}")
    print(f"Saved: {dynamic_int8_path}")
    print(f"Saved: {int8_path}")


if __name__ == "__main__":
    main()
