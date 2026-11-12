"""
Step 9: command-line prediction on a single new/unseen WAV file.

Run with:
    .venv\\Scripts\\python.exe -m src.model.predict path\\to\\audio.wav

IMPORTANT: this script calls the exact same `standardize_audio()` and
`extract_mfcc()` functions used by src/data/prepare_dataset.py during
training (via load_audio -> standardize_audio -> extract_mfcc), and the same
saved normalization mean/std from training. This is deliberate - training
and inference must see identically-processed input, or the model's learned
weights won't mean what they're supposed to mean on new audio (see
docs/dataset.md).
"""
import os
import sys
import json

import numpy as np
import joblib
from tensorflow import keras

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.features import extract_mfcc
from src.audio.fractal_features import extract_frta_features
from src.utils.config import SAMPLE_RATE, CLASS_NAMES, CONFIDENCE_THRESHOLD, MODELS_DIR


def predict_file_deep_svm(file_path, model, selector, mean, std):
    waveform, original_sr = load_audio(file_path)
    duration = len(waveform) / original_sr

    print("Loading audio...")
    print(f"Sample rate: {original_sr} Hz")
    print(f"Duration: {duration:.2f} seconds")

    print("Extracting 2D FRTA Fractal Features...")
    standardized = standardize_audio(waveform, original_sr)
    frta_vec = extract_frta_features(standardized, SAMPLE_RATE)
    frta_norm = (frta_vec[np.newaxis, :] - mean) / std

    print("Applying iNCA Feature Selection...")
    frta_inca = selector.transform(frta_norm)

    print("Running DeepSVM Model...")
    probs = model.predict_proba(frta_inca)[0]
    predicted_index = int(np.argmax(probs))
    confidence = float(probs[predicted_index])
    predicted_label = CLASS_NAMES[predicted_index]

    return predicted_label, confidence, probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.model.predict path/to/audio.wav [--cnn]")
        sys.exit(1)

    file_path = sys.argv[1]
    use_cnn = "--cnn" in sys.argv

    if not os.path.isfile(file_path):
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    if use_cnn:
        model_path = os.path.join(MODELS_DIR, "baby_cry_model.keras")
        model = keras.models.load_model(model_path)
        stats = np.load(os.path.join(MODELS_DIR, "feature_norm_stats.npz"))
        mean, std = float(stats["mean"]), float(stats["std"])

        waveform, original_sr = load_audio(file_path)
        standardized = standardize_audio(waveform, original_sr)
        mfcc = extract_mfcc(standardized, SAMPLE_RATE)
        mfcc = (mfcc - mean) / std
        probs = model.predict(mfcc[np.newaxis, ...], verbose=0)[0]
    else:
        model_path = os.path.join(MODELS_DIR, "deep_svm_model.joblib")
        selector_path = os.path.join(MODELS_DIR, "inca_selector.joblib")
        stats_path = os.path.join(MODELS_DIR, "frta_norm_stats.npz")

        model = joblib.load(model_path)
        selector = joblib.load(selector_path)
        stats = np.load(stats_path)
        mean, std = stats["mean"], stats["std"]

        predicted_label, confidence, probs = predict_file_deep_svm(file_path, model, selector, mean, std)

    predicted_index = int(np.argmax(probs))
    confidence = float(probs[predicted_index])
    predicted_label = CLASS_NAMES[predicted_index]

    print()
    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"Prediction: {predicted_label.upper()}")
    else:
        print(f"Prediction: {predicted_label.upper()} (confidence: {confidence * 100:.1f}%)")

    print("All class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:15s}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
