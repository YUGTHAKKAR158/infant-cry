"""
Step 7 (part 4): tie it all together - load data, split, extract features,
train the CNN, save everything needed to reproduce inference later.

Run with:
    .venv\\Scripts\\python.exe -m src.model.train
"""
import os
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

# Seeds Python's random module, NumPy, and TensorFlow all at once. Without
# this, Keras's weight initialization is different every run - usually
# harmless, but on a small/imbalanced dataset it occasionally produces a
# degenerate model that collapses to predicting one class for everything
# (observed once during development). Fixing the seed makes training
# reproducible and avoids re-rolling this risk on every run.
keras.utils.set_random_seed(42)

from src.audio.load_audio import load_audio
from src.audio.augmentation import augment_waveform, spec_augment
from src.audio.features import extract_mfcc
from src.data.dataset_loader import list_dataset_files, extract_features_for_files
from src.data.split_dataset import stratified_split
from src.data.source_subgroup import get_cry_native_sample_rates, is_minority_subgroup_file
from src.model.model import build_model
from src.utils.config import BATCH_SIZE, EPOCHS, CLASS_NAMES, SAMPLE_RATE, MODELS_DIR, OUTPUTS_DIR

# Every training file gets this many augmented copies (general robustness -
# noise/gain/time-shift + SpecAugment masking). The minority cry subgroup
# (see docs/dataset.md bias check) gets this many PLUS extra copies, since it
# was shown to need more help generalizing.
GENERAL_AUGMENT_COPIES = 1
MINORITY_EXTRA_AUGMENT_COPIES = 4


def normalize_features(X_train, X_val, X_test):
    """
    Z-score normalize MFCC features: subtract mean, divide by std.

    Why needed: raw MFCC values can have very different scales across
    coefficients. Normalizing helps the network train faster and more
    stably. Mean/std are computed ONLY from the training set (using
    validation/test statistics would leak information about data the model
    is supposed to have never seen) and then applied to all three splits.
    The same mean/std must be reused at prediction time - they are saved to
    disk alongside the model for exactly that reason.
    """
    mean = X_train.mean()
    std = X_train.std() + 1e-8

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_augmented_examples(train_files):
    """
    Create extra training examples for training files: waveform-level
    augmentation followed by SpecAugment masking on the resulting MFCC.
    Minority classes receive additional copies to help balance training.
    """
    non_cry_index = CLASS_NAMES.index("non_baby_cry")
    rng = np.random.default_rng(seed=42)

    # Count occurrences per class in train_files to boost rare classes
    class_counts = Counter(l for _, l in train_files)

    extra_features, extra_labels = [], []
    for path, label in train_files:
        if label == non_cry_index:
            copies = 1
        elif class_counts.get(label, 0) < 50:
            copies = 4
        elif class_counts.get(label, 0) < 150:
            copies = 2
        else:
            copies = 1

        waveform, sr = load_audio(path)
        for _ in range(copies):
            augmented_waveform = augment_waveform(waveform, sr, rng)
            mfcc = extract_mfcc(augmented_waveform, sr)
            mfcc = spec_augment(mfcc, rng)
            extra_features.append(mfcc)
            extra_labels.append(label)

    print(f"Augmented copies generated: {len(extra_features)} total")

    if not extra_features:
        return None, None
    return np.stack(extra_features).astype(np.float32), np.array(extra_labels, dtype=np.int64)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Listing dataset files...")
    all_files = list_dataset_files()
    train_files, val_files, test_files = stratified_split(all_files)

    print(f"Dataset sizes -> train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")

    print("Extracting MFCC features (train)...")
    X_train, y_train = extract_features_for_files(train_files)
    print("Extracting MFCC features (val)...")
    X_val, y_val = extract_features_for_files(val_files)
    print("Extracting MFCC features (test)...")
    X_test, y_test = extract_features_for_files(test_files)

    print(f"Feature shape per example: {X_train.shape[1:]}")

    print("Building augmented training examples (train split only)...")
    extra_X, extra_y = build_augmented_examples(train_files)
    if extra_X is not None:
        X_train = np.concatenate([X_train, extra_X], axis=0)
        y_train = np.concatenate([y_train, extra_y], axis=0)
        shuffle_idx = np.random.default_rng(seed=42).permutation(len(X_train))
        X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
        print(f"Train set size after augmentation: {len(X_train)}")

    X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)

    # Class weights: cry(565)/not_cry(324) is imbalanced (~1.7:1) - without
    # this, the model could get decent accuracy by leaning toward the
    # majority class instead of genuinely discriminating both.
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}
    print(f"Class weights (to counter {CLASS_NAMES} imbalance): {class_weights}")

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = os.path.join(MODELS_DIR, "baby_cry_model.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history, os.path.join(plots_dir, "training_curves.png"))

    # Save normalization stats so predict.py/evaluate.py use the EXACT same
    # transform as training - required for train/inference consistency.
    np.savez(os.path.join(MODELS_DIR, "feature_norm_stats.npz"), mean=mean, std=std)

    # Save the held-out test split file list so evaluate.py evaluates on
    # data the model never saw during training or validation.
    with open(os.path.join(MODELS_DIR, "test_split.json"), "w") as f:
        json.dump([{"path": p, "label": l} for p, l in test_files], f, indent=2)

    print(f"\nSaved best model to: {best_model_path}")
    print("Saved normalization stats to: models/feature_norm_stats.npz")
    print("Saved test split list to: models/test_split.json")
    print(f"Saved training curves to: {os.path.join(plots_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()
