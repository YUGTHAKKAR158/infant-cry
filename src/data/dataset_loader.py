"""
Step 7 (part 1): find every processed audio file and turn it into
(features, label) arrays the model can train on.

This module deliberately does NOT do the train/val/test split - that's
src/data/split_dataset.py's job - so each file has one responsibility.
"""
import os
import glob

import numpy as np

from src.audio.load_audio import load_audio
from src.audio.features import extract_mfcc
from src.utils.config import PROCESSED_DIR, CLASS_NAMES


def list_dataset_files():
    """
    Return a list of (file_path, label_index) for every file in
    data/processed/<class_name>/, using the order of CLASS_NAMES in
    config.py as the label index mapping (0 = non_baby_cry, 1 = baby_cry).
    """
    entries = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        files = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        entries.extend((f, label_index) for f in files)
    return entries


def extract_features_for_files(file_label_pairs):
    """
    Load + MFCC-extract every file in `file_label_pairs`.

    Returns:
        X: float32 array, shape (num_files, NUM_MFCC, num_frames, 1)
        y: int array, shape (num_files,) - 0 or 1, matching CLASS_NAMES
    """
    features, labels = [], []
    for file_path, label in file_label_pairs:
        waveform, sr = load_audio(file_path)
        mfcc = extract_mfcc(waveform, sr)
        features.append(mfcc)
        labels.append(label)

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y
