"""
Central place for every tunable constant used across the project.

Why this file exists: values like sample rate or clip duration get used in
multiple places (preprocessing, training, prediction). If they were hardcoded
in each file, changing one for an experiment would risk silently breaking
consistency between training and inference - which is the single most
important requirement for this project (see docs/dataset.md).
"""

# --- Audio standardization (decided in docs/dataset.md) ---
SAMPLE_RATE = 16000          # Hz - all audio is resampled to this before feature extraction
AUDIO_DURATION = 2.0         # seconds - all clips are trimmed/padded to this fixed length
NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)

LOWPASS_CUTOFF_HZ = 4000     # low-pass filter applied to ALL clips to remove the
                             # sample-rate/format bias described in docs/dataset.md

# --- Feature extraction (Updated per Rezaee et al. 2024: 25ms windowing, 40% overlap) ---
NUM_MFCC = 40
FRAME_LENGTH_MS = 25  # 25 ms frame length
OVERLAP_PERCENT = 40  # 40% frame overlap (60% hop)
WIN_LENGTH = int(SAMPLE_RATE * (FRAME_LENGTH_MS / 1000.0))  # 400 samples
HOP_LENGTH = int(WIN_LENGTH * (1.0 - OVERLAP_PERCENT / 100.0))  # 240 samples
N_FFT = 512  # Next power of 2 >= WIN_LENGTH

# --- Classes ---
CLASS_NAMES = ["hungry", "belly_pain", "burping", "discomfort", "tired", "non_baby_cry"]
NUM_CLASSES = len(CLASS_NAMES)

TARGET_REASON_MAP = {
    "hu": "hungry",
    "bp": "belly_pain",
    "bu": "burping",
    "dc": "discomfort",
    "ti": "tired",
}

# --- Training (used later, kept here so nothing is scattered) ---
CONFIDENCE_THRESHOLD = 0.80
BATCH_SIZE = 32
EPOCHS = 30

# --- Paths ---
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "infant-cry-dataset", "Dataset")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
