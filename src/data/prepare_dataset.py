"""
Step 3: convert the raw audio datasets into our standardized 6-class layout.

    data/raw/donateacry-corpus/**/*         (mapped by -hu, -bp, -bu, -dc, -ti tags)
    data/raw/infant-cry-dataset/Dataset/not_cry/*  (non_baby_cry)

        becomes

    data/processed/hungry/*.wav
    data/processed/belly_pain/*.wav
    data/processed/burping/*.wav
    data/processed/discomfort/*.wav
    data/processed/tired/*.wav
    data/processed/non_baby_cry/*.wav       (all standardized to 16kHz PCM_16 mono 2.0s)

Run with:
    .venv\Scripts\python.exe -m src.data.prepare_dataset
"""
import os
import glob
import soundfile as sf
from tqdm import tqdm

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.utils.config import (
    PROJECT_ROOT,
    RAW_DATASET_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    TARGET_REASON_MAP,
    CLASS_NAMES,
)

DONATE_CRY_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "donateacry-corpus")
NOT_CRY_DIR = os.path.join(RAW_DATASET_DIR, "not_cry")


def collect_raw_files():
    """Find all raw audio files and map them to target class names."""
    mapped_files = {cls: [] for cls in CLASS_NAMES}

    # 1. Scan donateacry-corpus by filename tags (-hu, -bp, -bu, -dc, -ti)
    if os.path.exists(DONATE_CRY_DIR):
        for root, _, files in os.walk(DONATE_CRY_DIR):
            for f in files:
                if f.endswith((".wav", ".caf", ".ogg", ".3gp", ".m4a")):
                    full_path = os.path.join(root, f)
                    name_no_ext = os.path.splitext(f)[0]
                    parts = name_no_ext.split("-")
                    if len(parts) >= 2:
                        reason = parts[-1].lower()
                        if reason in TARGET_REASON_MAP:
                            target_cls = TARGET_REASON_MAP[reason]
                            mapped_files[target_cls].append(full_path)

    # 2. Scan not_cry files
    if os.path.exists(NOT_CRY_DIR):
        for f in glob.glob(os.path.join(NOT_CRY_DIR, "*")):
            if os.path.isfile(f):
                mapped_files["non_baby_cry"].append(f)

    return mapped_files


def prepare_dataset():
    mapped_files = collect_raw_files()

    for class_name, file_paths in mapped_files.items():
        out_folder = os.path.join(PROCESSED_DIR, class_name)
        os.makedirs(out_folder, exist_ok=True)

        print(f"\nProcessing class '{class_name}' ({len(file_paths)} files)")

        num_ok, num_failed = 0, 0
        for i, src_path in enumerate(tqdm(file_paths, unit="file")):
            try:
                waveform, original_sr = load_audio(src_path)
                standardized = standardize_audio(waveform, original_sr)

                base_name = os.path.splitext(os.path.basename(src_path))[0]
                out_name = f"{class_name}_{i:04d}_{base_name[:12]}.wav"
                out_path = os.path.join(out_folder, out_name)

                sf.write(out_path, standardized, SAMPLE_RATE, subtype="PCM_16")
                num_ok += 1
            except Exception as e:
                num_failed += 1
                print(f"  FAILED: {os.path.basename(src_path)} -> {e}")

        print(f"  done: {num_ok} written, {num_failed} failed")


if __name__ == "__main__":
    prepare_dataset()
