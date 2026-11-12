import os
import shutil
from src.utils.config import PROJECT_ROOT, PROCESSED_DIR, CLASS_NAMES

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "app", "static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

VERIFIED_FILES = {
    "hungry": os.path.join(PROCESSED_DIR, "hungry", "hungry_0721_F4A0C710-AC2.wav"),
    "belly_pain": os.path.join(PROCESSED_DIR, "belly_pain", "belly_pain_0032_643D64AD-B71.wav"),
    "burping": os.path.join(PROCESSED_DIR, "burping", "burping_0007_efd13da0-d40.wav"),
    "discomfort": os.path.join(PROCESSED_DIR, "discomfort", "discomfort_0007_a40b9b8d-e4b.wav"),
    "tired": os.path.join(PROCESSED_DIR, "tired", "tired_0005_999bf14b-e41.wav"),
    "non_baby_cry": os.path.join(PROCESSED_DIR, "non_baby_cry", "115n.wav")
}

print("=== COPYING 6 VERIFIED SAMPLE FILES TO APP/STATIC/SAMPLES/ ===")
for cls, src_path in VERIFIED_FILES.items():
    if os.path.exists(src_path):
        dest_path = os.path.join(SAMPLES_DIR, f"{cls}.wav")
        shutil.copy2(src_path, dest_path)
        print(f"[OK] Copying {cls:15s} -> {dest_path}")
    else:
        print(f"[FAIL] Source not found: {src_path}")

print("Static sample files successfully updated!")
