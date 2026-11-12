import os
import glob
from src.utils.config import PROJECT_ROOT, CLASS_NAMES, PROCESSED_DIR

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
RAW_DONATE = os.path.join(PROJECT_ROOT, "data", "raw", "donateacry-corpus")
RAW_NOTCRY = os.path.join(PROJECT_ROOT, "data", "raw", "infant-cry-dataset", "Dataset", "not_cry")

print("=== DATASET COUNT REPORT ===")
print("1. Raw Corpus Counts:")
raw_donate_count = len(glob.glob(os.path.join(RAW_DONATE, "**", "*.*"), recursive=True)) if os.path.exists(RAW_DONATE) else 0
raw_notcry_count = len(glob.glob(os.path.join(RAW_NOTCRY, "*"))) if os.path.exists(RAW_NOTCRY) else 0
print(f"   DonateACry Raw Clips: {raw_donate_count}")
print(f"   Not-Cry Raw Clips: {raw_notcry_count}")
print(f"   Total Raw Files: {raw_donate_count + raw_notcry_count}")

print("\n2. Processed 16kHz 2.0s Clips (Standardized):")
processed_total = 0
for cls in CLASS_NAMES:
    p_folder = os.path.join(PROCESSED_DIR, cls)
    p_count = len(glob.glob(os.path.join(p_folder, "*.wav"))) if os.path.exists(p_folder) else 0
    processed_total += p_count
    print(f"   {cls:15s}: {p_count} clips")
print(f"   Total Processed Clips: {processed_total}")

print("\n3. Cleaned High-Confidence Clips (Audit Filtered):")
cleaned_total = 0
if os.path.exists(CLEANED_DIR):
    for cls in CLASS_NAMES:
        c_folder = os.path.join(CLEANED_DIR, cls)
        c_count = len(glob.glob(os.path.join(c_folder, "*.wav"))) if os.path.exists(c_folder) else 0
        cleaned_total += c_count
        print(f"   {cls:15s}: {c_count} clips")
    print(f"   Total Cleaned Clips: {cleaned_total}")
