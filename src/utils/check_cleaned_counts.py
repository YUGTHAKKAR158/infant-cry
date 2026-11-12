import os
import glob
from src.utils.config import PROJECT_ROOT, CLASS_NAMES

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
for cls in CLASS_NAMES:
    folder = os.path.join(CLEANED_DIR, cls)
    files = glob.glob(os.path.join(folder, "*.wav")) if os.path.exists(folder) else []
    print(f"Class '{cls:15s}': {len(files)} files in data/cleaned/")
