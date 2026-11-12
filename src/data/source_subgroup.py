"""
Identifies which `baby_cry` files come from the minority recording source
(native 44.1kHz) vs the dominant one (native 8kHz, Donate-a-Cry signature) -
see the bias discussion in docs/dataset.md.

This is metadata about the RAW files (before standardization), so we look at
data/raw/, not data/processed/ (which is uniformly 16kHz after preprocessing
and no longer carries this information).

Used by:
  - src/model/train.py    - to oversample/augment the minority subgroup
  - src/model/evaluate.py - to report accuracy separately per subgroup
"""
import os
import glob

import soundfile as sf

from src.utils.config import RAW_DATASET_DIR

DOMINANT_SOURCE_SAMPLE_RATE = 8000
MINORITY_SOURCE_SAMPLE_RATE = 44100


def get_cry_native_sample_rates():
    """Return {basename_without_extension: native_sample_rate} for every raw cry file."""
    native_rates = {}
    for f in glob.glob(os.path.join(RAW_DATASET_DIR, "cry", "*")):
        if os.path.isfile(f):
            name = os.path.splitext(os.path.basename(f))[0]
            native_rates[name] = sf.info(f).samplerate
    return native_rates


def is_minority_subgroup_file(processed_file_path, native_rates):
    """
    Given a path into data/processed/baby_cry/, check whether the
    corresponding raw file was natively MINORITY_SOURCE_SAMPLE_RATE.
    """
    name = os.path.splitext(os.path.basename(processed_file_path))[0]
    return native_rates.get(name) == MINORITY_SOURCE_SAMPLE_RATE
