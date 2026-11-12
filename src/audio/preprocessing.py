"""
Step 2 of the audio pipeline: turn a raw, inconsistent waveform into a
standardized one that every downstream step (visualization, feature
extraction, training, and eventually the ESP32 firmware) can rely on having
the exact same shape and characteristics.

Each function below does ONE transformation, so you can print/plot the
waveform after any individual step while learning what it changes.
"""
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

from src.utils.config import SAMPLE_RATE, NUM_SAMPLES, LOWPASS_CUTOFF_HZ


def resample_audio(waveform, original_sr, target_sr=SAMPLE_RATE):
    """
    Convert a waveform recorded at `original_sr` to `target_sr`.

    Why needed: our raw dataset has files at both 8000 Hz and 44100 Hz
    (see docs/dataset.md). A neural network expects every input to represent
    the same amount of real time per array position, so all clips must share
    one sample rate before we can batch them together.
    """
    if original_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)


def apply_lowpass_filter(waveform, sr=SAMPLE_RATE, cutoff_hz=LOWPASS_CUTOFF_HZ):
    """
    Remove frequency content above `cutoff_hz` from the waveform.

    Why needed: part of our raw dataset was originally recorded at 8000 Hz,
    which physically cannot contain any content above 4000 Hz (the Nyquist
    limit). The rest of the dataset was recorded at 44100 Hz and DOES contain
    higher-frequency content. If we left this untouched, a model could learn
    to tell classes apart just by checking "is there anything above 4kHz?"
    instead of learning real cry-vs-not-cry acoustic patterns - see the bias
    discussion in docs/dataset.md. Applying the same cutoff to every clip
    removes that shortcut.

    Implementation note: a Butterworth filter is a standard, smooth way to
    cut frequencies above a threshold without introducing sharp artifacts.
    `sos` (second-order sections) form is used because it is numerically more
    stable than the alternative `ba` form for this kind of filter.
    """
    nyquist = sr / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(N=8, Wn=normalized_cutoff, btype="low", output="sos")
    return sosfilt(sos, waveform).astype(np.float32)


def crop_to_most_energetic_window(waveform, num_samples=NUM_SAMPLES):
    """
    Cut the waveform down to `num_samples`, choosing the window that contains
    the most energy (i.e. is loudest / most "active") rather than always
    taking the start of the clip.

    Why needed: our raw clips (~5-7s) often have several seconds of near-
    silence before/after the actual sound of interest (verified by measuring
    RMS energy across full clips - see project notes). Naively slicing
    `waveform[:num_samples]` from the start would frequently grab mostly
    silence and cut off most of the real cry, corrupting the label. Instead,
    we slide a `num_samples`-long window across the clip in small steps and
    keep the position with the highest total energy - a simple, well-known
    technique ("energy-based endpointing") for locating the informative part
    of a longer recording.
    """
    if len(waveform) <= num_samples:
        return waveform  # nothing to crop; padding is handled separately

    step = int(0.1 * SAMPLE_RATE)  # slide the window in 0.1s steps
    best_start = 0
    best_energy = -1.0

    for start in range(0, len(waveform) - num_samples + 1, step):
        window = waveform[start:start + num_samples]
        energy = np.sum(window ** 2)
        if energy > best_energy:
            best_energy = energy
            best_start = start

    return waveform[best_start:best_start + num_samples]


def trim_or_pad(waveform, num_samples=NUM_SAMPLES):
    """
    Force the waveform to be exactly `num_samples` long.

    - If it's longer, crop to the most energetic `num_samples`-long window
      (see `crop_to_most_energetic_window`) instead of blindly taking the
      start - a cry can occur anywhere in the original clip.
    - If it's shorter, pad the end with zeros (silence).
    """
    current_length = len(waveform)

    if current_length == num_samples:
        return waveform
    if current_length > num_samples:
        return crop_to_most_energetic_window(waveform, num_samples)

    padding = np.zeros(num_samples - current_length, dtype=waveform.dtype)
    return np.concatenate([waveform, padding])


def normalize_amplitude(waveform):
    """
    Scale the waveform so its peak absolute amplitude is 1.0.

    Why needed: recordings were made at different volumes/distances. Without
    normalization, the model could partly learn "loud = one class" instead of
    the actual sound pattern, which would not generalize to a real ESP32
    microphone at an unknown distance/gain.
    """
    peak = np.max(np.abs(waveform))
    if peak < 1e-9:
        return waveform  # silent clip - avoid divide-by-zero, leave as-is
    return waveform / peak


def standardize_audio(waveform, original_sr):
    """
    Run the full preprocessing chain in the correct order:
    resample -> low-pass filter -> trim/pad -> normalize.

    This is the ONE function that both `prepare_dataset.py` (training data)
    and `predict.py` (new/unseen audio) must both call, so training and
    inference never drift apart - see docs/dataset.md's note on why that
    consistency matters.
    """
    waveform = resample_audio(waveform, original_sr, SAMPLE_RATE)
    waveform = apply_lowpass_filter(waveform, SAMPLE_RATE, LOWPASS_CUTOFF_HZ)
    waveform = trim_or_pad(waveform, NUM_SAMPLES)
    waveform = normalize_amplitude(waveform)
    return waveform
