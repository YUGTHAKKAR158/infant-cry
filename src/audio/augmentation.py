"""
Simple waveform augmentations, used ONLY on training data (never on
validation/test - augmenting eval data would let a model "cheat" by seeing
near-duplicates of what it's evaluated on).

Currently used for one specific purpose: oversampling the minority cry
subgroup identified in docs/dataset.md (cry clips from a different original
source than the dominant one), which the model was found to generalize to
poorly. Each augmented copy is a plausible, slightly different version of the
same clip - not a duplicate - which gives the network more to learn from
without literally repeating exact waveforms it might otherwise memorize.
"""
import numpy as np


def add_gaussian_noise(waveform, snr_db, rng):
    """
    Add random background noise at a target signal-to-noise ratio (SNR).
    A lower snr_db means MORE noise relative to the signal.
    """
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


def random_gain(waveform, gain_range, rng):
    """Scale amplitude by a random factor within gain_range, e.g. (0.7, 1.3)."""
    factor = rng.uniform(*gain_range)
    return waveform * factor


def time_shift(waveform, max_shift_seconds, sample_rate, rng):
    """
    Circularly shift the waveform in time by a random amount (wraps around
    at the boundary instead of losing samples), simulating the cry not
    starting at exactly the same offset every time.
    """
    max_shift_samples = int(max_shift_seconds * sample_rate)
    shift = rng.integers(-max_shift_samples, max_shift_samples + 1)
    return np.roll(waveform, shift)


def augment_waveform(waveform, sample_rate, rng):
    """Apply a random combination of the augmentations above, once."""
    waveform = time_shift(waveform, max_shift_seconds=0.2, sample_rate=sample_rate, rng=rng)
    waveform = random_gain(waveform, gain_range=(0.7, 1.3), rng=rng)
    waveform = add_gaussian_noise(waveform, snr_db=rng.uniform(15, 30), rng=rng)
    peak = np.max(np.abs(waveform))
    if peak > 1e-9:
        waveform = waveform / peak  # re-normalize after gain/noise changes
    return waveform.astype(np.float32)


def spec_augment(mfcc, rng, freq_mask_max=6, time_mask_max=12, num_masks=1):
    """
    "SpecAugment"-style masking, applied to an already-extracted MFCC (not
    the waveform) - zero out one random contiguous band of MFCC coefficients
    and one random contiguous span of time frames.

    Why this helps: our dataset is small (a few hundred clips per class), so
    a CNN can start memorizing specific coefficient/time patterns instead of
    learning general cry characteristics. Masking random chunks forces it to
    rely on the sound being present across multiple regions of the MFCC
    instead of a few specific cells - a well-established, cheap regularizer
    for exactly this kind of small-audio-dataset overfitting risk.

    `mfcc` shape: (n_mfcc, num_frames, 1) as produced by extract_mfcc().
    """
    augmented = mfcc.copy()
    n_mfcc, n_frames, _ = augmented.shape

    for _ in range(num_masks):
        f_width = int(rng.integers(0, freq_mask_max + 1))
        if 0 < f_width < n_mfcc:
            f_start = int(rng.integers(0, n_mfcc - f_width + 1))
            augmented[f_start:f_start + f_width, :, :] = 0.0

        t_width = int(rng.integers(0, time_mask_max + 1))
        if 0 < t_width < n_frames:
            t_start = int(rng.integers(0, n_frames - t_width + 1))
            augmented[:, t_start:t_start + t_width, :] = 0.0

    return augmented.astype(np.float32)
