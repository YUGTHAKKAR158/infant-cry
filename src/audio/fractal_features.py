"""
Fractal-based Regional Texture Analysis (FRTA) & Iterative Neighborhood Component Analysis (iNCA)

Based on Section 3.2 & Algorithm 1 of:
"Can You Understand Why I Am Crying? A Decision-making System for Classifying Infants' Cry Languages Based on DeepSVM Model"
(Rezaee et al., ACM 2024)
"""

import numpy as np
import librosa
from scipy.ndimage import binary_dilation
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from src.utils.config import SAMPLE_RATE, WIN_LENGTH, HOP_LENGTH, N_FFT


def box_counting_fractal_dimension(binary_image):
    """
    Compute 2D Box-Counting Fractal Dimension (D_f) of a binary mask/boundary.
    D_f = slope of log(1 / box_size) vs log(N(box_size)).
    """
    if not np.any(binary_image):
        return 0.0

    min_dim = min(binary_image.shape)
    if min_dim < 4:
        return 0.0

    box_sizes, box_counts = [], []
    size = 2
    while size <= min_dim // 2:
        hb, wb = binary_image.shape[0] // size, binary_image.shape[1] // size
        trimmed = binary_image[: hb * size, : wb * size]
        blocks = trimmed.reshape(hb, size, wb, size)
        cnt = np.sum(np.sum(blocks, axis=(1, 3)) > 0)
        if cnt > 0:
            box_sizes.append(size)
            box_counts.append(cnt)
        size *= 2

    if len(box_sizes) < 2:
        return 0.0

    log_sizes = np.log(1.0 / np.array(box_sizes, dtype=np.float64))
    log_counts = np.log(np.array(box_counts, dtype=np.float64))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])


def extract_frta_features(waveform, sr=SAMPLE_RATE):
    """
    Extract 60+ FRTA fractal texture and spectral descriptors from an audio waveform.
    """
    # 1. Compute STFT Spectrogram (25ms window, 40% overlap)
    stft = np.abs(librosa.stft(y=waveform, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH))
    spec_db = librosa.amplitude_to_db(stft, ref=np.max)

    # Normalize spectrogram to 0-255 uint8
    s_min, s_max = spec_db.min(), spec_db.max()
    s_norm = ((spec_db - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.uint8)

    features = []

    # Otsu-inspired bi-thresholding intervals
    thresholds = [(0, 64), (64, 128), (128, 192), (192, 255), (32, 160), (96, 224)]

    for t_l, t_u in thresholds:
        mask = (s_norm > t_l) & (s_norm <= t_u)
        area = np.mean(mask)
        mean_gray = np.mean(s_norm[mask]) / 255.0 if np.any(mask) else 0.0

        dilated = binary_dilation(mask)
        boundary = dilated & (~mask)

        df_mask = box_counting_fractal_dimension(mask)
        df_boundary = box_counting_fractal_dimension(boundary)

        features.extend([area, mean_gray, df_mask, df_boundary])

    # 2. Spectral descriptors (centroid, bandwidth, roll-off, ZCR, spectral contrast)
    cent = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=waveform, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)

    for feat in [cent, bw, rolloff, zcr]:
        features.extend([float(np.mean(feat)), float(np.std(feat))])

    for band in contrast:
        features.extend([float(np.mean(band)), float(np.std(band))])

    # 3. Summary MFCC statistics (mean & std for 12 coefficients)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=12, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    for m in mfcc:
        features.extend([float(np.mean(m)), float(np.std(m))])

    return np.array(features, dtype=np.float32)


class INCAFeatureSelector:
    """
    Iterative Neighborhood Component Analysis (iNCA) feature selector.
    """

    def __init__(self, n_components=32, random_state=42):
        self.n_components = n_components
        self.nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=random_state)

    def fit_transform(self, X, y):
        return self.nca.fit_transform(X, y)

    def transform(self, X):
        return self.nca.transform(X)
