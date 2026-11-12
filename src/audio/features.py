"""
Step 6: turn a standardized waveform into the numeric representation the CNN
actually trains on.

We use MFCC (Mel-Frequency Cepstral Coefficients) rather than a raw
spectrogram. Quick recap (full explanation in src/audio/visualize.py's
docstring):

    waveform -> STFT -> mel filter bank -> log -> DCT -> MFCC

Why MFCC for this project specifically:
  - Compact: a spectrogram frame might have 257 frequency bins; MFCC
    compresses that down to ~40 numbers per frame while keeping the
    information most relevant to distinguishing sounds. Smaller input means
    a smaller, faster CNN - directly relevant to fitting on an ESP32 later.
  - Standard for speech/audio classification, well supported by librosa here
    and by lightweight C implementations on embedded targets later.
  - We are NOT choosing it purely for laptop accuracy - a raw spectrogram
    might do just as well or better here, but MFCC's smaller size is the
    deciding factor given the eventual TinyML deployment target.

Output shape: with our fixed 2.0s / 16kHz clips (NUM_SAMPLES = 32000) and
N_FFT=512, HOP_LENGTH=256, librosa produces (NUM_MFCC, 126) - i.e. 40
coefficients per frame, 126 frames covering the 2 seconds. We add a trailing
channel dimension (-> (40, 126, 1)) because Keras Conv2D layers expect a
channel axis, the same way an RGB image has 3 channels; ours just has 1
("grayscale" MFCC).
"""
import numpy as np
import librosa

from src.utils.config import SAMPLE_RATE, NUM_MFCC, N_FFT, WIN_LENGTH, HOP_LENGTH


def extract_mfcc(waveform, sr=SAMPLE_RATE):
    """
    Compute MFCCs using 25ms window (WIN_LENGTH=400) and 40% overlap (HOP_LENGTH=240),
    returning them as (NUM_MFCC, num_frames, 1) float32 array.
    """
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
    )
    mfcc = mfcc.astype(np.float32)
    return mfcc[..., np.newaxis]  # add channel dimension
