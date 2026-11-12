"""
Step 5 & 6: look at what the model will actually receive.

This script is deliberately separate from any training code. Its only job is
to plot, for a few example clips from each class:

  1. Waveform    - amplitude over time. The rawest possible view of the audio.
  2. Spectrogram - how frequency content changes over time. Computed with the
                   Short-Time Fourier Transform (STFT): the signal is cut into
                   short overlapping windows, and each window's frequency
                   content is computed. Stacking these side by side over time
                   gives a 2-D "image" of frequency (y-axis) vs time (x-axis),
                   with color = energy at that frequency/time.
  3. MFCC        - Mel-Frequency Cepstral Coefficients. Built from the
                   spectrogram by (a) warping the frequency axis onto the
                   "Mel scale" (which matches how human hearing perceives
                   pitch - roughly logarithmic, more resolution at low
                   frequencies), (b) taking the log of the energy, and
                   (c) applying a Discrete Cosine Transform to compress it
                   into a small number of coefficients per time frame. The
                   result is a much more compact representation than a full
                   spectrogram, which is exactly why it's popular for
                   speech/audio classification on constrained hardware like
                   an ESP32 - fewer numbers in means a smaller/faster model.

Run with:
    .venv\\Scripts\\python.exe -m src.audio.visualize
"""
import os
import glob
import random

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # save to file instead of trying to open a window
import matplotlib.pyplot as plt

from src.utils.config import PROCESSED_DIR, SAMPLE_RATE, NUM_MFCC, N_FFT, HOP_LENGTH, OUTPUTS_DIR

NUM_EXAMPLES_PER_CLASS = 3


def plot_example(file_path, label, out_dir):
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    name = os.path.splitext(os.path.basename(file_path))[0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"{label} - {name}")

    # 1. Waveform
    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    axes[0].plot(time_axis, waveform, linewidth=0.5)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # 2. Spectrogram (STFT magnitude, in decibels for visibility)
    stft = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img1 = librosa.display.specshow(
        spectrogram_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=axes[1]
    )
    axes[1].set_title("Spectrogram (STFT)")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    img2 = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis="time", ax=axes[2])
    axes[2].set_title(f"MFCC ({NUM_MFCC} coefficients)")
    axes[2].set_ylabel("Coefficient index")
    fig.colorbar(img2, ax=axes[2])

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{label}_{name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path, mfcc.shape


def main():
    out_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(42)  # reproducible example selection

    for label in ["baby_cry", "non_baby_cry"]:
        files = glob.glob(os.path.join(PROCESSED_DIR, label, "*.wav"))
        chosen = random.sample(files, min(NUM_EXAMPLES_PER_CLASS, len(files)))

        for f in chosen:
            out_path, mfcc_shape = plot_example(f, label, out_dir)
            print(f"saved {out_path}  (MFCC shape: {mfcc_shape})")


if __name__ == "__main__":
    main()
