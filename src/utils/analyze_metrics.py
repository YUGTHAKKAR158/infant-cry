import os
import glob
import numpy as np
import librosa
import joblib

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.fractal_features import extract_frta_features
from src.utils.config import PROCESSED_DIR, CLASS_NAMES, MODELS_DIR, SAMPLE_RATE

def analyze_audio_characteristics(file_path):
    waveform, sr = load_audio(file_path)
    
    # 1. Measure raw RMS before normalization
    raw_rms = float(np.sqrt(np.mean(waveform**2)))
    raw_peak = float(np.max(np.abs(waveform)))
    
    # Standardize
    std_waveform = standardize_audio(waveform, sr)
    
    # 2. Spectral energy in infant cry vocal band (250 Hz - 2500 Hz)
    stft = np.abs(librosa.stft(std_waveform, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    cry_band_mask = (freqs >= 250) & (freqs <= 2500)
    
    total_energy = np.sum(stft**2) + 1e-9
    cry_band_energy = np.sum(stft[cry_band_mask, :]**2)
    cry_energy_ratio = float(cry_band_energy / total_energy)
    
    # 3. Pitch / Fundamental frequency detection
    pitches, magnitudes = librosa.piptrack(y=std_waveform, sr=sr, fmin=200, fmax=800)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    has_valid_cry_pitch = len(pitch_vals) > 10 and np.mean(pitch_vals) >= 250
    
    # 4. Zero crossing rate & Spectral Flatness (noise vs harmonic voice)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=std_waveform)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=std_waveform)))
    
    return {
        "raw_rms": raw_rms,
        "raw_peak": raw_peak,
        "cry_energy_ratio": cry_energy_ratio,
        "spectral_flatness": flatness,
        "zcr": zcr,
        "has_valid_cry_pitch": has_valid_cry_pitch
    }

if __name__ == "__main__":
    print("Analyzing audio characteristics across classes...")
    for cls in CLASS_NAMES:
        files = glob.glob(os.path.join(PROCESSED_DIR, cls, "*.wav"))[:5]
        print(f"\n--- Class: {cls} ---")
        for f in files:
            stats = analyze_audio_characteristics(f)
            fname = os.path.basename(f)[:25]
            print(f"{fname:25s} | RawRMS: {stats['raw_rms']:.5f} | Peak: {stats['raw_peak']:.4f} | CryBand: {stats['cry_energy_ratio']:.3f} | Flatness: {stats['spectral_flatness']:.4f}")
