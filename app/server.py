"""
FastAPI Server for Baby Cry Acoustic Intelligence Dashboard

Provides API endpoints for live audio classification, sample loading,
spectrogram generation, and metric reporting.
"""

import os
import io
import base64
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.audio.load_audio import load_audio
from src.audio.preprocessing import standardize_audio
from src.audio.fractal_features import extract_frta_features
from src.utils.config import SAMPLE_RATE, CLASS_NAMES, MODELS_DIR, PROCESSED_DIR, WIN_LENGTH, HOP_LENGTH, N_FFT, PROJECT_ROOT

app = FastAPI(title="Baby Cry Diagnosis Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# Load trained DeepSVM model & feature artifacts at startup
deep_svm_model = None
inca_selector = None
norm_mean = None
norm_std = None


@app.on_event("startup")
def load_artifacts():
    global deep_svm_model, inca_selector, norm_mean, norm_std
    try:
        model_path = os.path.join(MODELS_DIR, "deep_svm_model.joblib")
        selector_path = os.path.join(MODELS_DIR, "inca_selector.joblib")
        stats_path = os.path.join(MODELS_DIR, "frta_norm_stats.npz")

        if os.path.exists(model_path):
            deep_svm_model = joblib.load(model_path)
            inca_selector = joblib.load(selector_path)
            stats = np.load(stats_path)
            norm_mean = stats["mean"]
            norm_std = stats["std"]
            print("Successfully loaded DeepSVM model and iNCA selector!")
        else:
            print("Warning: DeepSVM model file not found in models/.")
    except Exception as e:
        print(f"Error loading DeepSVM artifacts: {e}")


def generate_spectrogram_base64(waveform, sr=SAMPLE_RATE):
    """Generate a high-res spectrogram plot and return as base64 PNG string."""
    stft = np.abs(librosa.stft(y=waveform, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH))
    spec_db = librosa.amplitude_to_db(stft, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 2.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    img = ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


RECOMMENDED_ACTIONS = {
    "hungry": "The baby is exhibiting hunger acoustic cues (rhythmic crying with repetitive 'neh' phonetic patterns). Recommended action: Feed infant and check feeding log.",
    "belly_pain": "The baby is exhibiting belly pain / colic distress cues (intense, sudden high-pitched crying). Recommended action: Hold infant upright, gently massage tummy, or apply warmth.",
    "burping": "The baby needs to burp (trapped air pressure acoustics). Recommended action: Hold baby against shoulder and gently pat back until air is released.",
    "discomfort": "The baby is experiencing physical discomfort (wet diaper, temperature, friction). Recommended action: Check diaper, clothing tightness, and room temperature.",
    "tired": "The baby is showing fatigue / overtired vocalizations (yawn-like crying intervals). Recommended action: Reduce environmental stimulation and soothe infant to sleep.",
    "non_baby_cry": "No infant crying detected (background sound or non-cry voice). Recommended action: Continue ambient monitoring."
}


@app.get("/", response_class=HTMLResponse)
def read_root():
    template_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(template_file):
        with open(template_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard template loading...</h1>"


SAMPLE_DISPLAY_NAMES = {
    "hungry": "hungry_0721_F4A0C710-AC2.wav",
    "belly_pain": "belly_pain_0032_643D64AD-B71.wav",
    "burping": "burping_0007_efd13da0-d40.wav",
    "discomfort": "discomfort_0007_a40b9b8d-e4b.wav",
    "tired": "tired_0005_999bf14b-e41.wav",
    "non_baby_cry": "115n.wav",
}


@app.get("/api/samples")
def get_sample_files():
    """Return preset sample audio files for easy 1-click testing."""
    samples = []
    for cls in CLASS_NAMES:
        sample_path = os.path.join(STATIC_DIR, "samples", f"{cls}.wav")
        if os.path.exists(sample_path):
            rel_path = os.path.relpath(sample_path, PROJECT_ROOT).replace("\\", "/")
            samples.append({
                "class_name": cls,
                "file_name": SAMPLE_DISPLAY_NAMES.get(cls, f"{cls}.wav"),
                "rel_path": rel_path
            })
    return {"samples": samples}


@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(None), sample_path: str = Form(None)):
    """API endpoint to run live diagnosis on uploaded file or preset sample."""
    temp_path = None
    try:
        if file:
            contents = await file.read()
            temp_dir = os.path.join(BASE_DIR, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as f:
                f.write(contents)
            target_file = temp_path
        elif sample_path:
            target_file = sample_path
        else:
            raise HTTPException(status_code=400, detail="No audio file or sample path provided.")

        import time
        import soundfile as sf
        start_time = time.perf_counter()

        waveform, sr = load_audio(target_file)
        
        # Measure raw signal metrics before amplitude normalization
        raw_rms = float(np.sqrt(np.mean(waveform ** 2)))
        raw_peak = float(np.max(np.abs(waveform)))

        standardized = standardize_audio(waveform, sr)

        # Save processed standardized 16kHz PCM WAV for UI playback
        preview_dir = os.path.join(STATIC_DIR, "temp")
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, "preview.wav")
        sf.write(preview_path, standardized, SAMPLE_RATE, subtype="PCM_16")
        audio_url = f"/static/temp/preview.wav?t={int(time.time() * 1000)}"

        # Compute acoustic energy & vocalization metrics
        stft = np.abs(librosa.stft(standardized, n_fft=512))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=512)
        cry_band_mask = (freqs >= 250) & (freqs <= 2500)
        total_energy = float(np.sum(stft ** 2)) + 1e-9
        cry_band_energy = float(np.sum(stft[cry_band_mask, :] ** 2))
        cry_energy_ratio = cry_band_energy / total_energy
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=standardized)))

        # Extract FRTA features & run DeepSVM model
        frta_vec = extract_frta_features(standardized, SAMPLE_RATE)
        frta_norm = (frta_vec[np.newaxis, :] - norm_mean) / norm_std
        frta_inca = inca_selector.transform(frta_norm)

        probs = deep_svm_model.predict_proba(frta_inca)[0]
        inference_ms = (time.perf_counter() - start_time) * 1000.0

        predicted_idx = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Acoustic Safeguard: VAD Silence / Ambient Noise / Non-Cry Override
        is_silence_or_noise = (raw_rms < 0.012 or raw_peak < 0.025)
        is_unvocalized_flat_noise = (spectral_flatness > 0.015 and cry_energy_ratio < 0.35)

        if is_silence_or_noise or is_unvocalized_flat_noise:
            predicted_label = "non_baby_cry"
            confidence = 0.99
            probs = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.99])

        # Generate Base64 Spectrogram
        spec_b64 = generate_spectrogram_base64(standardized, SAMPLE_RATE)

        # Feature highlights
        frta_df = float(frta_vec[2])  # Fractal dimension sample
        spectral_cent = float(np.mean(librosa.feature.spectral_centroid(y=standardized, sr=SAMPLE_RATE)))

        file_name = file.filename if file else os.path.basename(sample_path)
        source_type = "Uploaded File" if file else "Preset Sample"

        response_data = {
            "prediction": predicted_label,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100, 1),
            "inference_ms": round(inference_ms, 2),
            "file_name": file_name,
            "source_type": source_type,
            "audio_url": audio_url,
            "probabilities": {name: round(float(p) * 100, 1) for name, p in zip(CLASS_NAMES, probs)},
            "recommended_action": RECOMMENDED_ACTIONS.get(predicted_label, ""),
            "spectrogram_b64": spec_b64,
            "features": {
                "fractal_dimension_df": round(frta_df, 3),
                "spectral_centroid_hz": round(spectral_cent, 1),
                "sample_rate_hz": SAMPLE_RATE,
                "duration_seconds": 2.0
            }
        }

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return JSONResponse(content=response_data)

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
