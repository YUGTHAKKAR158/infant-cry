import requests
import numpy as np
import soundfile as sf
import os

url = "http://127.0.0.1:8000/api/predict"

# 1. Test preset sample belly_pain
res1 = requests.post(url, data={"sample_path": "data/processed/belly_pain/belly_pain_0032_643D64AD-B71.wav"}).json()
print("1. Belly Pain Sample Test:")
print(f"   Prediction: {res1['prediction']} | Confidence: {res1['confidence_pct']}%")
print(f"   Audio URL: {res1['audio_url']}")

# 2. Test quiet silence / room noise
silence = np.random.normal(0, 0.0005, 32000).astype(np.float32)
temp_silence = "temp_silence_test.wav"
sf.write(temp_silence, silence, 16000)

with open(temp_silence, "rb") as f:
    res2 = requests.post(url, files={"file": ("silence.wav", f, "audio/wav")}).json()

print("\n2. Silence Audio Upload Test:")
print(f"   Prediction: {res2['prediction']} | Confidence: {res2['confidence_pct']}%")
print(f"   Action: {res2['recommended_action']}")
print(f"   Audio URL: {res2['audio_url']}")

if os.path.exists(temp_silence):
    os.remove(temp_silence)
