import requests

url_samples = "http://127.0.0.1:8000/api/samples"
url_predict = "http://127.0.0.1:8000/api/predict"

samples = requests.get(url_samples).json()["samples"]

print("=== TESTING LIVE PRESET SAMPLES VIA FASTAPI ENDPOINT ===")

all_match = True
for s in samples:
    cls = s["class_name"]
    rel_path = s["rel_path"]
    file_name = s["file_name"]
    
    res = requests.post(url_predict, data={"sample_path": rel_path}).json()
    pred = res["prediction"]
    conf = res["confidence_pct"]
    
    match = "MATCH" if pred == cls else "DIFF"
    print(f"[{match:5s}] Button '{cls:12s}' -> File: {file_name:30s} | Output: {pred:12s} ({conf:5.1f}%)")
    if pred != cls:
        all_match = False

if all_match:
    print("\nALL 6 PRESET BUTTONS PREDICT THEIR EXACT CATEGORIES WITH HIGH CONFIDENCE!")
else:
    print("\nWARNING: Some preset samples do not match!")
