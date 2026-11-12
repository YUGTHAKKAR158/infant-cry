# Infant Cry Acoustic Intelligence & Diagnosis System

An end-to-end Machine Learning and TinyML system designed to detect and classify infant crying reasons into 6 distinct categories: **Hungry**, **Belly Pain**, **Burping**, **Discomfort**, **Tired**, and **Non-Baby Cry**.

The project includes acoustic feature engineering (MFCCs, spectral features, fractal dimension analysis), model training (Deep SVM & CNN), TinyML quantization for edge devices, and an interactive real-time web dashboard.

---

## Key Features

- **Standardized Audio Preprocessing**: Standardizes input clips to 16 kHz mono with 4 kHz low-pass filtering to remove recording hardware bias.
- **Acoustic & Fractal Feature Extraction**: Extracts 40-dimensional MFCCs, spectral descriptors, and Fractal/FRTA features for high discriminatory power.
- **Dual Model Architectures**:
  - **Deep SVM Classifier** with INCA feature selection for robust tabular acoustic classification.
  - **Convolutional Neural Network (CNN)** in TensorFlow/Keras.
- **TinyML & Edge Deployment**: Quantized INT8 and Float32 `.tflite` models optimized for microcontrollers and edge hardware.
- **Interactive Web Dashboard**: Built with FastAPI, providing live audio upload, one-click preset test buttons, real-time waveform/spectrogram rendering, and confidence gauges.

---

## Repository Structure

```text
├── app/                      # Web dashboard application
│   ├── server.py             # FastAPI backend server
│   ├── templates/            # HTML interface
│   └── static/               # CSS, JavaScript & preset audio samples
├── docs/                     # Technical documentation & research notes
│   ├── dataset.md            # Dataset sources, statistics & bias analysis
│   └── tinyml_conversion.md  # Edge conversion & quantization benchmarks
├── models/                   # Pre-trained models (.keras, .tflite, .joblib)
├── outputs/                  # Training curves, confusion matrices & metrics
│   ├── metrics/              # Performance reports (JSON)
│   └── plots/                # High-res confusion matrix & loss curves
├── src/                      # Core Python source code
│   ├── audio/                # Audio loading, augmentation, and feature extraction
│   ├── model/                # Model architectures, training, evaluation, TinyML
│   └── utils/                # Configuration constants and validation scripts
├── requirements.txt          # Python dependencies
└── .gitignore                # Excludes virtual environments, datasets & caches
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Web Dashboard

Start the FastAPI application:

```bash
python -m uvicorn app.server:app --reload --port 8000
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

You can test audio directly using the built-in preset samples or upload your own `.wav` files.

---

## Dataset

This project utilizes public acoustic corpora:
- **Infant Cry Dataset** (Kaggle)
- **Donate-a-Cry Corpus**

Due to GitHub size recommendations, raw audio datasets (`data/`) are excluded from this repository. For details on how the dataset was cleaned, balanced, and preprocessed, see [docs/dataset.md](docs/dataset.md).

---

## TinyML & Edge Deployment

The model was converted and quantized into TensorFlow Lite (`.tflite`) formats (Dynamic INT8, Full INT8, and Float32) to evaluate latency, memory footprint, and edge feasibility. See [docs/tinyml_conversion.md](docs/tinyml_conversion.md) for full benchmarks.

