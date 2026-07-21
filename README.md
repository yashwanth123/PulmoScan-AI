# PulmoScan AI — Lung Imaging Intelligence Platform

<p align="center">
  <strong>AI-powered chest X-ray & lung scan analysis with explainable deep learning</strong>
</p>

<p align="center">
  EfficientNetB0 · Multi-class detection · Grad-CAM · Clinical-grade UI · Docker-ready
</p>

---

## Overview

**PulmoScan AI** is a production-style medical imaging platform evolved from a GITAM University B.Tech research project. It analyzes chest X-rays and related lung scans to detect **COVID-19**, **Pneumonia**, **Normal**, and **Tuberculosis** using transfer learning.

> ⚠️ **Disclaimer:** Research and screening tool only. Not FDA/CE approved. Always consult qualified medical professionals.

---

## Project Structure

```
pulmoscan-ai/
├── backend/app/              # FastAPI application
│   ├── main.py               # Server entry point
│   ├── config.py             # Configuration
│   ├── api/routes/           # REST endpoints
│   └── ml/                   # Model, inference, Grad-CAM
├── frontend/                 # Web UI (HTML/CSS/JS)
├── ml_training/              # Training & evaluation modules
│   ├── dataset.py            # Data download & generators
│   ├── train.py              # Training CLI
│   ├── evaluate.py           # Metrics & reports
│   └── evaluate_cli.py       # Evaluation CLI
├── scripts/
│   ├── download_data.py      # Fetch dataset
│   ├── predict.py            # CLI inference
│   └── test.sh               # Run test suite
├── tests/                    # Pytest test suite
├── models/                   # Saved model weights
├── run.py                    # Start web platform
├── Makefile                  # Common commands
├── requirements.txt
└── requirements-dev.txt
```

---

## Quick Start

### Requirements

- **Python 3.10, 3.11, or 3.12** (recommended)
- Python 3.13 works with TensorFlow 2.20+
- macOS: use `python3` and `pip3` (or a virtual environment)

### 1. Install

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for tests
```

### 2. Train (recommended for accurate predictions)

```bash
# Download dataset (~1.3 GB) and train
python scripts/download_data.py
python ml_training/train.py

# Quick smoke training (subset, 2 epochs)
python ml_training/train.py --quick --epochs 2 --no-fine-tune
```

### 3. Run the platform

```bash
python3 run.py
```

Open **http://localhost:8000**

### Docker

```bash
docker compose up --build
```

---

## How to Test

### Run all tests

```bash
# Using Make (recommended)
make install-dev
make test

# Or directly
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

### Fast tests (skip TensorFlow inference)

```bash
make test-fast
# equivalent to:
python3 -m pytest tests/ -v -m "not slow"
```

### API tests only

```bash
make test-api
```

### Manual API smoke test

Start the server in one terminal:

```bash
python3 run.py
```

In another terminal:

```bash
# Health check
curl http://localhost:8000/api/health

# Model info
curl http://localhost:8000/api/model

# Predict on an image
curl -X POST http://localhost:8000/api/predict \
  -F "file=@/path/to/chest_xray.png" \
  -F "scan_type=chest_xray" \
  -F "include_gradcam=true"
```

### CLI inference test

```bash
python3 scripts/predict.py /path/to/chest_xray.png --scan-type chest_xray
```

### Evaluate a trained model

```bash
python3 ml_training/evaluate_cli.py --model models/pulmoscan_efficientnet.keras
```

### Expected test output

```
tests/test_preprocessing.py ....          PASSED
tests/test_dataset.py ....                PASSED
tests/test_predictor.py ....              PASSED
tests/test_api.py ........                PASSED
```

Slow tests (`@pytest.mark.slow`) load TensorFlow and run real inference — include them before release:

```bash
python3 -m pytest tests/ -v -m slow
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/model` | Model info & status |
| `GET` | `/api/scan-types` | Supported imaging modalities |
| `POST` | `/api/predict` | Upload scan → diagnosis + Grad-CAM |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/history` | Recent predictions |

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install runtime dependencies |
| `make install-dev` | Install runtime + test dependencies |
| `make run` | Start web platform |
| `make train` | Quick training smoke run |
| `make test` | Full pytest suite |
| `make test-fast` | Skip slow TensorFlow tests |
| `make test-api` | API integration tests only |
| `make clean` | Remove Python cache files |

---

## Model Performance

From original research (extended dataset, 6568 images):

| Task | Best Model | Accuracy | F1 | AUC |
|------|-----------|----------|-----|-----|
| Binary (COVID vs Normal) | InceptionResNetV2 | 99.59% | 0.996 | 99.2% |
| 3-class | DenseNet121 | 79.0% | 0.82 | 92.0% |
| Extended | DenseNet201 | 83.6% | 0.81 | 95.6% |

PulmoScan AI uses **EfficientNetB0** for the best speed/accuracy tradeoff.

---

## Tech Stack

- **Backend:** Python 3.11, FastAPI, TensorFlow/Keras, OpenCV
- **Model:** EfficientNetB0 + custom classification head
- **Testing:** pytest, httpx, FastAPI TestClient
- **Frontend:** Vanilla JS, responsive CSS
- **Deploy:** Docker, Uvicorn

---

## Original Research

Developed as a B.Tech final year project at **GITAM University** under **Dr. Don S. Kumar**.

**Authors:** L. Naga Sai Sri Ravi Teja · S. Ritesh Dev · K. Bharath · T. Yashwanth Sai

**Datasets:**
- [Kaggle Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [education454 COVID-19 Dataset](https://github.com/education454/datasets)

---

## License

MIT License — for research and educational use only. Not for clinical diagnosis.
