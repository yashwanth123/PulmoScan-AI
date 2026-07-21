# PulmoScan AI — Lung Imaging Intelligence Platform

<p align="center">
  <strong>AI-powered chest X-ray & lung scan analysis with explainable deep learning</strong>
</p>

<p align="center">
  EfficientNetB0 · Multi-class detection · Grad-CAM · Clinical-grade UI · Docker-ready
</p>

---

## Overview

**PulmoScan AI** is a full-stack medical imaging platform evolved from a GITAM University B.Tech research project. It analyzes chest X-rays and related lung scans to detect **COVID-19**, **Pneumonia**, **Normal**, and **Tuberculosis** patterns using state-of-the-art transfer learning.

> ⚠️ **Disclaimer:** Research and screening tool only. Not FDA/CE approved. Always consult qualified medical professionals.

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **EfficientNetB0** | ImageNet-pretrained backbone, fine-tuned for lung imaging |
| 🎯 **Multi-class detection** | COVID-19, Normal, Pneumonia, Tuberculosis |
| 🔥 **Grad-CAM** | Visual explainability heatmaps on predictions |
| 🩻 **Multi-modality UI** | Chest X-Ray, CT Scan (MRI coming soon) |
| 📊 **Dashboard** | Real-time stats, detection breakdown, history |
| 🐳 **Docker** | One-command deployment |
| 📱 **Responsive UI** | Clinical dark-theme interface for desktop & mobile |

---

## Architecture

```
pulmoscan-ai/
├── backend/app/          # FastAPI server
│   ├── main.py           # Application entry
│   ├── config.py         # Settings & class labels
│   ├── api/routes/       # REST endpoints
│   └── ml/               # Model, predictor, Grad-CAM
├── frontend/             # Modern web UI
│   ├── index.html
│   └── assets/css,js/
├── ml_training/          # Training pipeline
│   └── train.py
├── models/               # Saved model weights
├── uploads/              # Uploaded scans (audit)
├── data/                 # Dataset cache
├── run.py                # Start server
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (recommended)

Downloads the COVID-19 X-ray dataset (~1.3 GB) and fine-tunes EfficientNetB0:

```bash
# Full training (~30 min on GPU)
python ml_training/train.py

# Quick demo training (~5 min)
python ml_training/train.py --quick --epochs 5
```

### 3. Launch the platform

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

### Docker

```bash
docker compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/model` | Model info & status |
| `GET` | `/api/scan-types` | Supported imaging modalities |
| `POST` | `/api/predict` | Upload scan → diagnosis + Grad-CAM |
| `GET` | `/api/stats` | Dashboard statistics |
| `GET` | `/api/history` | Recent predictions |

### Example: Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@chest_xray.jpg" \
  -F "scan_type=chest_xray" \
  -F "include_gradcam=true"
```

---

## Model Performance

From original research (extended dataset, 6568 images):

| Task | Best Model | Accuracy | F1 | AUC |
|------|-----------|----------|-----|-----|
| Binary (COVID vs Normal) | InceptionResNetV2 | 99.59% | 0.996 | 99.2% |
| 3-class | DenseNet121 | 79.0% | 0.82 | 92.0% |
| Extended | DenseNet201 | 83.6% | 0.81 | 95.6% |

PulmoScan AI uses **EfficientNetB0** for the best speed/accuracy tradeoff. Retrain with `ml_training/train.py` for dataset-specific results.

---

## Tech Stack

- **Backend:** Python, FastAPI, TensorFlow/Keras, OpenCV
- **Model:** EfficientNetB0 + custom classification head
- **Frontend:** Vanilla JS, modern CSS (no build step)
- **Deploy:** Docker, Uvicorn

---

## Roadmap

- [ ] Expand to full 4-class dataset (Pneumonia + TB from Kaggle)
- [ ] Ensemble models (EfficientNet + DenseNet + InceptionResNetV2)
- [ ] DICOM file support
- [ ] CT scan-specific model
- [ ] Mobile app (React Native)
- [ ] Cloud deployment (AWS/GCP)
- [ ] FHIR integration for hospital systems

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
