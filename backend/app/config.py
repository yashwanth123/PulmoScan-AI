"""Application configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model settings
IMG_SIZE = (224, 224)
CLASS_NAMES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_FILENAME = "pulmoscan_efficientnet.keras"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Scan types supported in UI
SCAN_TYPES = {
    "chest_xray": {
        "name": "Chest X-Ray",
        "icon": "xray",
        "description": "PA/AP chest radiograph analysis",
        "supported": True,
        "color": "#00e5ff",
    },
    "ct_scan": {
        "name": "CT Scan",
        "icon": "ct",
        "description": "Axial chest CT slice analysis",
        "supported": True,
        "color": "#a855f7",
    },
    "mri": {
        "name": "Chest MRI",
        "icon": "mri",
        "description": "Not available yet — upload not supported",
        "supported": False,
        "color": "#64748b",
        "status_note": "Planned for a future release. Use Chest X-Ray or CT Scan.",
    },
}

# Public sample images — run: python3 scripts/setup_samples.py
SAMPLE_IMAGES = {
    "chest_xray": [
        {
            "id": "xray_normal",
            "name": "Normal Chest X-Ray",
            "label_hint": "Expected: Normal",
            "expected_label": "Normal",
            "url": "local",
        },
        {
            "id": "xray_covid",
            "name": "COVID-19 X-Ray",
            "label_hint": "Expected: COVID-19",
            "expected_label": "COVID-19",
            "url": "local",
        },
        {
            "id": "xray_pneumonia",
            "name": "Abnormal X-Ray (COVID sample)",
            "label_hint": "Expected: COVID-19",
            "expected_label": "COVID-19",
            "url": "local",
        },
    ],
    "ct_scan": [
        {
            "id": "ct_chest",
            "name": "Chest Scan Sample",
            "label_hint": "Expected: Normal",
            "expected_label": "Normal",
            "url": "local",
        },
        {
            "id": "ct_lung",
            "name": "Lung Scan Sample",
            "label_hint": "Expected: Normal",
            "expected_label": "Normal",
            "url": "local",
        },
    ],
}

# Default binary classes when model is not yet fine-tuned
BINARY_CLASS_NAMES = ["COVID-19", "Normal"]

# Training defaults
TRAIN_CONFIG = {
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "fine_tune_epochs": 10,
    "fine_tune_lr": 1e-5,
    "validation_split": 0.2,
    "minority_boost": 2.5,
}
