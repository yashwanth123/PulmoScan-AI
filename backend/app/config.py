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
    },
    "ct_scan": {
        "name": "CT Scan",
        "icon": "ct",
        "description": "Chest CT slice analysis (experimental)",
        "supported": True,
    },
    "mri": {
        "name": "Chest MRI",
        "icon": "mri",
        "description": "MRI lung imaging (coming soon)",
        "supported": False,
    },
}

# Training defaults
TRAIN_CONFIG = {
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "fine_tune_epochs": 10,
    "fine_tune_lr": 1e-5,
    "validation_split": 0.2,
}
