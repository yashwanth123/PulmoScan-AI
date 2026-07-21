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
        "description": "MRI lung imaging (coming soon)",
        "supported": False,
        "color": "#64748b",
    },
}

# Public sample images for demo/testing (Wikimedia & open datasets)
SAMPLE_IMAGES = {
    "chest_xray": [
        {
            "id": "xray_normal",
            "name": "Normal Chest X-Ray",
            "label_hint": "Normal",
            "url": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Chest_Xray_PA_3-8-2010.png",
        },
        {
            "id": "xray_pa",
            "name": "PA Radiograph",
            "label_hint": "Screening",
            "url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Pneumonia_X-ray.jpg",
        },
    ],
    "ct_scan": [
        {
            "id": "ct_chest",
            "name": "Chest CT Axial Slice",
            "label_hint": "CT Analysis",
            "url": "https://upload.wikimedia.org/wikipedia/commons/4/4e/CT_of_a_normal_Thorax%2C_axial_plane_%2830%29.jpg",
        },
        {
            "id": "ct_lung",
            "name": "Lung CT Cross-section",
            "label_hint": "High-res CT",
            "url": "https://upload.wikimedia.org/wikipedia/commons/6/6e/CT_scan_of_the_Thorax_%28axial%29_%281%29.jpg",
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
}
