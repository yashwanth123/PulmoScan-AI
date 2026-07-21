"""Dataset download and data pipeline for lung scan training."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from backend.app.config import DATA_DIR

DATASET_REPO = "https://github.com/education454/datasets.git"
DATASET_CLASSES = ["COVID19", "NORMAL"]
IMG_SIZE = (224, 224)


def get_dataset_root() -> Path:
    """Return path to raw dataset root (`.../Data`)."""
    return DATA_DIR / "datasets" / "Data"


def download_dataset(force: bool = False) -> Path:
    """Clone the public COVID-19 X-ray dataset if missing."""
    target = DATA_DIR / "datasets"
    data_root = target / "Data"

    if data_root.exists() and not force:
        return data_root

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and force:
        import shutil
        shutil.rmtree(target)

    subprocess.run(
        ["git", "clone", "--depth", "1", DATASET_REPO, str(target)],
        check=True,
    )
    return data_root


def idx_to_label(index: int, class_indices: dict[str, int]) -> str:
    """Convert numeric class index to folder label."""
    mapping = {value: key for key, value in class_indices.items()}
    return mapping.get(index, str(index))


def compute_class_weights(generator, boost_minority: float = 2.5) -> dict[int, float]:
    """
    Inverse-frequency weights with extra boost for the minority (COVID) class.

    Without boost, the model often predicts all NORMAL (~65% accuracy but 0% COVID recall).
    """
    counts = np.bincount(generator.classes, minlength=generator.num_classes)
    total = counts.sum()
    weights = {
        index: total / (len(counts) * count)
        for index, count in enumerate(counts)
        if count > 0
    }
    if len(counts) >= 2:
        minority_idx = int(np.argmin(counts))
        weights[minority_idx] = weights.get(minority_idx, 1.0) * boost_minority
    return weights


def build_generators(
    data_root: Path,
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2,
    quick: bool = False,
) -> tuple:
    """
    Create train, validation, and test image generators.

    Returns:
        (train_flow, val_flow, test_flow)
    """
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found at {data_root}. Run: python scripts/download_data.py"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    # Same validation_split required — without it Keras returns 0 validation images
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=DATASET_CLASSES,
    )

    train_flow = train_datagen.flow_from_directory(
        str(train_dir),
        subset="training",
        shuffle=True,
        **common,
    )
    val_flow = val_datagen.flow_from_directory(
        str(train_dir),
        subset="validation",
        shuffle=False,
        **common,
    )
    test_flow = test_datagen.flow_from_directory(
        str(test_dir),
        shuffle=False,
        **common,
    )

    return train_flow, val_flow, test_flow
