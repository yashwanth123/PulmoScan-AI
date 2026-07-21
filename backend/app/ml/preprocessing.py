"""Image preprocessing for lung scan inference."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

from backend.app.config import IMG_SIZE


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load PIL image from raw bytes."""
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image_from_path(path: str | Path) -> Image.Image:
    """Load PIL image from filesystem path."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model-ready batch tensor."""
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_from_bytes(data: bytes) -> np.ndarray:
    """Full pipeline: bytes -> model tensor."""
    return preprocess_for_model(load_image_from_bytes(data))


def preprocess_from_path(path: str | Path) -> np.ndarray:
    """Full pipeline: path -> model tensor."""
    return preprocess_for_model(load_image_from_path(path))
