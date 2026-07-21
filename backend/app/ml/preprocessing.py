"""Image preprocessing for lung scan inference."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

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


def apply_clahe(pil_img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Contrast Limited Adaptive Histogram Equalization for medical images."""
    lab = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
    channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(channel)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def enhance_medical_image(pil_img: Image.Image, scan_type: str = "chest_xray") -> Image.Image:
    """Apply scan-type-specific enhancement before model inference."""
    img = apply_clahe(pil_img)

    if scan_type == "ct_scan":
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.1)
    else:
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.02)

    return img


def preprocess_for_model(
    img: Image.Image,
    scan_type: str = "chest_xray",
    augment_flip: bool = False,
) -> np.ndarray:
    """Convert PIL image to model-ready batch tensor with medical preprocessing."""
    img = enhance_medical_image(img, scan_type=scan_type)
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    if augment_flip:
        arr = np.fliplr(arr)

    return np.expand_dims(arr, axis=0)


def build_tta_batch(img: Image.Image, scan_type: str = "chest_xray") -> np.ndarray:
    """Build test-time augmentation batch: original + horizontal flip."""
    original = preprocess_for_model(img, scan_type=scan_type, augment_flip=False)
    flipped = preprocess_for_model(img, scan_type=scan_type, augment_flip=True)
    return np.vstack([original, flipped])


def preprocess_from_bytes(data: bytes, scan_type: str = "chest_xray") -> np.ndarray:
    """Full pipeline: bytes -> model tensor."""
    return preprocess_for_model(load_image_from_bytes(data), scan_type=scan_type)


def preprocess_from_path(path: str | Path, scan_type: str = "chest_xray") -> np.ndarray:
    """Full pipeline: path -> model tensor."""
    return preprocess_for_model(load_image_from_path(path), scan_type=scan_type)
