"""Tests for image preprocessing utilities."""

from __future__ import annotations

import numpy as np

from backend.app.ml.preprocessing import (
    load_image_from_bytes,
    preprocess_for_model,
    preprocess_from_bytes,
)


def test_load_image_from_bytes(sample_xray_bytes: bytes) -> None:
    img = load_image_from_bytes(sample_xray_bytes)
    assert img.mode == "RGB"
    assert img.size == (224, 224)


def test_preprocess_for_model_shape(sample_xray_bytes: bytes) -> None:
    img = load_image_from_bytes(sample_xray_bytes)
    tensor = preprocess_for_model(img)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_from_bytes_pipeline(sample_xray_bytes: bytes) -> None:
    tensor = preprocess_from_bytes(sample_xray_bytes)
    assert tensor.ndim == 4
    assert tensor.shape[0] == 1
