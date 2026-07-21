"""Shared pytest fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_xray_bytes() -> bytes:
    """Generate a synthetic 224x224 grayscale chest-xray-like PNG."""
    img = Image.new("L", (224, 224), color=40)
    pixels = img.load()
    for x in range(60, 164):
        for y in range(40, 180):
            pixels[x, y] = 180
    rgb = img.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_xray_path(tmp_path: Path, sample_xray_bytes: bytes) -> Path:
    """Write synthetic X-ray image to a temp file."""
    path = tmp_path / "sample_xray.png"
    path.write_bytes(sample_xray_bytes)
    return path
