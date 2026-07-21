"""Tests for dataset utilities (no TensorFlow training)."""

from __future__ import annotations

import numpy as np

from ml_training.dataset import compute_class_weights, idx_to_label


def test_idx_to_label() -> None:
    mapping = {"COVID19": 0, "NORMAL": 1}
    assert idx_to_label(0, mapping) == "COVID19"
    assert idx_to_label(1, mapping) == "NORMAL"


def test_compute_class_weights_balanced() -> None:
    class FakeGenerator:
        classes = np.array([0, 0, 1, 1])
        num_classes = 2

    weights = compute_class_weights(FakeGenerator())
    assert weights[0] == weights[1] == 1.0


def test_compute_class_weights_imbalanced() -> None:
    class FakeGenerator:
        classes = np.array([0, 0, 0, 1])
        num_classes = 2

    weights = compute_class_weights(FakeGenerator())
    assert weights[0] < weights[1]
