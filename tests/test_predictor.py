"""Tests for predictor logic and inference."""

from __future__ import annotations

import pytest

from backend.app.ml.predictor import (
    PredictionResult,
    _risk_and_recommendation,
    predict,
    reset_runtime_state,
)


@pytest.fixture(autouse=True)
def _clean_state():
    reset_runtime_state()
    yield
    reset_runtime_state()


def test_risk_high_for_covid() -> None:
    risk, rec = _risk_and_recommendation("COVID-19", 0.92)
    assert risk == "high"
    assert "COVID-19" in rec


def test_risk_low_for_normal() -> None:
    risk, rec = _risk_and_recommendation("Normal", 0.95)
    assert risk == "low"
    assert "checkup" in rec.lower()


def test_risk_uncertain_for_low_confidence_normal() -> None:
    risk, _ = _risk_and_recommendation("Normal", 0.55)
    assert risk == "uncertain"


def test_prediction_result_to_dict() -> None:
    result = PredictionResult(
        diagnosis="Normal",
        confidence=0.91,
        probabilities={"Normal": 0.91, "COVID-19": 0.09},
        scan_type="Chest X-Ray",
        risk_level="low",
        recommendation="OK",
    )
    data = result.to_dict()
    assert data["diagnosis"] == "Normal"
    assert data["confidence"] == 0.91


@pytest.mark.slow
def test_predict_returns_valid_result(sample_xray_bytes: bytes) -> None:
    result = predict(sample_xray_bytes, scan_type="chest_xray", include_gradcam=False)
    assert result.diagnosis in {"COVID-19", "Normal", "Pneumonia", "Tuberculosis"}
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) >= 0.0
