"""Inference service for PulmoScan AI."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from backend.app.config import CLASS_NAMES, MODEL_PATH, SCAN_TYPES
from backend.app.ml.gradcam import generate_gradcam
from backend.app.ml.model import build_model, compile_model
from backend.app.ml.preprocessing import load_image_from_bytes, preprocess_for_model

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: tf.keras.Model | None = None
_prediction_history: list[dict[str, Any]] = []
_stats = {"total_scans": 0, "by_class": {c: 0 for c in CLASS_NAMES}, "by_scan_type": {}}


@dataclass
class PredictionResult:
    """Structured prediction output."""

    diagnosis: str
    confidence: float
    probabilities: dict[str, float]
    scan_type: str
    risk_level: str
    recommendation: str
    gradcam_image: str | None = None
    model_loaded: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "scan_type": self.scan_type,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "gradcam_image": self.gradcam_image,
            "model_loaded": self.model_loaded,
            "timestamp": self.timestamp,
        }


def _risk_and_recommendation(label: str, confidence: float) -> tuple[str, str]:
    """Map diagnosis to risk level and clinical recommendation text."""
    high_risk = {"COVID-19", "Pneumonia", "Tuberculosis"}
    if label in high_risk:
        if confidence >= 0.85:
            risk = "high"
            rec = (
                f"Model indicates possible {label} with high confidence. "
                "Seek immediate medical evaluation and confirm with RT-PCR / clinical assessment."
            )
        elif confidence >= 0.6:
            risk = "moderate"
            rec = (
                f"Possible {label} detected. Consult a pulmonologist and consider "
                "additional diagnostic tests (RT-PCR, sputum culture, CT if indicated)."
            )
        else:
            risk = "low-moderate"
            rec = (
                f"Weak signal for {label}. Results are inconclusive — "
                "clinical correlation and repeat imaging may be needed."
            )
    else:
        if confidence >= 0.8:
            risk = "low"
            rec = "No significant abnormality detected. Maintain routine health checkups."
        else:
            risk = "uncertain"
            rec = (
                "Classification uncertain. Please consult a radiologist "
                "for professional interpretation."
            )
    return risk, rec


def load_model(force: bool = False) -> tf.keras.Model:
    """Load or build the classification model (thread-safe)."""
    global _model
    with _lock:
        if _model is not None and not force:
            return _model

        if MODEL_PATH.exists():
            logger.info("Loading trained model from %s", MODEL_PATH)
            _model = tf.keras.models.load_model(str(MODEL_PATH))
        else:
            logger.warning(
                "No trained weights at %s — using ImageNet-pretrained backbone. "
                "Run `python ml_training/train.py` for accurate predictions.",
                MODEL_PATH,
            )
            _model = build_model()
            compile_model(_model)

        return _model


def get_model_status() -> dict[str, Any]:
    """Return model availability info."""
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "classes": CLASS_NAMES,
        "architecture": "EfficientNetB0 + custom head",
        "input_size": list(preprocess_for_model(load_image_from_bytes(
            _placeholder_png_bytes()
        )).shape[1:3]),
    }


def _placeholder_png_bytes() -> bytes:
    """Minimal 1x1 PNG for shape probing."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def predict(
    image_bytes: bytes,
    scan_type: str = "chest_xray",
    include_gradcam: bool = True,
) -> PredictionResult:
    """Run full inference pipeline on uploaded scan."""
    model = load_model()
    img = load_image_from_bytes(image_bytes)
    tensor = preprocess_for_model(img)

    probs = model.predict(tensor, verbose=0)[0]
    num_outputs = len(probs)
    active_classes = CLASS_NAMES[:num_outputs]
    # Pad remaining classes with zero probability for UI consistency
    prob_map = {name: 0.0 for name in CLASS_NAMES}
    for i, name in enumerate(active_classes):
        prob_map[name] = float(probs[i])

    idx = int(np.argmax(probs))
    label = active_classes[idx]
    confidence = float(probs[idx])
    risk, rec = _risk_and_recommendation(label, confidence)

    gradcam_b64 = None
    if include_gradcam:
        try:
            _, gradcam_b64 = generate_gradcam(model, tensor, idx)
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)

    scan_info = SCAN_TYPES.get(scan_type, SCAN_TYPES["chest_xray"])
    result = PredictionResult(
        diagnosis=label,
        confidence=confidence,
        probabilities=prob_map,
        scan_type=scan_info["name"],
        risk_level=risk,
        recommendation=rec,
        gradcam_image=gradcam_b64,
        model_loaded=MODEL_PATH.exists(),
    )

    _record_prediction(result, scan_type)
    return result


def _record_prediction(result: PredictionResult, scan_type: str) -> None:
    """Update in-memory stats and history."""
    _stats["total_scans"] += 1
    _stats["by_class"][result.diagnosis] = _stats["by_class"].get(result.diagnosis, 0) + 1
    _stats["by_scan_type"][scan_type] = _stats["by_scan_type"].get(scan_type, 0) + 1

    entry = result.to_dict()
    entry["scan_type_key"] = scan_type
    _prediction_history.insert(0, entry)
    if len(_prediction_history) > 50:
        _prediction_history.pop()


def get_stats() -> dict[str, Any]:
    """Dashboard statistics."""
    return {
        **_stats,
        "classes": CLASS_NAMES,
        "scan_types": SCAN_TYPES,
        "model_trained": MODEL_PATH.exists(),
    }


def get_history(limit: int = 20) -> list[dict[str, Any]]:
    """Recent prediction history."""
    return _prediction_history[:limit]
