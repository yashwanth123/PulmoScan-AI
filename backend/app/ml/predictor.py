"""Inference service for PulmoScan AI."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import tensorflow as tf

from backend.app.config import BINARY_CLASS_NAMES, CLASS_NAMES, IMG_SIZE, MODEL_PATH, SCAN_TYPES
from backend.app.ml.gradcam import generate_gradcam
from backend.app.ml.model import build_model, compile_model
from backend.app.ml.preprocessing import build_tta_batch, load_image_from_bytes, preprocess_for_model

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
    demo_mode: bool = False
    tta_enabled: bool = True
    gradcam_enabled: bool = False
    reliability: str = "demo"
    reliability_message: str = ""
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
            "gradcam_enabled": self.gradcam_enabled,
            "model_loaded": self.model_loaded,
            "demo_mode": self.demo_mode,
            "tta_enabled": self.tta_enabled,
            "reliability": self.reliability,
            "reliability_message": self.reliability_message,
            "timestamp": self.timestamp,
        }


def _compute_reliability(confidence: float, model_loaded: bool) -> tuple[str, str]:
    """Tell the user how much to trust this specific prediction."""
    if not model_loaded:
        return (
            "demo",
            "Model not trained — this prediction is a demo only and is often incorrect.",
        )
    if confidence >= 0.85:
        return (
            "high",
            "High confidence — the model is fairly sure. Still verify with a doctor.",
        )
    if confidence >= 0.65:
        return (
            "medium",
            "Moderate confidence — result may be correct but should be confirmed.",
        )
    return (
        "low",
        "Low confidence — prediction is unreliable. Try a trained model or another image.",
    )


def _risk_and_recommendation(
    label: str,
    confidence: float,
    demo_mode: bool = False,
) -> tuple[str, str]:
    """Map diagnosis to risk level and clinical recommendation text."""
    if demo_mode and confidence < 0.55:
        return "uncertain", (
            "Demo mode — model not fine-tuned on lung data. "
            "Run `python ml_training/train.py --quick` for reliable results."
        )

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

    if demo_mode:
        rec += " Note: Train the model with `python ml_training/train.py` for clinical-grade accuracy."

    return risk, rec


def _load_display_labels(num_outputs: int) -> list[str]:
    """Load saved training labels or fall back to defaults."""
    label_file = MODEL_PATH.parent / "class_labels.json"
    if label_file.exists():
        import json
        data = json.loads(label_file.read_text())
        labels = data.get("display_labels", [])
        if len(labels) == num_outputs:
            return labels
    if num_outputs == len(BINARY_CLASS_NAMES):
        return list(BINARY_CLASS_NAMES)
    if num_outputs == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [f"Class_{i}" for i in range(num_outputs)]


def _resolve_class_names(num_outputs: int) -> list[str]:
    """Map model output dimension to human-readable labels."""
    return _load_display_labels(num_outputs)


def load_model(force: bool = False) -> tf.keras.Model:
    """Load or build the classification model (thread-safe)."""
    global _model
    with _lock:
        if _model is not None and not force:
            return _model

        if MODEL_PATH.exists():
            logger.info("Loading trained model from %s", MODEL_PATH)
            _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        else:
            logger.warning(
                "No trained weights at %s — binary demo model (COVID-19 vs Normal). "
                "Run `python ml_training/train.py` for accurate predictions.",
                MODEL_PATH,
            )
            _model = build_model(num_classes=len(BINARY_CLASS_NAMES))
            compile_model(_model)

        return _model


def get_model_status() -> dict[str, Any]:
    """Return model availability info."""
    trained = MODEL_PATH.exists()
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": trained,
        "classes": CLASS_NAMES if trained else BINARY_CLASS_NAMES,
        "architecture": "EfficientNetB0 + TTA + CLAHE preprocessing",
        "input_size": list(IMG_SIZE),
        "demo_mode": not trained,
        "tta_enabled": True,
    }


def _run_inference(model: tf.keras.Model, img, scan_type: str) -> np.ndarray:
    """Run test-time augmented inference and return averaged probabilities."""
    batch = build_tta_batch(img, scan_type=scan_type)
    predictions = model.predict(batch, verbose=0)
    return predictions.mean(axis=0)


def predict(
    image_bytes: bytes,
    scan_type: str = "chest_xray",
    include_gradcam: bool = True,
) -> PredictionResult:
    """Run full inference pipeline on uploaded scan."""
    model = load_model()
    demo_mode = not MODEL_PATH.exists()
    img = load_image_from_bytes(image_bytes)

    probs = _run_inference(model, img, scan_type)
    num_outputs = len(probs)
    active_classes = _resolve_class_names(num_outputs)

    prob_map = {name: 0.0 for name in CLASS_NAMES}
    for i, name in enumerate(active_classes):
        if name in prob_map:
            prob_map[name] = float(probs[i])
        elif i < len(probs):
            prob_map[CLASS_NAMES[i] if i < len(CLASS_NAMES) else name] = float(probs[i])

    idx = int(np.argmax(probs))
    label = active_classes[idx]
    confidence = float(probs[idx])
    reliability, reliability_msg = _compute_reliability(confidence, MODEL_PATH.exists())
    risk, rec = _risk_and_recommendation(label, confidence, demo_mode=demo_mode)

    tensor = preprocess_for_model(img, scan_type=scan_type)
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
        gradcam_enabled=include_gradcam and gradcam_b64 is not None,
        model_loaded=MODEL_PATH.exists(),
        demo_mode=demo_mode,
        tta_enabled=True,
        reliability=reliability,
        reliability_message=reliability_msg,
    )

    _record_prediction(result, scan_type)
    return result


def reset_runtime_state() -> None:
    """Clear in-memory stats and history (used in tests)."""
    global _prediction_history, _stats
    _prediction_history = []
    _stats = {
        "total_scans": 0,
        "by_class": {c: 0 for c in CLASS_NAMES},
        "by_scan_type": {},
    }


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
