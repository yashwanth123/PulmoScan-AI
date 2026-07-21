"""Load saved evaluation metrics from training."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import BASE_DIR, MODEL_PATH

EVAL_REPORT = BASE_DIR / "logs" / "evaluation_report.json"


def load_evaluation_metrics() -> dict | None:
    """Return evaluation report if training has been run."""
    if not EVAL_REPORT.exists():
        return None
    try:
        return json.loads(EVAL_REPORT.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def get_model_performance_summary() -> dict:
    """Summarize model readiness and measured accuracy for the UI."""
    trained = MODEL_PATH.exists()
    metrics = load_evaluation_metrics()

    summary = {
        "model_trained": trained,
        "evaluation_available": metrics is not None,
        "test_accuracy": None,
        "test_auc": None,
        "test_f1_macro": None,
        "classes_evaluated": [],
        "reliability_guide": _reliability_guide(trained, metrics),
    }

    if metrics:
        summary.update({
            "test_accuracy": round(metrics.get("accuracy", 0) * 100, 1),
            "test_auc": round(metrics.get("auc", 0) * 100, 1),
            "test_f1_macro": round(
                metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0) * 100,
                1,
            ),
            "classes_evaluated": metrics.get("classes", []),
            "confusion_matrix": metrics.get("confusion_matrix"),
        })

    return summary


def _reliability_guide(trained: bool, metrics: dict | None) -> list[str]:
    """Steps shown in UI so users know how to trust predictions."""
    steps = []

    if not trained:
        steps.append("Model is NOT trained yet — current results are demo-only and often wrong.")
        steps.append("Train first: python ml_training/train.py --quick")
        steps.append("After training, run: python ml_training/evaluate_cli.py")
    else:
        steps.append("Model is trained on lung X-ray data (COVID-19 vs Normal).")
        if metrics:
            acc = metrics.get("accuracy", 0) * 100
            steps.append(f"Measured test accuracy: {acc:.1f}% on held-out images.")
        steps.append("Use sample images with known labels — UI will show Match / Mismatch.")
        steps.append("Confidence ≥ 85% = more trustworthy; below 60% = treat as uncertain.")

    steps.append("Chest MRI is not supported yet — only X-Ray and CT upload work.")
    steps.append("Always confirm with a radiologist — this is a research tool.")
    return steps
