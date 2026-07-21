"""Model evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(
    model: tf.keras.Model,
    test_flow,
    output_path: Path | None = None,
) -> dict:
    """
    Evaluate a trained model on the test generator.

    Returns metrics dict and optionally writes JSON report to disk.
    """
    loss, accuracy, precision, recall, auc = model.evaluate(test_flow, verbose=0)

    test_flow.reset()
    probabilities = model.predict(test_flow, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_flow.classes[: len(y_pred)]

    index_to_class = {value: key for key, value in test_flow.class_indices.items()}
    target_names = [index_to_class[i] for i in sorted(index_to_class)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "classification_report": report,
        "confusion_matrix": matrix,
        "classes": target_names,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def print_evaluation(metrics: dict) -> None:
    """Pretty-print evaluation metrics to stdout."""
    print(
        f"Test — Loss: {metrics['loss']:.4f} | "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"Prec: {metrics['precision']:.4f} | "
        f"Rec: {metrics['recall']:.4f} | "
        f"AUC: {metrics['auc']:.4f}"
    )
    print("\nPer-class report:")
    for label in metrics["classes"]:
        stats = metrics["classification_report"].get(label, {})
        if isinstance(stats, dict):
            print(
                f"  {label:10s}  "
                f"P={stats.get('precision', 0):.3f}  "
                f"R={stats.get('recall', 0):.3f}  "
                f"F1={stats.get('f1-score', 0):.3f}"
            )
    print("\nConfusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(" ", row)
