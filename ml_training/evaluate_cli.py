#!/usr/bin/env python3
"""Evaluate a saved PulmoScan model on the test split."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

from backend.app.config import MODEL_PATH
from ml_training.dataset import build_generators, download_dataset
from ml_training.evaluate import evaluate_model, print_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pulmoscan.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PulmoScan AI model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .keras model")
    parser.add_argument("--output", type=Path, default=ROOT / "logs" / "evaluation_report.json")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Train first with: python ml_training/train.py")

    data_root = download_dataset()
    _, _, test_flow = build_generators(data_root)

    logger.info("Loading model from %s", args.model)
    model = tf.keras.models.load_model(str(args.model))

    metrics = evaluate_model(model, test_flow, output_path=args.output)
    print_evaluation(metrics)
    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
