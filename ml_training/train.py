#!/usr/bin/env python3
"""Train PulmoScan AI lung scan classifier."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.config import MODEL_PATH, TRAIN_CONFIG
from backend.app.ml.model import build_model, compile_model, get_callbacks
from ml_training.dataset import (
    build_generators,
    compute_class_weights,
    download_dataset,
    idx_to_label,
)
from ml_training.evaluate import evaluate_model, print_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pulmoscan.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNetB0 for lung scan classification",
    )
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--fine-tune", action="store_true", default=True)
    parser.add_argument("--no-fine-tune", dest="fine_tune", action="store_false")
    parser.add_argument("--fine-tune-epochs", type=int, default=TRAIN_CONFIG["fine_tune_epochs"])
    parser.add_argument("--fine-tune-lr", type=float, default=TRAIN_CONFIG["fine_tune_lr"])
    parser.add_argument("--quick", action="store_true", help="Train on a small subset for smoke tests")
    return parser.parse_args()


def fine_tune_backbone(model, learning_rate: float) -> None:
    """Unfreeze top EfficientNet layers for phase-2 training."""
    backbone = next(
        (layer for layer in model.layers if "efficientnet" in layer.name.lower()),
        None,
    )
    if backbone is None:
        logger.warning("EfficientNet backbone not found; skipping fine-tune phase")
        return

    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    compile_model(model, learning_rate=learning_rate)


def train() -> None:
    args = parse_args()
    data_root = download_dataset()

    logger.info("Building data generators from %s", data_root)
    train_flow, val_flow, test_flow = build_generators(
        data_root,
        batch_size=args.batch_size,
        validation_split=TRAIN_CONFIG["validation_split"],
        quick=args.quick,
    )

    num_classes = train_flow.num_classes
    class_labels = [idx_to_label(i, train_flow.class_indices) for i in range(num_classes)]
    logger.info(
        "Samples — train: %d, val: %d, test: %d | classes: %s",
        train_flow.samples,
        val_flow.samples,
        test_flow.samples,
        class_labels,
    )

    model = build_model(num_classes=num_classes, trainable_base=False)
    compile_model(model, learning_rate=args.lr)

    checkpoint = MODEL_PATH.parent / "checkpoint.keras"
    log_dir = ROOT / "logs" / "tensorboard"
    callbacks = get_callbacks(str(checkpoint), str(log_dir))

    logger.info("Phase 1 — training classification head (%d epochs)", args.epochs)
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        class_weight=compute_class_weights(train_flow),
        callbacks=callbacks,
        verbose=1,
    )

    if args.fine_tune:
        logger.info("Phase 2 — fine-tuning backbone (%d epochs)", args.fine_tune_epochs)
        fine_tune_backbone(model, args.fine_tune_lr)
        model.fit(
            train_flow,
            validation_data=val_flow,
            epochs=args.fine_tune_epochs,
            class_weight=compute_class_weights(train_flow),
            callbacks=callbacks,
            verbose=1,
        )

    logger.info("Evaluating on held-out test set")
    metrics = evaluate_model(
        model,
        test_flow,
        output_path=ROOT / "logs" / "evaluation_report.json",
    )
    print_evaluation(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    train()
