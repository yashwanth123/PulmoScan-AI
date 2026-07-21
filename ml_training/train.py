#!/usr/bin/env python3
"""Train PulmoScan AI lung scan classifier."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

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

# Human-readable labels saved alongside model for inference
LABEL_MAP = {"COVID19": "COVID-19", "NORMAL": "Normal"}


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
    parser.add_argument("--quick", action="store_true", help="Fewer epochs only — still uses full dataset")
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
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    compile_model(model, learning_rate=learning_rate, use_focal_loss=True)


def save_label_map(class_indices: dict, path: Path) -> None:
    """Persist folder-name → display label mapping for inference."""
    ordered = sorted(class_indices.items(), key=lambda x: x[1])
    labels = [LABEL_MAP.get(name, name) for name, _ in ordered]
    path.write_text(json.dumps({"class_indices": class_indices, "display_labels": labels}, indent=2))
    logger.info("Label map saved to %s", path)


def train() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 15)
        args.fine_tune = False
        logger.info("Quick mode: %d epochs, no fine-tune, FULL dataset for class balance", args.epochs)

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
    class_weights = compute_class_weights(train_flow, boost_minority=2.5)
    logger.info(
        "Samples — train: %d, val: %d, test: %d | classes: %s | weights: %s",
        train_flow.samples,
        val_flow.samples,
        test_flow.samples,
        class_labels,
        class_weights,
    )

    if val_flow.samples == 0:
        logger.warning("Validation set is empty — using training metrics for checkpoints")

    model = build_model(num_classes=num_classes, trainable_base=False)
    compile_model(model, learning_rate=args.lr, use_focal_loss=True)

    checkpoint = MODEL_PATH.parent / "checkpoint.keras"
    log_dir = ROOT / "logs" / "tensorboard"
    use_val = val_flow.samples > 0
    callbacks = get_callbacks(str(checkpoint), str(log_dir), use_validation=use_val)
    val_data = val_flow if use_val else None

    logger.info("Phase 1 — training head with focal loss (%d epochs)", args.epochs)
    model.fit(
        train_flow,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    if args.fine_tune:
        logger.info("Phase 2 — fine-tuning backbone (%d epochs)", args.fine_tune_epochs)
        fine_tune_backbone(model, args.fine_tune_lr)
        model.fit(
            train_flow,
            validation_data=val_data,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    # Load best checkpoint if saved during training
    if checkpoint.exists():
        logger.info("Loading best checkpoint from %s", checkpoint)
        model = tf.keras.models.load_model(str(checkpoint), compile=False)
        compile_model(model, learning_rate=args.lr, use_focal_loss=True)

    logger.info("Evaluating on held-out test set")
    metrics = evaluate_model(
        model,
        test_flow,
        output_path=ROOT / "logs" / "evaluation_report.json",
    )
    print_evaluation(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    save_label_map(train_flow.class_indices, MODEL_PATH.parent / "class_labels.json")
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    train()
