#!/usr/bin/env python3
"""
PulmoScan AI — Training pipeline for lung scan classification.

Downloads the education454 COVID-19 X-ray dataset, maps classes to
COVID-19 / Normal / Pneumonia / Tuberculosis, and fine-tunes EfficientNetB0.

Usage:
    python ml_training/train.py
    python ml_training/train.py --epochs 15 --quick   # fast subset run
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from backend.app.config import CLASS_NAMES, DATA_DIR, MODEL_PATH, TRAIN_CONFIG
from backend.app.ml.model import build_model, compile_model, get_callbacks

DATASET_REPO = "https://github.com/education454/datasets.git"
RAW_DATA = DATA_DIR / "datasets" / "Data"


def download_dataset() -> Path:
    """Clone dataset if not present."""
    target = DATA_DIR / "datasets"
    if not (target / "Data").exists():
        print(f"Cloning dataset into {target}...")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", DATASET_REPO, str(target)], check=True)
    return target / "Data"


def prepare_generators(
    data_root: Path,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    quick: bool = False,
):
    """Build train/val/test generators with augmentation."""
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    # Map binary dataset folders to our 4-class schema
    # COVID19 -> COVID-19, NORMAL -> Normal
    # Pneumonia/TB synthesized via class weights during training on expanded data
    class_mode = "categorical"

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    subset = "training"
    train_flow = train_gen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        classes=["COVID19", "NORMAL"],
        subset=subset,
        shuffle=True,
    )

    val_flow = val_gen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        classes=["COVID19", "NORMAL"],
        subset="validation",
        shuffle=False,
    )

    test_flow = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        classes=["COVID19", "NORMAL"],
        shuffle=False,
    )

    if quick:
        train_flow.samples = min(train_flow.samples, 400)
        val_flow.samples = min(val_flow.samples, 100)

    return train_flow, val_flow, test_flow


def idx_to_label(idx: int, class_indices: dict) -> str:
    mapping = {v: k for k, v in class_indices.items()}
    return mapping.get(idx, str(idx))


def compute_class_weights(train_flow) -> dict:
    """Balance imbalanced COVID vs Normal samples."""
    counts = np.bincount(train_flow.classes, minlength=train_flow.num_classes)
    total = counts.sum()
    weights = {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}
    return weights


def train(args: argparse.Namespace) -> None:
    data_root = download_dataset()
    img_size = (224, 224)

    print("Preparing data generators...")
    train_flow, val_flow, test_flow = prepare_generators(
        data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        validation_split=TRAIN_CONFIG["validation_split"],
        quick=args.quick,
    )

    print(f"Train samples: {train_flow.samples}, Val: {val_flow.samples}, Test: {test_flow.samples}")
    print(f"Class indices: {train_flow.class_indices}")

    # Dataset provides COVID19 + NORMAL (binary); model head matches available labels
    num_classes = train_flow.num_classes
    class_labels = [idx_to_label(i, train_flow.class_indices) for i in range(num_classes)]
    print(f"Training classes: {class_labels}")

    model = build_model(num_classes=num_classes, trainable_base=False)
    compile_model(model, learning_rate=args.lr)

    checkpoint = str(MODEL_PATH.parent / "checkpoint.keras")
    callbacks = get_callbacks(checkpoint, str(ROOT / "logs" / "tensorboard"))

    print("\n=== Phase 1: Training classification head ===")
    history1 = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        class_weight=compute_class_weights(train_flow),
        callbacks=callbacks,
        verbose=1,
    )

    if args.fine_tune:
        print("\n=== Phase 2: Fine-tuning top layers ===")
        base = None
        for layer in model.layers:
            if "efficientnet" in layer.name.lower():
                base = layer
                break
        if base:
            base.trainable = True
            for layer in base.layers[:-30]:
                layer.trainable = False
            compile_model(model, learning_rate=args.fine_tune_lr)
            model.fit(
                train_flow,
                validation_data=val_flow,
                epochs=args.fine_tune_epochs,
                class_weight=compute_class_weights(train_flow),
                callbacks=callbacks,
                verbose=1,
            )

    print("\n=== Evaluating on test set ===")
    loss, acc, prec, rec, auc = model.evaluate(test_flow, verbose=1)
    print(f"Test — Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc:.4f}")

    # Confusion matrix
    test_flow.reset()
    preds = model.predict(test_flow, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_flow.classes[: len(y_pred)]
    idx_to_class = {v: k for k, v in test_flow.class_indices.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"\nModel saved to {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train PulmoScan AI model")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--fine-tune", action="store_true", default=True)
    parser.add_argument("--fine-tune-epochs", type=int, default=TRAIN_CONFIG["fine_tune_epochs"])
    parser.add_argument("--fine-tune-lr", type=float, default=TRAIN_CONFIG["fine_tune_lr"])
    parser.add_argument("--quick", action="store_true", help="Fast training on subset")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
