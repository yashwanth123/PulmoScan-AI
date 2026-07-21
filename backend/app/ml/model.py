"""Transfer-learning model builder for lung scan classification."""
from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from backend.app.config import IMG_SIZE, NUM_CLASSES

logger = logging.getLogger(__name__)


def build_model(
    num_classes: int = NUM_CLASSES,
    trainable_base: bool = False,
    dropout: float = 0.4,
) -> tf.keras.Model:
    """Build EfficientNetB0-based classifier (ImageNet pretrained)."""
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = trainable_base

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="pulmoscan_efficientnet")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    """Compile model with standard classification metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_callbacks(
    checkpoint_path: str,
    log_dir: str | None = None,
    use_validation: bool = True,
) -> list:
    """Training callbacks. TensorBoard is optional if package is not installed."""
    monitor_metric = "val_accuracy" if use_validation else "accuracy"
    monitor_loss = "val_loss" if use_validation else "loss"

    callbacks: list = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_loss,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    if log_dir:
        try:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        except Exception as exc:
            logger.warning("TensorBoard disabled (%s). Training continues without it.", exc)

    return callbacks
