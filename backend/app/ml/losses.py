"""Focal loss for imbalanced lung scan classification."""
from __future__ import annotations

import tensorflow as tf


def focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    """
    Focal loss — down-weights easy examples, focuses on hard minority cases.

    alpha=0.75 gives more weight to the positive/minority class (COVID).
    """

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))

    return loss_fn
