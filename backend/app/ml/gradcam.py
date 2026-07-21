"""Grad-CAM explainability for lung scan predictions."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """Locate the last convolutional layer for Grad-CAM (searches nested models)."""
    conv_layers: list[str] = []

    def _walk(m: tf.keras.Model, prefix: str = "") -> None:
        for layer in m.layers:
            name = f"{prefix}{layer.name}"
            if isinstance(layer, tf.keras.Model):
                _walk(layer, prefix=f"{name}/")
            elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                conv_layers.append(layer.name if not prefix else layer.name)

    _walk(model)
    if not conv_layers:
        raise ValueError("No convolutional layer found for Grad-CAM")
    return conv_layers[-1]


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int,
    alpha: float = 0.45,
) -> tuple[np.ndarray, str]:
    """
    Generate Grad-CAM heatmap overlaid on input image.

    Returns (overlay_rgb_array, base64_png_string).
    """
    last_conv_name = _find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs, model.get_layer(last_conv_name).output],
        model.output,
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to image size
    h, w = img_array.shape[1], img_array.shape[2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original = np.uint8(img_array[0])
    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)

    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return overlay, f"data:image/png;base64,{b64}"
