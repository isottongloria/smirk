"""Small, dependency-light helpers for composing demo video frames."""

import cv2
import numpy as np


def compose_render_overlay(original, rendered, mask, alpha=0.55, color=(70, 190, 255)):
    """Blend a colorized render over an RGB image.

    ``original`` and ``rendered`` must be HxWx3 uint8 RGB arrays and ``mask`` an
    HxW boolean (or 0/1) array. The result is RGB uint8. Render luminance
    modulates ``color`` so that mesh lighting and surface detail remain visible.
    """
    if original.dtype != np.uint8 or rendered.dtype != np.uint8:
        raise TypeError('original and rendered must be uint8 arrays in RGB order')
    if original.shape != rendered.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('original and rendered must have the same HxWx3 shape')
    mask = np.asarray(mask)
    if mask.shape != original.shape[:2]:
        raise ValueError('mask must have shape HxW')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be between 0 and 1')
    color = np.asarray(color, dtype=np.float32)
    if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
        raise ValueError('color must contain three values between 0 and 255')

    intensity = rendered.astype(np.float32).mean(axis=2, keepdims=True) / 255.0
    colored_render = intensity * color.reshape(1, 1, 3)
    weight = alpha * mask.astype(np.float32)[..., None]
    blended = original.astype(np.float32) * (1.0 - weight) + colored_render * weight
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def compose_overlay_layout(original, rendered, mask, alpha=0.55, color=(70, 190, 255)):
    """Return ``original | overlay | isolated render`` as an RGB uint8 array."""
    overlay = compose_render_overlay(original, rendered, mask, alpha, color)
    return np.concatenate([original, overlay, rendered], axis=1)


def rgb_to_opencv_bgr(image):
    """Convert a final RGB layout to the BGR channel order expected by OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
