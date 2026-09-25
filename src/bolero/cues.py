"""Depth-derived contextual cues, applied to an RGB image at a strength in [0, 1].

Ported from notebooks/01_bolero_exploration.ipynb (cells 11, 12, 15, 18) and made
strength-parameterized so each cue can be swept. Strength 0 is always identity.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from matplotlib import colormaps

from .depth import boundary_strength, normalize

_INFERNO = colormaps["inferno"]


def border_strengthening(image, depth, strength):
    """Brighten pixels in proportion to depth-gradient strength (notebook cell 11)."""
    return image * (1.0 + strength * boundary_strength(depth)[..., None])


def depth_boundary_emphasis(image, depth, strength):
    """Darken depth boundaries into an outline around the entity."""
    edge = cv2.GaussianBlur(boundary_strength(depth), (0, 0), 1.0)
    edge = normalize(edge, 0.0, 99.5)
    return image * (1.0 - strength * edge[..., None])


def entity_background_separation(image, depth, strength):
    """Attenuate weak-structure regions, keeping near/boundary regions (notebook cell 18)."""
    structure = 0.6 * normalize(depth, 5, 95) + 0.4 * boundary_strength(depth)
    structure = structure / (structure.max() + 1e-8)
    return image * ((1.0 - strength) + strength * structure[..., None])


def shadow_reinforcement(image, depth, strength):
    """Shade far regions darker, reinforcing the depth ordering."""
    return image * (1.0 - strength * (1.0 - depth)[..., None])


def contrast_luminance(image, depth, strength):
    """CLAHE on lightness, weighted toward the near region."""
    lab = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2LAB)
    l_u8 = np.round(lab[..., 0] * 255.0 / 100.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab_eq = lab.copy()
    lab_eq[..., 0] = clahe.apply(l_u8).astype(np.float32) * 100.0 / 255.0
    equalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    weight = strength * (0.5 + 0.5 * depth)[..., None]
    return (1.0 - weight) * image + weight * equalized


def thermal_injection(image, depth, strength):
    """Blend an inferno-colormapped depth field into the RGB image (notebook cell 15)."""
    thermal = _INFERNO(depth)[..., :3].astype(np.float32)
    return (1.0 - strength) * image + strength * thermal


CUES: dict[str, Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = {
    "border_strengthening": border_strengthening,
    "depth_boundary_emphasis": depth_boundary_emphasis,
    "entity_background_separation": entity_background_separation,
    "shadow_reinforcement": shadow_reinforcement,
    "contrast_luminance": contrast_luminance,
    "thermal_injection": thermal_injection,
}

# Conservative defaults used when a cue is applied as a restoration step.
DEFAULT_STRENGTH = {
    "border_strengthening": 0.5,
    "depth_boundary_emphasis": 0.3,
    "entity_background_separation": 0.3,
    "shadow_reinforcement": 0.3,
    "contrast_luminance": 0.5,
    "thermal_injection": 0.1,
}


def apply_cue(name: str, image: np.ndarray, depth: np.ndarray, strength: float | None = None) -> np.ndarray:
    strength = DEFAULT_STRENGTH[name] if strength is None else strength
    out = CUES[name](image, depth, strength)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
