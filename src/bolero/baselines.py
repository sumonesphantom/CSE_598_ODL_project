"""Depth-free preprocessing baselines, applied blindly to every input at a strength in [0, 1].

Each baseline is the depth-free counterpart of one or more depth cues, so the
comparison isolates what the depth map adds:

  unsharp        vs. border_strengthening, depth_boundary_emphasis (edge emphasis)
  autocontrast   vs. shadow_reinforcement (global tone change)
  clahe          vs. contrast_luminance (the same CLAHE, without depth weighting)

Unlike the classical restoration plans, a baseline does not know which
perturbation was applied. Strength 0 is always identity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .cues import equalize_lightness
from .restoration import auto_levels, sharpen


def unsharp(image, strength):
    """Unsharp mask (sigma 2) with amount 2 x strength."""
    return sharpen(image, amount=2.0 * strength)


def autocontrast(image, strength):
    """Blend toward a global auto-levels stretch."""
    return (1.0 - strength) * image + strength * auto_levels(image)


def clahe(image, strength):
    """Blend toward CLAHE-equalized lightness, uniformly over the image."""
    return (1.0 - strength) * image + strength * equalize_lightness(image)


BASELINES: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "unsharp": unsharp,
    "autocontrast": autocontrast,
    "clahe": clahe,
}

# Used only when no tuned strengths file exists (see scripts/tune_strengths.py).
DEFAULT_STRENGTH = {"unsharp": 0.5, "autocontrast": 0.5, "clahe": 0.5}


def apply_baseline(name: str, image: np.ndarray, strength: float | None = None) -> np.ndarray:
    strength = DEFAULT_STRENGTH[name] if strength is None else strength
    out = BASELINES[name](image, strength)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
