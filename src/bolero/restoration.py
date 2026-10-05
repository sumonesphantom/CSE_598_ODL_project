"""Lightweight classical restoration, one plan per perturbation type.

A plan is a short sequence of image operations. The restoration stage receives
only the perturbed image (and optionally a detector's guess of what happened);
it never sees the clean image or the clean depth map.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .data import to_uint8
from .perturbations import BLACK, GRAY


def identity(image: np.ndarray) -> np.ndarray:
    return image


def auto_levels(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Global linear stretch (same for all channels, so color balance is preserved)."""
    low, high = np.percentile(image, [low_pct, high_pct])
    if high - low < 1e-3:
        return image
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def invert(image: np.ndarray) -> np.ndarray:
    return 1.0 - image


def fill_mask(image: np.ndarray, min_area: int = 25) -> np.ndarray:
    """Pixels that are exactly the constant fill used by occlusion perturbations."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for value in (BLACK, GRAY):
        mask |= np.all(np.abs(image - value) < 1e-4, axis=-1).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def inpaint(image: np.ndarray, max_area: float = 1.0, radius: int = 3) -> np.ndarray:
    """Telea inpainting of the constant-fill region, skipped if the hole is too large.

    Telea diffuses surrounding texture inward, which is plausible for a small
    hole and fabrication for a large one. `background_removal` blanks 30-70 % of
    the frame, and inpainting it costs 15 pp. `max_area` (a fraction of the
    image, selected on dev data) bounds what the plan is willing to invent;
    1.0 keeps the original always-inpaint behaviour.
    """
    mask = fill_mask(image)
    if not mask.any() or mask.mean() > max_area:
        return image
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(to_uint8(image), mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


def estimate_noise(image: np.ndarray) -> float:
    """Gaussian noise std (Immerkaer 1996), averaged over channels, in [0, 1] units."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    h, w = image.shape[:2]
    sigmas = []
    for c in range(image.shape[-1]):
        response = np.abs(cv2.filter2D(image[..., c], -1, kernel)[1:-1, 1:-1]).sum()
        sigmas.append(np.sqrt(np.pi / 2.0) * response / (6.0 * (w - 2) * (h - 2)))
    return float(np.mean(sigmas))


def denoise(image: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Non-local-means denoising at `h = strength x estimated sigma`.

    `strength` is selected on dev data (scripts/tune_plans.py). Strong settings
    smear the fine texture the classifier relies on: at strength 0.9 the whole
    plan costs 16 pp on noisy images, more than the noise itself does.
    """
    sigma = estimate_noise(image) * 255.0
    h = max(3.0, strength * sigma)
    out = cv2.fastNlMeansDenoisingColored(to_uint8(image), None, h, h, 7, 21)
    return out.astype(np.float32) / 255.0


def denoise_adaptive(image: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """Mild blur only when the estimated noise is heavy, otherwise pass through.

    One plan per perturbation cannot express a severity-dependent response, but
    estimated sigma is a severity proxy available at deployment.
    """
    return gaussian(image, 0.5) if estimate_noise(image) >= threshold else image


def gaussian(image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def sharpen(image: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + amount * (image - blurred), 0.0, 1.0)


def deblock(image: np.ndarray) -> np.ndarray:
    out = cv2.bilateralFilter(to_uint8(image), d=5, sigmaColor=30, sigmaSpace=5)
    return out.astype(np.float32) / 255.0


Op = Callable[[np.ndarray], np.ndarray]

# Perturbation name -> restoration plan. Grayscale has no classical inverse
# (color is gone), so its plan is empty; a learned colorizer would go here.
PLANS: dict[str, list[Op]] = {
    "clean": [],
    "region_removal": [inpaint],
    "inversion": [invert, auto_levels],
    "grayscale": [],
    "low_contrast": [auto_levels],
    "darken": [auto_levels],
    "brighten": [auto_levels],
    "gaussian_blur": [sharpen],
    "gaussian_noise": [denoise],
    "jpeg_compression": [deblock],
    "boundary_blur": [sharpen],
    "boundary_erase": [inpaint],
    "foreground_blur": [sharpen],
    "foreground_occlusion": [inpaint],
    "background_removal": [inpaint],
}


# Candidate plans per perturbation, scored on the train split by
# scripts/tune_plans.py. "pass_through" is always a candidate: where a
# perturbation destroys information rather than transforming it, no classical
# inverse exists and the best available plan is to leave the image alone.
CANDIDATES: dict[str, dict[str, list[Op]]] = {
    "gaussian_noise": {
        "pass_through": [],
        "nlm_0.15": [lambda x: denoise(x, 0.15)],
        "nlm_0.30": [lambda x: denoise(x, 0.30)],
        "nlm_0.90": [lambda x: denoise(x, 0.90)],   # the original hand-set plan
        "adaptive": [denoise_adaptive],
        "bilateral": [deblock],
    },
    **{
        name: {
            "pass_through": [],
            "inpaint_guarded_0.10": [lambda x: inpaint(x, 0.10)],
            "inpaint_guarded_0.25": [lambda x: inpaint(x, 0.25)],
            "inpaint_always": [inpaint],                 # the original hand-set plan
        }
        for name in ("region_removal", "boundary_erase", "foreground_occlusion",
                     "background_removal")
    },
}


def load_plans(path) -> tuple[dict[str, list[Op]], str]:
    """Plans with dev-selected candidates substituted in, falling back to PLANS."""
    plans = dict(PLANS)
    path = Path(path)
    if not path.exists():
        warnings.warn(f"{path} not found; using hand-set restoration plans. "
                      "Run scripts/tune_plans.py to choose them on dev data.")
        return plans, "default"
    chosen = json.loads(path.read_text())["plans"]
    for perturbation, variant in chosen.items():
        plans[perturbation] = CANDIDATES[perturbation][variant]
    return plans, "tuned"


def restore(image: np.ndarray, perturbation: str,
            plans: dict[str, list[Op]] | None = None) -> np.ndarray:
    out = image
    for op in (plans or PLANS)[perturbation]:
        out = op(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
