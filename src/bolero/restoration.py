
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


def auto_levels(
    image: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
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

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1

    return keep


def inpaint(
    image: np.ndarray,
    max_area: float = 1.0,
    radius: int = 3,
) -> np.ndarray:
    """Telea inpainting, skipped when the detected fill region is too large."""
    mask = fill_mask(image)

    if not mask.any() or mask.mean() > max_area:
        return image

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(
        to_uint8(image),
        mask,
        inpaintRadius=radius,
        flags=cv2.INPAINT_TELEA,
    )
    return out.astype(np.float32) / 255.0


def estimate_noise(image: np.ndarray) -> float:
    """Gaussian noise std (Immerkaer 1996), averaged over channels, in [0, 1] units."""
    kernel = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
        dtype=np.float32,
    )

    h, w = image.shape[:2]
    sigmas = []

    for c in range(image.shape[-1]):
        response = np.abs(
            cv2.filter2D(image[..., c], -1, kernel)[1:-1, 1:-1]
        ).sum()

        sigmas.append(
            np.sqrt(np.pi / 2.0)
            * response
            / (6.0 * (w - 2) * (h - 2))
        )

    return float(np.mean(sigmas))


def denoise(image: np.ndarray, strength: float = 0.9) -> np.ndarray:
    """Non-local-means denoising at h = strength x estimated sigma."""
    sigma = estimate_noise(image) * 255.0
    h = max(3.0, strength * sigma)

    out = cv2.fastNlMeansDenoisingColored(
        to_uint8(image), None, h, h, 7, 21
    )
    return out.astype(np.float32) / 255.0


def denoise_adaptive(
    image: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Apply mild blur only when estimated noise exceeds the threshold."""
    return gaussian(image, 0.5) if estimate_noise(image) >= threshold else image


def gaussian(image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def sharpen(
    image: np.ndarray,
    amount: float = 1.0,
    sigma: float = 2.0,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image + amount * (image - blurred), 0.0, 1.0)


def bilateral_filter(image: np.ndarray) -> np.ndarray:
    """Bilateral filtering used by the selected Gaussian-noise candidate."""
    out = cv2.bilateralFilter(
        to_uint8(image),
        d=5,
        sigmaColor=30,
        sigmaSpace=5,
    )
    return out.astype(np.float32) / 255.0


Op = Callable[[np.ndarray], np.ndarray]


# Original hand-set plans. These remain the fallback when no tuned plan
# exists for a perturbation.
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
    "jpeg_compression": [bilateral_filter],
    "boundary_blur": [sharpen],
    "boundary_erase": [inpaint],
    "foreground_blur": [sharpen],
    "foreground_occlusion": [inpaint],
    "background_removal": [inpaint],
}


# Candidate plans are evaluated on development data by scripts/tune_plans.py.
# Pass-through is always available when restoration is not demonstrably helpful.
CANDIDATES: dict[str, dict[str, list[Op]]] = {
    "gaussian_noise": {
        "pass_through": [],
        "nlm_0.15": [lambda x: denoise(x, 0.15)],
        "nlm_0.30": [lambda x: denoise(x, 0.30)],
        "nlm_0.90": [lambda x: denoise(x, 0.90)],
        "adaptive": [denoise_adaptive],
        "bilateral": [bilateral_filter],
    },
    **{
        name: {
            "pass_through": [],
            "inpaint_guarded_0.10": [lambda x: inpaint(x, 0.10)],
            "inpaint_guarded_0.25": [lambda x: inpaint(x, 0.25)],
            "inpaint_always": [inpaint],
        }
        for name in (
            "region_removal",
            "boundary_erase",
            "foreground_occlusion",
            "background_removal",
        )
    },
}


def load_plans(
    path: Path,
) -> tuple[dict[str, list[Op]], str, dict[str, str]]:
    """Load dev-selected plans, retaining hand-set plans for untuned types."""
    plans = dict(PLANS)
    selected = {name: "hand_set" for name in PLANS}
    path = Path(path)

    if not path.is_file():
        warnings.warn(
            f"{path} not found; using hand-set restoration plans. "
            "Run scripts/tune_plans.py to select plans on development data."
        )
        return plans, "default", selected

    data = json.loads(path.read_text(encoding="utf-8"))
    chosen = data.get("plans")

    if not isinstance(chosen, dict):
        raise ValueError(
            f"Invalid plan file: {path}. Expected a 'plans' object."
        )

    for perturbation, variant in chosen.items():
        if perturbation not in CANDIDATES:
            raise ValueError(
                f"Unknown perturbation in {path}: {perturbation}"
            )

        if variant not in CANDIDATES[perturbation]:
            raise ValueError(
                f"Unknown plan variant for {perturbation}: {variant}"
            )

        plans[perturbation] = CANDIDATES[perturbation][variant]
        selected[perturbation] = variant

    return plans, "tuned", selected


def restore(
    image: np.ndarray,
    perturbation: str,
    plans: dict[str, list[Op]] | None = None,
) -> np.ndarray:
    """Restore using the supplied plan, falling back to the hand-set plan."""
    active_plans = PLANS if plans is None else plans

    operations = active_plans.get(perturbation)

    if operations is None:
        operations = PLANS.get(perturbation)

    if operations is None:
        raise ValueError(
            f"Unknown perturbation type: {perturbation}"
        )

    out = image

    for op in operations:
        out = op(out)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
