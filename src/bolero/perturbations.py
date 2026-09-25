"""The 14 controlled perturbations, each at three severity levels.

Every perturbation is disruptive but non-destructive: the scene is still present,
but some of the evidence a recognizer relies on (color, contrast, texture,
boundaries, the entity itself, or its context) is degraded.

Depth-selective perturbations use the clean image's depth map, i.e. the
perturbation "knows" where the entity is. Restoration never sees that depth map.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .depth import boundary_mask, far_mask, near_mask

BLACK = 0.0
GRAY = 0.5


@dataclass(frozen=True)
class Perturbation:
    name: str
    family: str
    severities: tuple[float, float, float]
    fn: Callable[[np.ndarray, np.ndarray | None, float, np.random.Generator], np.ndarray]
    needs_depth: bool = False

    def __call__(
        self,
        image: np.ndarray,
        level: int,
        depth: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.needs_depth and depth is None:
            raise ValueError(f"{self.name} needs a depth map")
        rng = rng or np.random.default_rng(0)
        out = self.fn(image, depth, self.severities[level - 1], rng)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def rng_for(image_id: str, name: str, level: int, seed: int) -> np.random.Generator:
    """Deterministic per-(image, perturbation, level) generator."""
    return np.random.default_rng([seed, zlib.crc32(f"{image_id}|{name}|{level}".encode())])


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _fill(image: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    out = image.copy()
    out[mask] = value
    return out


# ---- Occlusion ------------------------------------------------------------

def region_removal(image, depth, area, rng):
    """Black out one random rectangle covering `area` of the image."""
    h, w = image.shape[:2]
    aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    rh = int(round(min(h, np.sqrt(area * h * w * aspect))))
    rw = int(round(min(w, area * h * w / max(rh, 1))))
    top = rng.integers(0, h - rh + 1)
    left = rng.integers(0, w - rw + 1)
    out = image.copy()
    out[top:top + rh, left:left + rw] = BLACK
    return out


# ---- Photometric / color ---------------------------------------------------

def inversion(image, depth, alpha, rng):
    return (1.0 - alpha) * image + alpha * (1.0 - image)


def grayscale(image, depth, alpha, rng):
    gray = _luminance(image)[..., None]
    return (1.0 - alpha) * image + alpha * gray


def low_contrast(image, depth, factor, rng):
    mean = image.mean()
    return mean + factor * (image - mean)


def darken(image, depth, factor, rng):
    return image * factor


def brighten(image, depth, factor, rng):
    return 1.0 - (1.0 - image) * factor


# ---- Blur / noise / compression --------------------------------------------

def gaussian_blur(image, depth, sigma, rng):
    return _blur(image, sigma)


def gaussian_noise(image, depth, std, rng):
    return image + rng.normal(0.0, std, size=image.shape).astype(np.float32)


def jpeg_compression(image, depth, quality, rng):
    bgr = cv2.cvtColor(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ---- Boundary operations (depth-guided) -------------------------------------

def boundary_blur(image, depth, fraction, rng):
    """Blur the strongest depth boundaries, softening the entity outline."""
    mask = cv2.dilate(boundary_mask(depth, fraction).astype(np.uint8), np.ones((3, 3), np.uint8))
    return np.where(mask[..., None] > 0, _blur(image, 3.0), image)


def boundary_erase(image, depth, fraction, rng):
    """Replace the strongest depth boundaries with flat gray."""
    return _fill(image, boundary_mask(depth, fraction), GRAY)


# ---- Depth-selective ----------------------------------------------------------

def foreground_blur(image, depth, sigma, rng):
    """Blur the near (entity) region, leaving context intact."""
    mask = near_mask(depth, 0.4)
    return np.where(mask[..., None], _blur(image, sigma), image)


def foreground_occlusion(image, depth, fraction, rng):
    """Gray out the nearest `fraction` of pixels: the entity is removed, context remains."""
    return _fill(image, near_mask(depth, fraction), GRAY)


def background_removal(image, depth, fraction, rng):
    """Gray out the farthest `fraction` of pixels: context is removed, the entity remains."""
    return _fill(image, far_mask(depth, fraction), GRAY)


PERTURBATIONS: dict[str, Perturbation] = {
    p.name: p
    for p in [
        Perturbation("region_removal", "occlusion", (0.10, 0.20, 0.35), region_removal),
        Perturbation("inversion", "color", (0.70, 0.85, 1.00), inversion),
        Perturbation("grayscale", "color", (0.50, 0.75, 1.00), grayscale),
        Perturbation("low_contrast", "photometric", (0.50, 0.30, 0.15), low_contrast),
        Perturbation("darken", "photometric", (0.60, 0.40, 0.20), darken),
        Perturbation("brighten", "photometric", (0.60, 0.40, 0.20), brighten),
        Perturbation("gaussian_blur", "blur", (1.0, 2.0, 3.5), gaussian_blur),
        Perturbation("gaussian_noise", "noise", (0.04, 0.08, 0.16), gaussian_noise),
        Perturbation("jpeg_compression", "compression", (30, 15, 5), jpeg_compression),
        Perturbation("boundary_blur", "boundary", (0.10, 0.20, 0.35), boundary_blur, True),
        Perturbation("boundary_erase", "boundary", (0.05, 0.10, 0.20), boundary_erase, True),
        Perturbation("foreground_blur", "depth_selective", (2.0, 4.0, 6.0), foreground_blur, True),
        Perturbation("foreground_occlusion", "depth_selective", (0.10, 0.20, 0.35), foreground_occlusion, True),
        Perturbation("background_removal", "depth_selective", (0.30, 0.50, 0.70), background_removal, True),
    ]
}

LEVELS = (1, 2, 3)
