"""The 14 controlled perturbations, each at three severity levels.

Every perturbation is disruptive but non-destructive: the scene is still present,
but some of the evidence a recognizer relies on (color, contrast, texture,
boundaries, the entity itself, or its context) is degraded.

Depth-selective perturbations use the clean image's depth map, i.e. the
perturbation "knows" where the entity is. Restoration never sees that depth map.

HELD_OUT holds stress-test perturbations that are evaluated in the diagnostic
and restoration stages but never used to train the detector or tune strengths,
so they measure how the system copes with distortions it was not built for.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import RandAugment

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


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    # Same kernel size rule as cv2.GaussianBlur(ksize=(0, 0)) on float32 images.
    ksize = int(round(sigma * 8 + 1)) | 1
    x = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).float()


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an (H, W, C) image with reflect-101 borders."""
    k = _gaussian_kernel(sigma)
    pad = len(k) // 2
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)[:, None]
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1))
    return x[:, 0].permute(1, 2, 0).numpy()


def _dilate3(mask: np.ndarray) -> np.ndarray:
    """Binary dilation with a 3x3 square."""
    x = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.max_pool2d(x, 3, stride=1, padding=1)[0, 0].numpy() > 0


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
    rgb = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    decoded = decode_jpeg(encode_jpeg(x, quality=int(quality)))
    return decoded.permute(1, 2, 0).numpy().astype(np.float32) / 255.0


# ---- Boundary operations (depth-guided) -------------------------------------

def boundary_blur(image, depth, fraction, rng):
    """Blur the strongest depth boundaries, softening the entity outline."""
    mask = _dilate3(boundary_mask(depth, fraction))
    return np.where(mask[..., None], _blur(image, 3.0), image)


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


# ---- Held-out stress tests ------------------------------------------------------

def rand_augment(image, depth, magnitude, rng):
    """torchvision RandAugment: two randomly chosen ops at `magnitude` (out of 30)."""
    x = torch.from_numpy(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).permute(2, 0, 1).contiguous()
    # RandAugment draws from torch's global RNG; seed it from `rng` without disturbing it.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(rng.integers(0, 2**31)))
        x = RandAugment(num_ops=2, magnitude=int(magnitude))(x)
    return x.permute(1, 2, 0).numpy().astype(np.float32) / 255.0


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

HELD_OUT: dict[str, Perturbation] = {
    p.name: p
    for p in [
        Perturbation("rand_augment", "mixed", (6, 12, 18), rand_augment),
    ]
}

ALL_PERTURBATIONS: dict[str, Perturbation] = {**PERTURBATIONS, **HELD_OUT}

LEVELS = (1, 2, 3)
