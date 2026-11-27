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


# ---- Physics-based weather: heavy rain and dust storms -------------------------
#
# Kept out of PERTURBATIONS / ALL_PERTURBATIONS on purpose: the trained detector,
# tuned strengths, restoration plans and every earlier result cover the 14 + 1
# suite above. Pass these by name (`--perturbations rain_streaks ...` where a
# script resolves names through ANY_PERTURBATION, or `conditions.iter_samples`).
#
# Haze uses the atmospheric scattering model I = J * t + A * (1 - t) with
# transmission t = exp(-beta * d). Depth maps are 1 = nearest, so scene distance
# is d = d0 + (1 - d0) * (1 - depth): even the nearest pixel sits at d0 > 0.
# Names start with "rain" / "dust" so conditions.condition_label maps them onto
# the `weather` taxonomy.

RAIN_AIRLIGHT = np.array([0.80, 0.82, 0.85], dtype=np.float32)   # cool gray-white
DUST_AIRLIGHT = np.array([0.76, 0.60, 0.38], dtype=np.float32)   # yellow-brown
# Wavelength-dependent extinction for dust, relative to green: blue fades first.
DUST_BETA_RGB = np.array([0.80, 1.00, 1.35], dtype=np.float32)
_NEAR_DISTANCE = 0.15


def _distance(depth: np.ndarray) -> np.ndarray:
    return (_NEAR_DISTANCE + (1.0 - _NEAR_DISTANCE) * (1.0 - depth)).astype(np.float32)


def _scatter(image: np.ndarray, depth: np.ndarray, beta, airlight: np.ndarray) -> np.ndarray:
    """I = J t + A (1 - t), t = exp(-beta d); `beta` is a scalar or per-channel (3,)."""
    t = np.exp(-np.asarray(beta, dtype=np.float32) * _distance(depth)[..., None])
    return image * t + airlight * (1.0 - t)


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalized motion-blur kernel: a line of `length` pixels at `angle_deg` from vertical."""
    size = length | 1
    k = np.zeros((size, size), np.float32)
    c, a = size // 2, np.deg2rad(angle_deg)
    for s in np.linspace(-c, c, 2 * size):
        k[int(round(c + s * np.cos(a))), int(round(c + s * np.sin(a)))] = 1.0
    return k / k.sum()


def _rain_layer(shape: tuple[int, int], s: float, rng: np.random.Generator) -> np.ndarray:
    """(H, W) streak intensity in [0, 1]: sparse drops smeared along a near-vertical wind angle."""
    import cv2

    h, w = shape
    drops = (rng.random((h, w)) < 0.004 + 0.012 * s).astype(np.float32)
    drops *= rng.uniform(0.5, 1.0, size=(h, w)).astype(np.float32)
    length = max(5, int(round(min(h, w) * (0.06 + 0.10 * s))))
    angle = float(rng.uniform(-15.0, 15.0))
    streaks = cv2.filter2D(drops, -1, _line_kernel(length, angle), borderType=cv2.BORDER_REFLECT)
    streaks = cv2.GaussianBlur(streaks, (0, 0), 0.6)
    return np.clip(streaks / (streaks.max() + 1e-6) * (0.55 + 0.35 * s), 0.0, 1.0)


def _screen(image: np.ndarray, layer: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Screen-blend a bright (H, W) layer of `color` over the image."""
    return 1.0 - (1.0 - image) * (1.0 - layer[..., None] * color)


def rain_streaks(image, depth, s, rng):
    """Falling rain: bright motion-blurred streaks, slight veiling and contrast loss."""
    streaks = _rain_layer(image.shape[:2], s, rng)
    veiled = image * (1.0 - 0.12 * s) + RAIN_AIRLIGHT * 0.12 * s
    return _screen(veiled, streaks, RAIN_AIRLIGHT)


def rain_mist(image, depth, s, rng):
    """Heavy rain: depth-dependent rain haze (far scene washes out first) plus streaks."""
    hazed = _scatter(image, depth, 0.8 + 2.4 * s, RAIN_AIRLIGHT)
    return _screen(hazed, _rain_layer(image.shape[:2], s, rng), RAIN_AIRLIGHT)


def rain_lens(image, depth, s, rng):
    """Raindrops on the lens: defocused, slightly refracted, rim-darkened discs."""
    h, w = image.shape[:2]
    blurred = _blur(image, 2.0 + 3.0 * s)
    out = image.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(round(4 + 18 * s))):
        r = float(rng.uniform(0.03, 0.06 + 0.06 * s) * min(h, w))
        cy, cx = float(rng.uniform(0, h)), float(rng.uniform(0, w))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r
        inside = dist < 1.0
        if not inside.any():
            continue
        # Refraction: sample the blurred image flipped about the drop center (a lens inverts).
        sy = np.clip(np.round(2 * cy - yy), 0, h - 1).astype(int)
        sx = np.clip(np.round(2 * cx - xx), 0, w - 1).astype(int)
        drop = blurred[sy, sx] * 1.05 + 0.03
        rim = np.clip((dist - 0.8) / 0.2, 0.0, 1.0)[..., None] * 0.2
        out = np.where(inside[..., None], drop * (1.0 - rim), out)
    return out


def dust_haze(image, depth, s, rng):
    """Dust: yellow-brown scattering with blue fading first, plus contrast loss."""
    hazed = _scatter(image, depth, (0.4 + 2.0 * s) * DUST_BETA_RGB, DUST_AIRLIGHT)
    mean = hazed.mean(axis=(0, 1), keepdims=True)
    return mean + (hazed - mean) * (1.0 - 0.30 * s)


def dust_storm(image, depth, s, rng):
    """Sand storm: dense dust haze plus wind-blown, near-horizontal sand grains."""
    import cv2

    hazed = dust_haze(image, depth, min(1.0, 1.2 * s), rng)
    h, w = image.shape[:2]
    grains = (rng.random((h, w)) < 0.02 + 0.05 * s).astype(np.float32)
    length = max(3, int(round(min(h, w) * (0.02 + 0.04 * s))))
    grains = cv2.filter2D(grains, -1, _line_kernel(length, float(rng.uniform(70.0, 110.0))),
                          borderType=cv2.BORDER_REFLECT)
    grains = np.clip(grains / (np.quantile(grains[grains > 0], 0.5) + 1e-6) * (0.25 + 0.25 * s), 0.0, 0.9)
    return hazed * (1.0 - grains[..., None]) + DUST_AIRLIGHT * 0.9 * grains[..., None]


WEATHER: dict[str, Perturbation] = {
    p.name: p
    for p in [
        Perturbation("rain_streaks", "rain", (0.35, 0.65, 1.00), rain_streaks),
        Perturbation("rain_mist", "rain", (0.35, 0.65, 1.00), rain_mist, True),
        Perturbation("rain_lens", "rain", (0.35, 0.65, 1.00), rain_lens),
        Perturbation("dust_haze", "dust", (0.35, 0.65, 1.00), dust_haze, True),
        Perturbation("dust_storm", "dust", (0.35, 0.65, 1.00), dust_storm, True),
    ]
}

# Every perturbation by name, including the opt-in weather set.
ANY_PERTURBATION: dict[str, Perturbation] = {**ALL_PERTURBATIONS, **WEATHER}

LEVELS = (1, 2, 3)
