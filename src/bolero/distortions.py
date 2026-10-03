"""Geometric distortions modeled on the whole-image effects of imagy.app/image-distorter.

That tool only runs interactively in the browser, so these are local PyTorch
re-implementations: each effect is an inverse coordinate map sampled with
`grid_sample`. Unlike the photometric perturbations, they move pixels rather than
change their values.

Each distortion is a `Perturbation` with three severity levels and works on
float32 (H, W, 3) images in [0, 1]. They are not part of the study's perturbation
set. scripts/smoke_distortions.py renders and classifies them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .perturbations import Perturbation


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)[None]


def _grid(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-center coordinates in [-1, 1] (grid_sample, align_corners=False)."""
    ys = (torch.arange(h, dtype=torch.float32) * 2 + 1) / h - 1
    xs = (torch.arange(w, dtype=torch.float32) * 2 + 1) / w - 1
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return x, y


def _sample(image: np.ndarray, x: torch.Tensor, y: torch.Tensor, padding: str = "border") -> np.ndarray:
    """Output pixel (i, j) takes the input value at normalized source coordinate (x, y)[i, j]."""
    grid = torch.stack([x, y], dim=-1)[None]
    out = F.grid_sample(_to_tensor(image), grid, mode="bilinear", padding_mode=padding, align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


def _polar_warp(image: np.ndarray, fn, padding: str = "border") -> np.ndarray:
    """Remap (radius, angle) -> fn(radius, angle) about the center.

    Radii are aspect-corrected so the inscribed circle has radius 1.
    """
    h, w = image.shape[:2]
    x, y = _grid(h, w)
    sx, sy = w / min(h, w), h / min(h, w)
    u, v = x * sx, y * sy
    r, theta = torch.hypot(u, v), torch.atan2(v, u)
    r_src, theta_src = fn(r, theta)
    return _sample(image, r_src * torch.cos(theta_src) / sx, r_src * torch.sin(theta_src) / sy, padding)


def swirl(image, depth, turns, rng):
    """Rotate by up to `turns` full turns at the center, fading to none at radius 1."""
    return _polar_warp(image, lambda r, t: (r, t + turns * 2 * math.pi * (1 - r).clamp(min=0) ** 2))


def bulge(image, depth, power, rng):
    """Magnify the center disc (a lens pushed out of the image)."""
    return _polar_warp(image, lambda r, t: (torch.where(r < 1, r ** (1 + power), r), t))


def pinch(image, depth, power, rng):
    """Shrink the center disc toward the middle (the inverse of bulge)."""
    return _polar_warp(image, lambda r, t: (torch.where(r < 1, r ** (1 / (1 + power)), r), t))


def fisheye(image, depth, k, rng):
    """Full-frame barrel distortion; corners that map outside the image turn black."""
    return _polar_warp(image, lambda r, t: (r * (1 + k * r ** 2) / (1 + k), t), padding="zeros")


def _wave(image, amplitude, horizontal: bool, cycles: float = 4.0):
    h, w = image.shape[:2]
    x, y = _grid(h, w)
    if horizontal:   # rows slide left and right
        return _sample(image, x + amplitude * torch.sin(math.pi * cycles * (y + 1)), y)
    return _sample(image, x, y + amplitude * torch.sin(math.pi * cycles * (x + 1)))


def horizontal_wave(image, depth, amplitude, rng):
    return _wave(image, amplitude, horizontal=True)


def vertical_wave(image, depth, amplitude, rng):
    return _wave(image, amplitude, horizontal=False)


def stretch(image, depth, factor, rng):
    """Widen the image by `1 + factor`, cropping the sides."""
    x, y = _grid(*image.shape[:2])
    return _sample(image, x / (1 + factor), y)


def squeeze(image, depth, factor, rng):
    """Narrow the image by `1 + factor`, leaving black bars at the sides."""
    x, y = _grid(*image.shape[:2])
    return _sample(image, x * (1 + factor), y, padding="zeros")


def kaleidoscope(image, depth, radius, rng, segments: int = 6):
    """Mirror one wedge into `segments` around the center, within `radius`."""
    seg = 2 * math.pi / segments

    def fold(r, t):
        a = torch.remainder(t, seg)
        return r, torch.where(r < radius, torch.minimum(a, seg - a), t)

    return _polar_warp(image, fold)


def polar_coordinates(image, depth, mix, rng):
    """Morph toward a rectangular-to-polar unwrap (x -> angle, y -> radius); mix=1 is the full unwrap."""
    x, y = _grid(*image.shape[:2])
    r, t = (y + 1) / 2 * math.sqrt(2), math.pi * x
    px, py = r * torch.cos(t), r * torch.sin(t)
    return _sample(image, (1 - mix) * x + mix * px, (1 - mix) * y + mix * py)


def pixelate_shift(image, depth, block, rng):
    """Pixelate into `block`-pixel squares and slide each row of blocks sideways at random."""
    b = int(block)
    h, w = image.shape[:2]
    small = F.avg_pool2d(_to_tensor(image), b, ceil_mode=True)
    out = F.interpolate(small, scale_factor=b, mode="nearest")[0, :, :h, :w].permute(1, 2, 0).numpy().copy()
    for top in range(0, h, b):
        out[top:top + b] = np.roll(out[top:top + b], int(rng.integers(-b, b + 1)), axis=1)
    return out


DISTORTIONS: dict[str, Perturbation] = {
    p.name: p
    for p in [
        Perturbation("swirl", "geometric", (0.25, 0.5, 1.0), swirl),
        Perturbation("bulge", "geometric", (0.3, 0.6, 1.0), bulge),
        Perturbation("pinch", "geometric", (0.3, 0.6, 1.0), pinch),
        Perturbation("fisheye", "geometric", (0.3, 0.6, 1.0), fisheye),
        Perturbation("horizontal_wave", "geometric", (0.02, 0.04, 0.08), horizontal_wave),
        Perturbation("vertical_wave", "geometric", (0.02, 0.04, 0.08), vertical_wave),
        Perturbation("stretch", "geometric", (0.15, 0.3, 0.6), stretch),
        Perturbation("squeeze", "geometric", (0.15, 0.3, 0.6), squeeze),
        Perturbation("kaleidoscope", "geometric", (0.5, 1.0, 2.0), kaleidoscope),
        Perturbation("polar_coordinates", "geometric", (0.25, 0.5, 1.0), polar_coordinates),
        Perturbation("pixelate_shift", "geometric", (3, 6, 10), pixelate_shift),
    ]
}
