"""Monocular relative depth (Depth-Anything-V2-Small) and depth-derived structure.

Depth maps are float32 HxW in [0, 1] with 1 = nearest, matching the model's
relative inverse-depth output after per-image normalization.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .config import CACHE_DIR, DEPTH_MODEL, get_device
from .data import ImageRecord, to_pil


def normalize(values: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    low, high = np.percentile(values, [low_pct, high_pct])
    return np.clip((values - low) / (high - low + 1e-8), 0.0, 1.0).astype(np.float32)


class DepthEstimator:
    def __init__(self, model_name: str = DEPTH_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = get_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        self.last_seconds = 0.0

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        inputs = self.processor(images=to_pil(image), return_tensors="pt").to(self.device)
        predicted = self.model(**inputs).predicted_depth
        depth = F.interpolate(
            predicted.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
        )[0, 0].float().cpu().numpy()
        self.last_seconds = time.perf_counter() - start
        return normalize(depth)


class DepthCache:
    """Caches clean-image depth maps on disk as float16 .npy files."""

    def __init__(self, estimator: DepthEstimator | None = None, root: Path | None = None):
        self._estimator = estimator
        slug = DEPTH_MODEL.split("/")[-1]
        self.root = (root or CACHE_DIR / "depth") / slug
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def estimator(self) -> DepthEstimator:
        if self._estimator is None:
            self._estimator = DepthEstimator()
        return self._estimator

    def path(self, record: ImageRecord) -> Path:
        return self.root / f"{record.image_id}.npy"

    def get(self, record: ImageRecord, image: np.ndarray) -> np.ndarray:
        path = self.path(record)
        if path.exists():
            return np.load(path).astype(np.float32)
        depth = self.estimator(image)
        np.save(path, depth.astype(np.float16))
        return depth


def gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(depth)
    return np.sqrt(dx**2 + dy**2)


def boundary_strength(depth: np.ndarray) -> np.ndarray:
    """Depth-gradient magnitude scaled to [0, 1]."""
    grad = gradient_magnitude(depth)
    return (grad / (grad.max() + 1e-8)).astype(np.float32)


def boundary_mask(depth: np.ndarray, fraction: float) -> np.ndarray:
    """The `fraction` of pixels with the strongest depth gradient."""
    grad = gradient_magnitude(depth)
    return grad >= np.quantile(grad, 1.0 - fraction)


def near_mask(depth: np.ndarray, fraction: float) -> np.ndarray:
    """The nearest `fraction` of pixels (candidate entity region)."""
    return depth >= np.quantile(depth, 1.0 - fraction)


def far_mask(depth: np.ndarray, fraction: float) -> np.ndarray:
    """The farthest `fraction` of pixels (context / background)."""
    return depth <= np.quantile(depth, fraction)


def smooth(depth: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(depth, sigma=sigma).astype(np.float32)
