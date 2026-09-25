"""Signature-based perturbation detector.

Hand-crafted image statistics (color, contrast, sharpness, noise, blockiness,
constant-fill regions) feed a random forest that predicts which perturbation,
if any, an image carries. The detector is trained on the Imagenette *train*
split and evaluated on *val*, so it never sees the evaluation images.

Decisions are conservative: an image is only routed to restoration when the
detector names a perturbation with probability >= threshold; otherwise it is
treated as clean and passed through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

from .config import SEED
from .perturbations import BLACK, GRAY
from .restoration import estimate_noise

CLEAN = "clean"


def _blockiness(lum: np.ndarray, axis: int) -> float:
    diffs = np.abs(np.diff(lum, axis=axis)).mean(axis=1 - axis)
    idx = np.arange(diffs.size)
    at_block = diffs[(idx % 8) == 7]
    elsewhere = diffs[(idx % 8) != 7]
    return float(at_block.mean() / (elsewhere.mean() + 1e-6)) if at_block.size else 1.0


def extract_features(image: np.ndarray) -> np.ndarray:
    lum = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    hue = hsv[..., 0] / 360.0
    h = lum.shape[0]

    lap = cv2.Laplacian(lum, cv2.CV_32F)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx**2 + gy**2)
    residual = lum - ndimage.median_filter(lum, size=3)

    dark = ndimage.minimum_filter(image.min(axis=-1), size=7)
    bright = ndimage.maximum_filter(image.max(axis=-1), size=7)

    black_fill = np.all(np.abs(image - BLACK) < 1e-4, axis=-1).mean()
    gray_fill = np.all(np.abs(image - GRAY) < 1e-4, axis=-1).mean()

    lum_hist = np.histogram(lum, bins=8, range=(0, 1))[0] / lum.size
    hue_hist = np.histogram(hue, bins=8, range=(0, 1), weights=sat)[0] / (sat.sum() + 1e-6)

    feats = [
        *np.percentile(lum, [1, 5, 50, 95, 99]), lum.mean(), lum.std(),
        *image.mean(axis=(0, 1)), *image.std(axis=(0, 1)),
        sat.mean(), sat.std(), (sat < 0.05).mean(),
        black_fill, gray_fill,
        lap.var(), lap.var() / (lum.var() + 1e-6),
        grad.mean(), np.percentile(grad, 95),
        residual.std(),
        estimate_noise(image),
        _blockiness(lum, 0), _blockiness(lum, 1),
        lum[: h // 3].mean() - lum[-h // 3:].mean(),
        dark.mean(), bright.mean(),
        *lum_hist, *hue_hist,
    ]
    return np.asarray(feats, dtype=np.float32)


@dataclass
class Decision:
    label: str        # the perturbation to undo, or "clean"
    predicted: str    # the detector's raw top-1 label
    probability: float


class PerturbationDetector:
    def __init__(self, threshold: float = 0.6, n_estimators: int = 400):
        self.threshold = threshold
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced", random_state=SEED,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PerturbationDetector":
        self.model.fit(features, labels)
        return self

    @property
    def classes(self) -> np.ndarray:
        return self.model.classes_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(features))

    def decide(self, image: np.ndarray) -> Decision:
        proba = self.predict_proba(extract_features(image))[0]
        best = int(np.argmax(proba))
        predicted = str(self.classes[best])
        confident = predicted != CLEAN and proba[best] >= self.threshold
        return Decision(predicted if confident else CLEAN, predicted, float(proba[best]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"threshold": self.threshold, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "PerturbationDetector":
        blob = joblib.load(path)
        detector = cls(threshold=blob["threshold"])
        detector.model = blob["model"]
        return detector
