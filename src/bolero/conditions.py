"""Condition taxonomy for the decision engine.

A `ConditionSample` is one (image, degradation) pair with a high-level condition
label and a severity. Deciders are asked to recover the condition and severity
from the image (and optional evidence), never from the label itself.

Two taxonomies are supported:

  family   clean + the perturbation families in `perturbations.py`. Works on the
           current synthetic suite and is the default for the first comparison.
  weather  clean / rain / dust / other. Becomes useful once the physics-based
           rain and dust perturbations land; existing perturbations map to
           "other" until then.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

from .config import SEED
from .data import ImageRecord, load_image
from .perturbations import ALL_PERTURBATIONS, LEVELS, rng_for

TAXONOMIES: dict[str, dict[str, str]] = {
    "family": {
        "clean": "No visible degradation.",
        "occlusion": "Part of the scene is covered or removed.",
        "color": "The colors are inverted or removed.",
        "photometric": "The image is too dark, too bright, or too flat.",
        "blur": "The whole image is blurred.",
        "noise": "Sensor noise is visible.",
        "compression": "JPEG compression artifacts are visible.",
        "boundary": "The outline of the main object is softened or erased.",
        "depth_selective": "The near object or the far background is altered, but not both.",
    },
    "weather": {
        "clean": "No visible degradation.",
        "rain": "Rain streaks, raindrops on the lens, or wet haze.",
        "dust": "Sand or dust storm, yellow-brown haze, low contrast.",
        "other": "A different corruption: blur, noise, compression, occlusion, or lighting.",
    },
}


@dataclass(frozen=True)
class ConditionSample:
    image_id: str
    wnid: str
    class_name: str
    condition: str
    severity: int | None          # 1..3, None for clean
    image: np.ndarray             # the image the decider sees
    clean_image: np.ndarray       # reference, never shown to the decider
    perturbation: str | None = None
    level: int | None = None

    @property
    def clean(self) -> bool:
        return self.perturbation is None


def condition_label(perturbation: str | None, family: str | None, taxonomy: str = "family") -> str:
    """Map a perturbation to a high-level condition label."""
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
    if perturbation is None:
        return "clean"
    if taxonomy == "weather":
        name = perturbation.lower()
        if name.startswith("rain"):
            return "rain"
        if name.startswith("dust") or "sand" in name:
            return "dust"
        return "other"
    return family or "other"


def iter_samples(
    records: Sequence[ImageRecord],
    taxonomy: str = "family",
    perturbations: Sequence[str] | None = None,
    levels: Sequence[int] = LEVELS,
    max_per_bucket: int | None = None,
    depth_for: Callable[[ImageRecord, np.ndarray], np.ndarray] | None = None,
) -> Iterator[ConditionSample]:
    """Yield clean and degraded samples in a deterministic order.

    `max_per_bucket` caps each (condition, severity) bucket, which keeps the
    comparison balanced across severities without reordering the source stream.
    `depth_for(record, image) -> depth` is required for depth-dependent
    perturbations and should be a `DepthCache`-like callable.
    """
    names = list(ALL_PERTURBATIONS if perturbations is None else perturbations)
    for name in names:
        if name not in ALL_PERTURBATIONS:
            raise ValueError(f"unknown perturbation {name!r}")
    counts: Counter[tuple[str, int | None]] = Counter()

    def keep(condition: str, severity: int | None) -> bool:
        key = (condition, severity)
        if max_per_bucket is not None and counts[key] >= max_per_bucket:
            return False
        counts[key] += 1
        return True

    for record in records:
        clean = load_image(record.path)
        if keep("clean", None):
            yield ConditionSample(
                image_id=record.image_id,
                wnid=record.wnid,
                class_name=record.class_name,
                condition="clean",
                severity=None,
                image=clean,
                clean_image=clean,
            )
        depth = None
        for name in names:
            perturbation = ALL_PERTURBATIONS[name]
            if perturbation.needs_depth:
                if depth_for is None:
                    raise ValueError(f"{name} needs a depth map; pass depth_for")
                if depth is None:
                    depth = depth_for(record, clean)
            for level in levels:
                condition = condition_label(name, perturbation.family, taxonomy)
                if not keep(condition, level):
                    continue
                rng = rng_for(record.image_id, name, level, SEED)
                degraded = perturbation(clean, level, depth if perturbation.needs_depth else None, rng)
                yield ConditionSample(
                    image_id=record.image_id,
                    wnid=record.wnid,
                    class_name=record.class_name,
                    condition=condition,
                    severity=level,
                    image=degraded,
                    clean_image=clean,
                    perturbation=name,
                    level=level,
                )
