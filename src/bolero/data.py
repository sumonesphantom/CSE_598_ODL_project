"""Imagenette image listing and loading. Images are float32 HxWx3 arrays in [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGENETTE_CLASSES, IMAGENETTE_INDEX, SEED, find_imagenette_root

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    wnid: str

    @property
    def class_name(self) -> str:
        return IMAGENETTE_CLASSES[self.wnid]

    @property
    def label_index(self) -> int:
        return IMAGENETTE_INDEX[self.wnid]


def list_images(
    split: str = "val",
    root: Path | None = None,
    per_class: int | None = None,
    seed: int = SEED,
) -> list[ImageRecord]:
    """List Imagenette images, optionally sampling `per_class` images from each class."""
    root = root or find_imagenette_root()
    if root is None:
        raise FileNotFoundError(
            "Imagenette not found. Run `python scripts/prepare_data.py` or set IMAGENETTE_ROOT."
        )
    rng = np.random.default_rng(seed)
    records: list[ImageRecord] = []
    for wnid in IMAGENETTE_CLASSES:
        paths = sorted(
            p for p in (root / split / wnid).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if per_class is not None and per_class < len(paths):
            keep = np.sort(rng.choice(len(paths), size=per_class, replace=False))
            paths = [paths[i] for i in keep]
        records += [ImageRecord(p.stem, p, wnid) for p in paths]
    return records


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8(image))
