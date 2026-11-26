"""Object-level spatial evidence from SAM (masks) or DETR (boxes).

A classifier gives classes only. These models add *where* the entity is, which
the depth-gradient outline only approximates and which fails on flat,
background-like entities (churches, altars). A region is turned into a
structure map with the same contract as a depth map (float32 HxW in [0, 1],
1 = entity), so every cue in cues.py runs unchanged on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import CACHE_DIR, SEGMENTERS, TEXT_SEGMENTER, get_device
from .data import ImageRecord, to_pil


@dataclass
class Region:
    mask: np.ndarray                     # HxW bool
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 (exclusive)
    score: float                         # model confidence
    label: str | None = None             # DETR class name; None for SAM

    @property
    def area(self) -> float:
        return float(self.mask.mean())


def box_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_mask(box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return mask


class Segmenter:
    """Wraps a Hugging Face mask-generation (SAM) or object-detection (DETR) pipeline."""

    def __init__(self, kind: str = "sam", device: str | None = None, min_score: float = 0.5,
                 sam_grid: int = 16):
        from transformers import pipeline

        self.kind = kind
        self.min_score = min_score
        self.sam_grid = sam_grid  # SAM prompt points per side; 32 is SAM's default, 16 is ~4x faster
        self.device = get_device(device)
        if kind == "sam":
            # SAM's point grid is float64, which MPS cannot hold; fall back to CPU there.
            if self.device.type == "mps":
                self.device = get_device("cpu")
            self.pipe = pipeline("mask-generation", model=SEGMENTERS[kind], framework="pt",
                                 device=self.device)
        elif kind == "detr":
            # The `no_timm` revision carries `dilation: null`, which transformers 5.x
            # rejects during config validation. The main revision uses the timm
            # ResNet-50 backbone instead (see requirements.txt).
            self.pipe = pipeline("object-detection", model=SEGMENTERS[kind],
                                 framework="pt", device=self.device)
        else:
            raise ValueError(f"unknown segmenter {kind!r}; choose from {list(SEGMENTERS)}")
        self.last_seconds = 0.0

    def __call__(self, image: np.ndarray) -> list[Region]:
        start = time.perf_counter()
        shape = image.shape[:2]
        regions: list[Region] = []
        if self.kind == "sam":
            out = self.pipe(to_pil(image), points_per_batch=64, points_per_crop=self.sam_grid)
            for mask, score in zip(out["masks"], np.asarray(out["scores"], dtype=float)):
                mask = np.asarray(mask, dtype=bool)
                if score >= self.min_score and mask.any():
                    regions.append(Region(mask, box_of(mask), float(score)))
        else:
            for det in self.pipe(to_pil(image), threshold=self.min_score):
                b = det["box"]
                box = (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]))
                regions.append(Region(box_mask(box, shape), box, float(det["score"]), det["label"]))
        self.last_seconds = time.perf_counter() - start
        return regions


def select_entity(regions: list[Region], shape: tuple[int, int], max_area: float = 0.95) -> Region | None:
    """The region most likely to be the photographed entity: large, central, confident.

    Regions covering more than `max_area` of the frame are treated as the whole
    scene and skipped. The cap is high on purpose: Imagenette churches often fill
    most of the frame and must not be thrown away as background.
    """
    h, w = shape
    best, best_score = None, -1.0
    for r in regions:
        if r.area > max_area or r.area == 0:
            continue
        cy, cx = ndimage.center_of_mass(r.mask)
        off_center = np.hypot((cy - h / 2) / h, (cx - w / 2) / w) / np.hypot(0.5, 0.5)
        score = r.score * np.sqrt(r.area) * (1.0 - off_center)
        if score > best_score:
            best, best_score = r, score
    return best


def structure_map(mask: np.ndarray | None, shape: tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """Entity mask -> soft map in [0, 1] (1 = entity), usable wherever cues expect depth.

    The smoothing gives cues a gradient band along the object outline, the
    signal that the depth-boundary cues found most useful. No region -> a flat
    map, which makes every gradient-based cue an identity.
    """
    if mask is None or not mask.any():
        return np.zeros(shape, dtype=np.float32)
    soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (soft / (soft.max() + 1e-8)).astype(np.float32)


class StructureCache:
    """Caches entity masks per (segmenter, image) on disk as packed bit arrays."""

    def __init__(self, segmenter: Segmenter, root: Path | None = None):
        self.segmenter = segmenter
        self.root = (root or CACHE_DIR / "segmentation") / segmenter.kind
        self.root.mkdir(parents=True, exist_ok=True)

    def entity_mask(self, image: np.ndarray, record: ImageRecord | None = None) -> np.ndarray | None:
        """Entity mask for `image`. Pass `record` only for a clean image, so it can be cached."""
        path = self.root / f"{record.image_id}.npz" if record else None
        if path is not None and path.exists():
            data = np.load(path)
            return np.unpackbits(data["bits"])[: int(np.prod(data["shape"]))].reshape(data["shape"]).astype(bool) \
                if data["found"] else None
        region = select_entity(self.segmenter(image), image.shape[:2])
        mask = None if region is None else region.mask
        if path is not None:
            bits = np.packbits(mask) if mask is not None else np.zeros(0, np.uint8)
            np.savez_compressed(path, bits=bits, shape=np.array(image.shape[:2]), found=mask is not None)
        return mask


class TextSegmenter:
    """Text-prompted soft mask from CLIPSeg: where in the image is "<prompt>"?

    The unprompted SAM picker (`select_entity`) often keeps one window, a door
    or the sky instead of a church that fills the frame. Naming the entity in a
    prompt fixes *which* region is the entity, the tool-augmented context idea.
    """

    def __init__(self, model_name: str = TEXT_SEGMENTER, device: str | None = None):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = get_device(device)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device).eval()

    def __call__(self, image: np.ndarray, prompt: str) -> np.ndarray:
        """Probability map HxW in [0, 1] that each pixel belongs to `prompt`."""
        import torch
        import torch.nn.functional as F

        inputs = self.processor(text=[prompt], images=[to_pil(image)], return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.reshape(1, 1, *inputs["pixel_values"].shape[-2:])
        prob = torch.sigmoid(F.interpolate(logits.float(), size=image.shape[:2], mode="bilinear",
                                           align_corners=False))[0, 0]
        return prob.cpu().numpy().astype(np.float32)


def threshold_text_mask(prob: np.ndarray, floor: float = 0.3, relative: float = 0.5) -> np.ndarray | None:
    """Binary entity mask from a CLIPSeg probability map; None if the prompt was not found."""
    if prob.max() < floor:
        return None
    mask = prob >= max(floor, relative * float(prob.max()))
    return mask if mask.any() else None


def regions_in_mask(regions: list[Region], guide: np.ndarray | None, min_inside: float = 0.5,
                    max_area: float = 0.95) -> np.ndarray | None:
    """Union of the SAM regions that lie mostly inside a guide mask (e.g. from a text prompt).

    Keeps SAM's crisp edges while the guide decides which parts are the entity.
    Whole-frame regions are skipped, as in `select_entity`.
    """
    if guide is None or not guide.any():
        return None
    union = np.zeros(guide.shape, dtype=bool)
    for r in regions:
        if r.area == 0 or r.area > max_area:
            continue
        if (r.mask & guide).sum() / r.mask.sum() >= min_inside:
            union |= r.mask
    return union if union.any() else None
