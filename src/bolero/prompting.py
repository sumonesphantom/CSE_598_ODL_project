"""In-context evidence for a VLM: serialize what the pipeline knows into a prompt.

Rather than sending the VLM the raw image alone, we hand it the structured
evidence the rest of the pipeline already computes: classifier candidates,
the detected perturbation, the entity region and depth layout. The VLM is then
asked to reason about the *context* of the damaged region, not to repaint it.

This module only builds the prompt text; which VLM receives it is up to the caller.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .conditions import TAXONOMIES
from .config import (
    JEV_IMAGE_MAX_PIXELS,
    JEV_IMAGE_MAX_SIDE,
    JEV_IMAGE_MIN_PIXELS,
    JEV_IMAGE_MIN_SIDE,
)
from .data import to_uint8
from .depth import near_mask
from .perturbations import BLACK, GRAY
from .restoration import estimate_noise


@dataclass
class RegionEvidence:
    box: tuple[int, int, int, int]      # x0, y0, x1, y1 in pixels
    area: float                         # fraction of the frame
    mean_depth: float | None = None     # 1 = nearest
    label: str | None = None


@dataclass
class SceneEvidence:
    image_size: tuple[int, int]                                   # (width, height)
    candidates: list[tuple[str, float]]                           # classifier top-k
    perturbation: str | None = None                               # detector decision
    perturbation_prob: float | None = None
    entity: RegionEvidence | None = None
    damaged: RegionEvidence | None = None                         # known/suspected corrupted region
    depth_layout: dict[str, float] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def to_state(self, include_detector: bool = True) -> dict:
        """JSON-safe evidence for a decision model, free of ground-truth labels."""
        state: dict = {
            "image": {"width": int(self.image_size[0]), "height": int(self.image_size[1])},
            "classifier_candidates": [
                {"label": label, "prob": round(float(prob), 4)}
                for label, prob in self.candidates
            ],
        }
        if include_detector and self.perturbation is not None:
            probability = 0.0 if self.perturbation_prob is None else float(self.perturbation_prob)
            state["detector"] = {"label": self.perturbation, "probability": round(probability, 4)}
        if self.entity is not None:
            state["entity_region"] = asdict(self.entity)
        if self.damaged is not None:
            state["damaged_region"] = asdict(self.damaged)
        if self.depth_layout:
            state["depth_layout"] = dict(self.depth_layout)
        if self.stats:
            state["image_statistics"] = dict(self.stats)
        return state


def depth_layout(depth: np.ndarray, near_fraction: float = 0.3) -> dict[str, float]:
    """Coarse depth summary a language model can read: where the near content sits."""
    near = near_mask(depth, near_fraction)
    h, w = depth.shape
    ys, xs = np.nonzero(near)
    return {
        "near_center_x": round(float(xs.mean() / w), 3) if len(xs) else 0.5,
        "near_center_y": round(float(ys.mean() / h), 3) if len(ys) else 0.5,
        "depth_spread": round(float(np.percentile(depth, 95) - np.percentile(depth, 5)), 3),
    }


def region_evidence(mask: np.ndarray, depth: np.ndarray | None = None, label: str | None = None) -> RegionEvidence:
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, 0, 0)
    mean_depth = round(float(depth[mask].mean()), 3) if depth is not None and mask.any() else None
    return RegionEvidence(box, round(float(mask.mean()), 4), mean_depth, label)


def _blockiness(lum: np.ndarray, axis: int) -> float:
    """Ratio of gradients on 8-pixel block boundaries to elsewhere (JPEG cue)."""
    diffs = np.abs(np.diff(lum, axis=axis)).mean(axis=1 - axis)
    idx = np.arange(diffs.size)
    at_block = diffs[(idx % 8) == 7]
    elsewhere = diffs[(idx % 8) != 7]
    return float(at_block.mean() / (elsewhere.mean() + 1e-6)) if at_block.size else 1.0


def image_statistics(image: np.ndarray) -> dict:
    """Compact, JSON-safe statistics describing appearance and corruption cues."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    h, w = image.shape[:2]
    lum = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    lap = cv2.Laplacian(lum, cv2.CV_32F)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx**2 + gy**2)
    p01, p50, p99 = np.percentile(lum, [1, 50, 99])
    top = lum[: max(h // 3, 1)].mean()
    bottom = lum[-max(h // 3, 1):].mean()

    stats = {
        "width": int(w),
        "height": int(h),
        "luminance_mean": float(lum.mean()),
        "luminance_std": float(lum.std()),
        "luminance_p01": float(p01),
        "luminance_p50": float(p50),
        "luminance_p99": float(p99),
        "channel_mean_r": float(image[..., 0].mean()),
        "channel_mean_g": float(image[..., 1].mean()),
        "channel_mean_b": float(image[..., 2].mean()),
        "channel_std_r": float(image[..., 0].std()),
        "channel_std_g": float(image[..., 1].std()),
        "channel_std_b": float(image[..., 2].std()),
        "saturation_mean": float(sat.mean()),
        "saturation_low_fraction": float((sat < 0.05).mean()),
        "laplacian_variance": float(lap.var()),
        "gradient_mean": float(grad.mean()),
        "gradient_p95": float(np.percentile(grad, 95)),
        "noise_sigma": float(estimate_noise(image)),
        "black_fill_fraction": float(np.all(np.abs(image - BLACK) < 1e-4, axis=-1).mean()),
        "gray_fill_fraction": float(np.all(np.abs(image - GRAY) < 1e-4, axis=-1).mean()),
        "blockiness_x": _blockiness(lum, 1),
        "blockiness_y": _blockiness(lum, 0),
        "top_bottom_luminance_diff": float(top - bottom),
    }
    return {key: (value if isinstance(value, int) else round(float(value), 4))
            for key, value in stats.items()}


def _reflect_pad(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Reflect-pad an HxWxC image for any pad size, including thin images."""
    h, w = image.shape[:2]

    def indices(n: int, before: int, after: int) -> np.ndarray:
        idx = np.arange(-before, n + after)
        idx = np.abs(idx) % (2 * n)
        return np.where(idx >= n, 2 * n - 1 - idx, idx)

    ys = indices(h, top, bottom)
    xs = indices(w, left, right)
    return np.ascontiguousarray(image[ys][:, xs])


def prepare_image(
    image: np.ndarray,
    min_side: int = JEV_IMAGE_MIN_SIDE,
    max_side: int = JEV_IMAGE_MAX_SIDE,
    min_pixels: int = JEV_IMAGE_MIN_PIXELS,
    max_pixels: int = JEV_IMAGE_MAX_PIXELS,
    quality: int = 95,
) -> str:
    """Encode a float RGB image as a baseline JPEG data URL within API bounds."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    h, w = image.shape[:2]
    max_aspect = max_side / min_side

    # An extreme aspect ratio can make the side and pixel bounds incompatible;
    # pad the short side with reflection until a scale can satisfy both.
    if w / h > max_aspect:
        target_h = int(math.ceil(w / max_aspect))
        deficit = target_h - h
        image = _reflect_pad(image, deficit // 2, deficit - deficit // 2, 0, 0)
    elif h / w > max_aspect:
        target_w = int(math.ceil(h / max_aspect))
        deficit = target_w - w
        image = _reflect_pad(image, 0, 0, deficit // 2, deficit - deficit // 2)
    h, w = image.shape[:2]

    scale_min = max(min_side / w, min_side / h, math.sqrt(min_pixels / (w * h)))
    scale_max = min(max_side / w, max_side / h, math.sqrt(max_pixels / (w * h)))
    assert scale_min <= scale_max + 1e-9, "image bounds are incompatible after padding"

    scale = min(max(1.0, scale_min), scale_max)
    width = min(max(int(round(w * scale)), min_side), max_side)
    height = min(max(int(round(h * scale)), min_side), max_side)

    if width * height < min_pixels:
        factor = math.sqrt(min_pixels / (width * height)) * 1.001
        width = min(max_side, int(math.ceil(width * factor)))
        height = min(max_side, int(math.ceil(height * factor)))
    if width * height > max_pixels:
        factor = math.sqrt(max_pixels / (width * height)) * 0.999
        width = max(min_side, int(math.floor(width * factor)))
        height = max(min_side, int(math.floor(height * factor)))

    resized = Image.fromarray(to_uint8(image)).resize((width, height), Image.LANCZOS)
    assert min_side <= resized.width <= max_side
    assert min_side <= resized.height <= max_side
    assert min_pixels <= resized.width * resized.height <= max_pixels

    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _region_box(region) -> tuple[int, int, int, int] | None:
    if region is None:
        return None
    box = region.box if hasattr(region, "box") else region
    return tuple(int(value) for value in box)


def draw_regions(image: np.ndarray, entity=None, damaged=None) -> np.ndarray:
    """Overlay labelled entity/damaged boxes; returns float32 RGB in [0, 1]."""
    canvas = Image.fromarray(to_uint8(image))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    w, h = canvas.size

    overlays = [
        (_region_box(entity), (0, 255, 0), "1 entity"),
        (_region_box(damaged), (255, 0, 0), "2 damaged"),
    ]
    for box, color, text in overlays:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        draw.rectangle(
            [max(0, x0), max(0, y0), min(w - 1, max(0, x1 - 1)), min(h - 1, max(0, y1 - 1))],
            outline=color,
            width=2,
        )
        text_box = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
        label_x = min(max(0, x0), max(0, w - text_w - 4))
        label_y = max(0, y0 - text_h - 4)
        draw.rectangle(
            [label_x, label_y, min(w - 1, label_x + text_w + 3), min(h - 1, label_y + text_h + 3)],
            fill=color,
        )
        draw.text((label_x + 2, label_y + 1), text, fill=(255, 255, 255), font=font)

    return np.asarray(canvas, dtype=np.float32) / 255.0


def build_jev_questions(taxonomy: str = "family") -> dict:
    """Question schema for the JEV decisions API (no ground-truth fields)."""
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
    return {
        "degraded": {
            "type": "noul",
            "instructions": (
                "Decide whether this view is degraded compared with a clean view of the "
                "same scene. Answer true if any visible corruption is present."
            ),
            "criteria": {
                "true": "Visible degradation such as blur, noise, occlusion, color change, or compression.",
                "false": "The view looks clean, with no visible degradation.",
            },
        },
        "condition": {
            "type": "choice",
            "instructions": "Choose the single condition that best describes the visible degradation.",
            "criteria": dict(TAXONOMIES[taxonomy]),
        },
        "severity": {
            "type": "score",
            "instructions": "Rate how severe the degradation is.",
            "criteria": ["mild", "moderate", "severe"],
        },
    }


def build_jev_payload(
    evidence: SceneEvidence,
    taxonomy: str = "family",
    image: np.ndarray | None = None,
    exemplars=None,
) -> dict:
    """Assemble the request body for a JEV decision (no model or API key)."""
    state: dict | list = {"evidence": evidence.to_state()}
    if exemplars is not None:
        state["exemplars"] = exemplars
    if image is not None:
        state = [state, {"type": "image_url", "image_url": {"url": prepare_image(image)}}]
    return {"state": state, "questions": build_jev_questions(taxonomy)}


INSTRUCTION = (
    "You are given an image that may be degraded, together with structured evidence "
    "extracted from it by other models. Coordinates are pixels, origin top-left; depth is "
    "relative, 1 = nearest. Using the image and the evidence, identify the main entity "
    "and describe what the damaged region most likely contains in terms of object "
    "identity, shape and spatial relation to the entity. Do not invent color or texture. "
    "Answer as JSON with keys: entity, damaged_region_content, confidence (0-1), rationale."
)


def build_prompt(evidence: SceneEvidence, instruction: str = INSTRUCTION) -> str:
    payload = asdict(evidence)
    payload["candidates"] = [{"label": c, "prob": round(p, 3)} for c, p in evidence.candidates]
    payload = {k: v for k, v in payload.items() if v not in (None, {}, [])}
    return f"{instruction}\n\nEVIDENCE:\n{json.dumps(payload, indent=2)}"
