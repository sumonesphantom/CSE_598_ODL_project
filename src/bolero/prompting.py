"""In-context evidence for a VLM: serialize what the pipeline knows into a prompt.

Rather than sending the VLM the raw image alone, we hand it the structured
evidence the rest of the pipeline already computes: classifier candidates,
the detected perturbation, the entity region and depth layout. The VLM is then
asked to reason about the *context* of the damaged region, not to repaint it.

This module only builds the prompt text; which VLM receives it is up to the caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .depth import near_mask


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
