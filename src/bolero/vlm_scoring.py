"""Spatial and answer scoring for the local VLM.

The VLM is asked to name the entity and to describe what the damaged region
contains. This module turns free text into comparable labels and checks the
VLM's spatial claims against geometry computed from boxes, so a plausible
sounding but geometrically wrong answer is not rewarded.
"""

from __future__ import annotations

import re

import numpy as np

from .deciders import parse_json_object
from .perturbations import BLACK, GRAY

RELATIONS = ("overlaps_entity", "adjacent", "background", "unclear")

# Conservative aliases for the ten Imagenette classes. Each list contains only
# words a VLM is likely to use for that exact class, not related categories.
IMAGENETTE_ALIASES: dict[str, list[str]] = {
    "tench": ["tench", "fish", "tinca"],
    "English springer": ["english springer", "springer", "springer spaniel", "spaniel"],
    "cassette player": ["cassette player", "cassette", "tape player"],
    "chain saw": ["chain saw", "chainsaw"],
    "church": ["church", "chapel", "cathedral"],
    "French horn": ["french horn", "horn"],
    "garbage truck": ["garbage truck", "truck", "refuse truck", "dustcart"],
    "gas pump": ["gas pump", "pump", "fuel pump", "petrol pump"],
    "golf ball": ["golf ball", "ball"],
    "parachute": ["parachute", "parachutist"],
}

_GRAY_PERTURBATIONS = {"foreground_occlusion", "boundary_erase", "background_removal"}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str | None) -> str:
    """Lowercase, replace non-alphanumeric characters with spaces, collapse space."""
    if not isinstance(text, str):
        return ""
    return " ".join(_NON_ALNUM.sub(" ", text.lower()).split())


def _candidate_phrases(target: str | None) -> list[str]:
    """Phrases worth testing for a target: the target, comma synonyms, and aliases."""
    if not isinstance(target, str) or not target.strip():
        return []
    phrases: list[str] = []
    for piece in [target, *target.split(",")]:
        normalized = normalize_text(piece)
        if not normalized:
            continue
        phrases.append(normalized)
        aliases = IMAGENETTE_ALIASES.get(piece.strip()) or IMAGENETTE_ALIASES.get(
            target.strip()
        )
        if aliases:
            phrases += [normalize_text(alias) for alias in aliases]
    # Also accept the canonical class name when the target is an ImageNet label
    # such as "tench, Tinca tinca": the first comma part is the class name.
    first = normalize_text(target.split(",")[0])
    if first in IMAGENETTE_ALIASES:
        phrases += [normalize_text(alias) for alias in IMAGENETTE_ALIASES[first]]

    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)
    return unique


def _contains_phrase(text: str, phrase: str) -> bool:
    tokens = phrase.split()
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return False
    return f" {phrase} " in f" {text} "


def match_label(text: str | None, target: str | None) -> bool:
    """Whether free text names the target class or one of its conservative aliases.

    A match requires a token-boundary phrase match against the target, a
    comma-separated ImageNet synonym, or an alias, or token Jaccard >= 0.6.
    """
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False
    text_tokens = set(normalized_text.split())
    for phrase in _candidate_phrases(target):
        if _contains_phrase(normalized_text, phrase):
            return True
        phrase_tokens = set(phrase.split())
        union = text_tokens | phrase_tokens
        if union and len(text_tokens & phrase_tokens) / len(union) >= 0.6:
            return True
    return False


def _valid_box(box) -> bool:
    if box is None:
        return False
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0)


def _area(box) -> float:
    x0, y0, x1, y1 = (float(value) for value in box)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(a, b) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes; 0 for invalid boxes."""
    if not (_valid_box(a) and _valid_box(b)):
        return 0.0
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = iw * ih
    union = _area(a) + _area(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def overlap_fraction(inner, outer) -> float:
    """Fraction of `inner`'s area that lies inside `outer`; 0 for invalid boxes."""
    if not (_valid_box(inner) and _valid_box(outer)):
        return 0.0
    ix0, iy0, ix1, iy1 = (float(value) for value in inner)
    ox0, oy0, ox1, oy1 = (float(value) for value in outer)
    iw = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    ih = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    inner_area = _area(inner)
    return float((iw * ih) / inner_area) if inner_area > 0 else 0.0


def relation_bucket(entity_box, damaged_box, shape) -> str:
    """Geometric relation between the entity box and the damaged box.

    "overlaps_entity" for meaningful intersection, "adjacent" when the boxes are
    within 5% of the image diagonal of each other, "background" otherwise.
    """
    if not (_valid_box(entity_box) and _valid_box(damaged_box)):
        return "unclear"
    if box_iou(entity_box, damaged_box) > 0.1:
        return "overlaps_entity"

    ex0, ey0, ex1, ey1 = (float(value) for value in entity_box)
    dx0, dy0, dx1, dy1 = (float(value) for value in damaged_box)
    cx, cy = (dx0 + dx1) / 2.0, (dy0 + dy1) / 2.0
    if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
        return "overlaps_entity"

    try:
        height, width = float(shape[0]), float(shape[1])
    except (TypeError, ValueError, IndexError):
        return "unclear"
    diagonal = float(np.hypot(height, width))
    gap_x = max(0.0, ex0 - dx1, dx0 - ex1)
    gap_y = max(0.0, ey0 - dy1, dy0 - ey1)
    return "adjacent" if float(np.hypot(gap_x, gap_y)) <= 0.05 * diagonal else "background"


def damaged_mask(
    perturbation: str,
    degraded: np.ndarray,
    clean: np.ndarray,
    depth: np.ndarray | None,
) -> np.ndarray | None:
    """Pixels the perturbation filled with a flat color, or None if not applicable.

    `clean` and `depth` are accepted for API symmetry with the perturbation
    pipeline; only the degraded image identifies flat-fill damage today.
    """
    image = np.asarray(degraded, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    if perturbation == "region_removal":
        fill = BLACK
    elif perturbation in _GRAY_PERTURBATIONS:
        fill = GRAY
    else:
        return None
    return np.all(np.abs(image - fill) < 1e-4, axis=-1)


def parse_vlm_answer(text: str | None) -> dict | None:
    """Parse and normalize the VLM JSON answer; None when no object was found."""
    parsed = parse_json_object(text)
    if parsed is None:
        return None

    entity = parsed.get("entity")
    damaged = parsed.get("damaged_region_content")
    relation = parsed.get("relation")
    confidence = parsed.get("confidence")
    rationale = parsed.get("rationale")

    if isinstance(confidence, bool):
        confidence = None
    else:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            confidence = None

    return {
        "entity": entity if isinstance(entity, str) else "",
        "damaged_region_content": damaged if isinstance(damaged, str) else "",
        "relation": relation if relation in RELATIONS else "unclear",
        "confidence": confidence,
        "rationale": rationale if isinstance(rationale, str) else "",
    }


def score_case(
    answer: dict,
    true_class: str,
    clean_prediction: str | None,
    degraded_prediction: str | None,
    entity_box,
    damaged_box,
    shape,
) -> dict:
    """Score one VLM answer against the true class and the boxes we already know.

    Entity utility follows the paper: the VLM is useful if it names the true
    class (or, as a fallback, the clean classifier's prediction). Damage credit
    is only awarded when the damaged box substantially overlaps the entity box,
    since otherwise naming the class in that region is not evidence of reasoning.
    Degraded correctness is the recognizer being right on the degraded view.
    """
    entity_text = answer.get("entity", "")
    damage_text = answer.get("damaged_region_content", "")
    answer_relation = answer.get("relation", "unclear")
    if answer_relation not in RELATIONS:
        answer_relation = "unclear"

    entity_hit = bool(match_label(entity_text, true_class) or match_label(entity_text, clean_prediction))

    boxes_valid = _valid_box(entity_box) and _valid_box(damaged_box)
    damage_eligible = bool(boxes_valid and overlap_fraction(damaged_box, entity_box) > 0.3)
    damage_hit = bool(damage_eligible and match_label(damage_text, true_class))

    relation_geometric = relation_bucket(entity_box, damaged_box, shape)
    relation_hit = (
        None
        if relation_geometric == "unclear"
        else bool(answer_relation == relation_geometric)
    )

    if relation_geometric != "unclear":
        relation_hybrid = relation_geometric
        relation_hybrid_source = "geometric"
    elif answer_relation != "unclear":
        relation_hybrid = answer_relation
        relation_hybrid_source = "vlm"
    else:
        relation_hybrid = "unclear"
        relation_hybrid_source = "unclear"

    degraded_correct = bool(
        match_label(degraded_prediction, true_class)
        or (normalize_text(degraded_prediction) == normalize_text(true_class))
    )

    return {
        "entity_hit": bool(entity_hit),
        "damage_eligible": damage_eligible,
        "damage_hit": damage_hit,
        "relation_geometric": relation_geometric,
        "relation_hit": relation_hit,
        "relation_hybrid": relation_hybrid,
        "relation_hybrid_source": relation_hybrid_source,
        "utility": bool(match_label(entity_text, true_class)),
        "degraded_correct": degraded_correct,
    }
