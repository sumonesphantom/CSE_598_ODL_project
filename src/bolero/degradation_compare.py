"""Helpers for comparing physics, generative and real degradation on one frozen detector.

Bootstrap intervals for mAP50 resample whole images. Box matching never crosses images, so
each detection's true/false-positive flag is fixed once its image is known (greedy matching
inside an image follows score order, the same order the pooled ranking uses). The flags are
computed once per image and class, and every resample only re-ranks and re-integrates,
giving exactly `detection_eval.average_precision` at a fraction of the cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection_eval import Detections, iou_matrix
from .weather_data import CLASSES

GRID = np.linspace(0, 1, 101)


@dataclass
class Matches:
    """One class on one image: detection scores, their TP flags, and the number of GT boxes."""
    scores: np.ndarray
    tp: np.ndarray
    n_gt: int


def match_image(gt_boxes: np.ndarray, gt_labels: list[str], det: Detections, cls: str,
                iou_thr: float = 0.5) -> Matches:
    gt = gt_boxes[np.asarray(gt_labels) == cls] if len(gt_labels) else np.zeros((0, 4), np.float32)
    keep = np.asarray(det.labels) == cls if det.labels else np.zeros(0, bool)
    boxes, scores = det.boxes[keep], det.scores[keep]
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    tp = np.zeros(len(scores), bool)
    if len(gt) and len(scores):
        ious = iou_matrix(boxes, gt)
        used = np.zeros(len(gt), bool)
        for k in range(len(scores)):
            cand = np.where(used, -1.0, ious[k])
            best = int(cand.argmax())
            if cand[best] >= iou_thr:
                used[best] = True
                tp[k] = True
    return Matches(scores.astype(np.float64), tp, len(gt))


def ap_from(matches: list[Matches]) -> float:
    """101-point interpolated AP over a list of per-image matches (NaN without ground truth)."""
    n_gt = sum(m.n_gt for m in matches)
    if n_gt == 0:
        return float("nan")
    scores = np.concatenate([m.scores for m in matches]) if matches else np.zeros(0)
    if not len(scores):
        return 0.0
    tp = np.concatenate([m.tp for m in matches]).astype(float)
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-9))[::-1])[::-1]
    idx = np.searchsorted(recall, GRID, side="left")
    return float(np.mean(np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)))


def image_matches(gt_boxes: list[np.ndarray], gt_labels: list[list[str]], dets: list[Detections],
                  classes: tuple[str, ...] = CLASSES) -> dict[str, list[Matches]]:
    return {c: [match_image(b, l, d, c) for b, l, d in zip(gt_boxes, gt_labels, dets)] for c in classes}


def map50(per_class: dict[str, list[Matches]], idx: np.ndarray | None = None) -> float:
    """Mean AP50 over classes present in the (optionally resampled) images."""
    aps = []
    for ms in per_class.values():
        sel = ms if idx is None else [ms[i] for i in idx]
        ap = ap_from(sel)
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def class_ap50(per_class: dict[str, list[Matches]]) -> dict[str, float]:
    return {c: ap_from(ms) for c, ms in per_class.items()}


def bootstrap_map50(per_class: dict[str, list[Matches]], n_boot: int = 500, seed: int = 42,
                    reference: dict[str, list[Matches]] | None = None) -> tuple[float, float, float]:
    """mAP50 (or, with `reference`, mAP50 minus the reference's on the same images) with a 95 % CI."""
    n = len(next(iter(per_class.values())))
    value = map50(per_class) - (map50(reference) if reference else 0.0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(map50(per_class, idx) - (map50(reference, idx) if reference else 0.0))
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    return value, float(low), float(high)


def kid(x: np.ndarray, y: np.ndarray) -> float:
    """Kernel inception distance (unbiased MMD^2, cubic polynomial kernel) between embedding sets.

    Used on CLIP embeddings: lower means the two sets look more alike. Unbiased, so it is
    usable at a few hundred images where a Frechet distance is badly biased.
    """
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1.0) ** 3
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def spearman(a, b) -> float:
    """Rank correlation, NaN-safe (pairs with a NaN are dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
