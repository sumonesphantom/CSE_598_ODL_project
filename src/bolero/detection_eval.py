"""Object-detection evaluation for the weather track: a frozen detector and COCO-style mAP.

`average_precision` follows COCO: detections sorted by score, greedy matching to the
highest-IoU unmatched ground truth of the same class, 101-point interpolated
precision. mAP50 uses IoU 0.5; mAP50-95 averages IoU 0.50:0.05:0.95. Classes with no
ground truth in the evaluated subset are left out of the mean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import SEGMENTERS, get_device
from .weather_data import CLASSES

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


@dataclass
class Detections:
    boxes: np.ndarray    # (N, 4) x1 y1 x2 y2 pixels
    scores: np.ndarray   # (N,)
    labels: list[str]
    seconds: float = 0.0


class FrozenDetector:
    """facebook/detr-resnet-50 (COCO), kept to the six weather-track classes. Never fine-tuned."""

    def __init__(self, model: str = "detr", device: str | None = None, min_score: float = 0.05):
        from transformers import pipeline

        self.min_score = min_score
        self.pipe = pipeline("object-detection", model=SEGMENTERS.get(model, model), revision="no_timm",
                             framework="pt", device=get_device(device))

    def __call__(self, image) -> Detections:
        """`image`: a PIL image."""
        start = time.perf_counter()
        boxes, scores, labels = [], [], []
        for det in self.pipe(image, threshold=self.min_score):
            label = det["label"].lower()
            if label in CLASSES:
                b = det["box"]
                boxes.append([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
                scores.append(det["score"])
                labels.append(label)
        return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                          labels, time.perf_counter() - start)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) IoU of x1 y1 x2 y2 boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(gts: list[np.ndarray], dets: list[tuple[np.ndarray, np.ndarray]],
                      iou_thr: float) -> tuple[float, float]:
    """One class. `gts[i]`: (G_i, 4) boxes; `dets[i]`: ((D_i, 4) boxes, (D_i,) scores).

    Returns (AP, recall at the end of the ranked list); NaN when there is no ground truth.
    """
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return float("nan"), float("nan")
    entries = [(s, i, j) for i, (_, sc) in enumerate(dets) for j, s in enumerate(sc)]
    entries.sort(key=lambda e: -e[0])
    used = [np.zeros(len(g), bool) for g in gts]
    ious = [iou_matrix(d[0], g) for d, g in zip(dets, gts)]
    tp = np.zeros(len(entries))
    for k, (_, i, j) in enumerate(entries):
        if len(gts[i]) == 0:
            continue
        cand = np.where(used[i], -1.0, ious[i][j])
        best = int(cand.argmax())
        if cand[best] >= iou_thr:
            used[i][best] = True
            tp[k] = 1
    if not len(entries):
        return 0.0, 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    ap = np.mean([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(ap), float(recall[-1])


def evaluate(gt_boxes: list[np.ndarray], gt_labels: list[list[str]],
             det: list[Detections], classes: tuple[str, ...] = CLASSES) -> dict:
    """mAP50, mAP50-95 and per-class AP50 / recall50 over a set of images."""
    per_class = {}
    for c in classes:
        gts = [b[np.asarray(l) == c] if len(l) else np.zeros((0, 4)) for b, l in zip(gt_boxes, gt_labels)]
        dets = []
        for d in det:
            m = np.asarray(d.labels) == c if d.labels else np.zeros(0, bool)
            dets.append((d.boxes[m], d.scores[m]))
        aps = [average_precision(gts, dets, t)[0] for t in IOU_THRESHOLDS]
        ap50, rec50 = average_precision(gts, dets, 0.5)
        per_class[c] = {"n_gt": int(sum(len(g) for g in gts)), "ap50": ap50,
                        "ap50_95": float(np.mean(aps)) if not np.isnan(aps[0]) else float("nan"),
                        "recall50": rec50}
    present = [v for v in per_class.values() if v["n_gt"] > 0]
    return {
        "map50": float(np.mean([v["ap50"] for v in present])) if present else float("nan"),
        "map50_95": float(np.mean([v["ap50_95"] for v in present])) if present else float("nan"),
        "per_class": per_class,
    }
