"""Did restoration invent, lose or relabel objects? Detections before vs. after, against ground truth.

mAP says whether a restorer helps on average; it does not say whether the restorer is
making things up. A generative restorer can raise mAP while painting a car that isn't
there, or lower it by erasing a pedestrian. These counts separate the two.

All counts are taken at one operating score threshold (`thr`, default 0.5): the boxes an
operator would actually see. For each image, detections on the raw input ("before") and on
the restored input ("after") are matched to the ground truth (same class, IoU >= 0.5,
greedy by score, as in COCO matching):

  lost       ground-truth objects found before but not after       (restoration erased them)
  gained     ground-truth objects found after but not before       (restoration revealed them)
  invented   after-boxes that match no ground truth and overlap no before-box (IoU < 0.5, any
             class): a new false positive in a place the detector saw nothing before
  relabeled  correct before-boxes whose best-overlapping after-box (IoU >= 0.5) has another class
  cleared    before false positives with no overlapping after-box (restoration removed clutter)

"invented" is measured against the before-boxes as well as the ground truth because
DAWN and XWOD do not label every object; a false positive that was already there before
restoration is a labeling gap or a detector habit, not something the restorer made up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_eval import Detections, average_precision, iou_matrix

MATCH_IOU = 0.5


def keep_above(det: Detections, thr: float) -> Detections:
    m = det.scores >= thr
    return Detections(det.boxes[m], det.scores[m], [l for l, k in zip(det.labels, m) if k], det.seconds)


def match_to_gt(det: Detections, gt_boxes: np.ndarray, gt_labels: list[str],
                iou_thr: float = MATCH_IOU) -> tuple[np.ndarray, np.ndarray]:
    """Greedy same-class matching by score. Returns (gt_found (G,), det_is_tp (D,))."""
    found = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(det.scores), bool)
    if len(gt_boxes) == 0 or len(det.scores) == 0:
        return found, tp
    ious = iou_matrix(det.boxes, gt_boxes)
    same = np.asarray(det.labels)[:, None] == np.asarray(gt_labels)[None, :]
    for j in np.argsort(-det.scores, kind="stable"):
        cand = np.where(same[j] & ~found, ious[j], -1.0)
        g = int(cand.argmax())
        if cand[g] >= iou_thr:
            found[g] = tp[j] = True
    return found, tp


def compare_image(before: Detections, after: Detections, gt_boxes: np.ndarray, gt_labels: list[str],
                  thr: float = 0.5) -> dict:
    """Hallucination counts for one image (see the module docstring)."""
    before, after = keep_above(before, thr), keep_above(after, thr)
    found_b, tp_b = match_to_gt(before, gt_boxes, gt_labels)
    found_a, tp_a = match_to_gt(after, gt_boxes, gt_labels)
    cross = iou_matrix(after.boxes, before.boxes)            # (A, B)
    overlaps_before = (cross >= MATCH_IOU).any(axis=1) if cross.size else np.zeros(len(after.scores), bool)
    overlaps_after = (cross >= MATCH_IOU).any(axis=0) if cross.size else np.zeros(len(before.scores), bool)

    relabeled = 0
    for j in np.where(tp_b)[0]:
        if cross.shape[0] == 0:
            break
        k = int(cross[:, j].argmax())
        if cross[k, j] >= MATCH_IOU and after.labels[k] != before.labels[j]:
            relabeled += 1

    return {
        "n_gt": len(gt_boxes),
        "found_before": int(found_b.sum()), "found_after": int(found_a.sum()),
        "lost": int((found_b & ~found_a).sum()), "gained": int((~found_b & found_a).sum()),
        "boxes_before": len(before.scores), "boxes_after": len(after.scores),
        "fp_before": int((~tp_b).sum()), "fp_after": int((~tp_a).sum()),
        "invented": int((~tp_a & ~overlaps_before).sum()),
        "cleared": int((~tp_b & ~overlaps_after).sum()),
        "relabeled": relabeled,
    }


def summarize(per_image: pd.DataFrame) -> dict:
    """Rates over a set of images (rows from `compare_image`)."""
    s = per_image.sum(numeric_only=True)
    missed_before = s["n_gt"] - s["found_before"]
    return {
        "lost_rate": s["lost"] / max(s["found_before"], 1),          # share of found objects erased
        "gained_rate": s["gained"] / max(missed_before, 1),          # share of missed objects recovered
        "net_found": int(s["gained"] - s["lost"]),
        "invented_per_100img": 100 * s["invented"] / max(len(per_image), 1),
        "invented_share": s["invented"] / max(s["boxes_after"], 1),  # share of shown boxes that are new FPs
        "relabel_rate": s["relabeled"] / max(s["found_before"], 1),
        "cleared_rate": s["cleared"] / max(s["fp_before"], 1),
        "images_with_invented": float((per_image["invented"] > 0).mean()),
    }


def map50(gt_boxes: list[np.ndarray], gt_labels: list[list[str]], dets: list[Detections],
          classes: tuple[str, ...]) -> float:
    """mAP at IoU 0.5 only (cheap enough to bootstrap)."""
    aps = []
    for c in classes:
        gts = [b[np.asarray(l) == c] if len(l) else np.zeros((0, 4)) for b, l in zip(gt_boxes, gt_labels)]
        if sum(len(g) for g in gts) == 0:
            continue
        ds = []
        for d in dets:
            m = np.asarray(d.labels) == c if d.labels else np.zeros(0, bool)
            ds.append((d.boxes[m], d.scores[m]))
        aps.append(average_precision(gts, ds, 0.5)[0])
    return float(np.mean(aps)) if aps else float("nan")


def paired_bootstrap_map50(gt_boxes, gt_labels, before, after, classes, reps: int = 200,
                           seed: int = 0) -> tuple[float, float, float]:
    """mAP50(after) - mAP50(before) with a 95% CI from resampling images (same draw for both)."""
    delta = map50(gt_boxes, gt_labels, after, classes) - map50(gt_boxes, gt_labels, before, classes)
    if reps <= 0:
        return delta, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n, draws = len(gt_boxes), []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        pick = lambda xs: [xs[k] for k in i]
        draws.append(map50(pick(gt_boxes), pick(gt_labels), pick(after), classes)
                     - map50(pick(gt_boxes), pick(gt_labels), pick(before), classes))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return delta, float(lo), float(hi)
