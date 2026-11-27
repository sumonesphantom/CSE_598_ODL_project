"""Hallucination counts and the paired mAP50 bootstrap (no model loads)."""

import numpy as np
import pandas as pd

from bolero.detection_eval import Detections
from bolero.hallucination import compare_image, map50, match_to_gt, paired_bootstrap_map50, summarize
from bolero.weather_data import CLASSES

GT = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], np.float32)
GT_LABELS = ["car", "person", "truck"]


def det(boxes, labels, scores=None):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.full(len(boxes), 0.9, np.float32) if scores is None else np.asarray(scores, np.float32)
    return Detections(boxes, scores, list(labels))


def test_match_same_class_only_and_no_double_use():
    d = det([[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]], ["car", "car", "truck"], [0.9, 0.8, 0.7])
    found, tp = match_to_gt(d, GT, GT_LABELS)
    assert found.tolist() == [True, False, False]       # person box labeled truck does not count
    assert tp.tolist() == [True, False, False]          # duplicate car is a false positive


def test_identical_detections_change_nothing():
    before = det(GT, GT_LABELS)
    out = compare_image(before, before, GT, GT_LABELS)
    assert out["lost"] == out["gained"] == out["invented"] == out["relabeled"] == out["cleared"] == 0
    assert out["found_before"] == out["found_after"] == 3


def test_lost_gained_invented_relabeled_cleared():
    # before: car + person found, a false positive at x=60, truck missed
    before = det([[0, 0, 10, 10], [20, 0, 30, 10], [60, 0, 70, 10]], ["car", "person", "car"])
    # after: car relabeled truck (lost + relabeled), person kept, truck found (gained),
    # FP at x=60 gone (cleared), new FP at x=80 (invented)
    after = det([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [80, 0, 90, 10]],
                ["truck", "person", "truck", "bus"])
    out = compare_image(before, after, GT, GT_LABELS)
    assert (out["lost"], out["gained"], out["invented"], out["relabeled"], out["cleared"]) == (1, 1, 1, 1, 1)
    # the relabeled car box is a false positive but overlaps a before-box: not "invented"
    assert out["fp_after"] == 2


def test_threshold_hides_weak_boxes():
    before = det(GT, GT_LABELS)
    after = det(np.vstack([GT, [[80, 0, 90, 10]]]), GT_LABELS + ["car"], [0.9, 0.9, 0.9, 0.3])
    assert compare_image(before, after, GT, GT_LABELS, thr=0.5)["invented"] == 0
    assert compare_image(before, after, GT, GT_LABELS, thr=0.2)["invented"] == 1


def test_summarize_rates():
    rows = pd.DataFrame([
        {"n_gt": 4, "found_before": 2, "found_after": 3, "lost": 1, "gained": 2, "boxes_before": 3,
         "boxes_after": 4, "fp_before": 1, "fp_after": 1, "invented": 1, "cleared": 1, "relabeled": 0},
        {"n_gt": 2, "found_before": 2, "found_after": 2, "lost": 0, "gained": 0, "boxes_before": 2,
         "boxes_after": 2, "fp_before": 0, "fp_after": 0, "invented": 0, "cleared": 0, "relabeled": 1},
    ])
    s = summarize(rows)
    assert s["lost_rate"] == 1 / 4 and s["gained_rate"] == 2 / 2 and s["net_found"] == 1
    assert s["invented_per_100img"] == 50 and s["images_with_invented"] == 0.5
    assert s["relabel_rate"] == 1 / 4 and s["invented_share"] == 1 / 6


def test_bootstrap_zero_for_identical_and_positive_for_better():
    gb = [GT] * 20
    gl = [GT_LABELS] * 20
    perfect = [det(GT, GT_LABELS)] * 20
    half = [det(GT[:1], GT_LABELS[:1])] * 20
    assert map50(gb, gl, perfect, CLASSES) == 1.0
    d, lo, hi = paired_bootstrap_map50(gb, gl, perfect, perfect, CLASSES, reps=20)
    assert d == lo == hi == 0.0
    d, lo, hi = paired_bootstrap_map50(gb, gl, half, perfect, CLASSES, reps=20)
    assert d > 0 and lo > 0
