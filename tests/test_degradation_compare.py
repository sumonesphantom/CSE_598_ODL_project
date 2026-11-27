"""Fast bootstrap mAP must equal the reference AP; loaders and helpers for the degradation compare."""

import json

import numpy as np
import pytest

from bolero.degradation_compare import (ap_from, bootstrap_map50, image_matches, kid, map50, match_image,
                                        spearman)
from bolero.detection_eval import Detections, average_precision, evaluate
from bolero.generative import IMAGE_GUIDANCE, PROMPTS, edit_seed, edit_size


def random_case(rng, n_images=12):
    gts, labels, dets = [], [], []
    for _ in range(n_images):
        g = rng.integers(0, 4)
        xy = rng.uniform(0, 80, (g, 2))
        gts.append(np.hstack([xy, xy + rng.uniform(10, 30, (g, 2))]).astype(np.float32))
        labels.append(list(rng.choice(["car", "person"], g)))
        d = rng.integers(0, 6)
        jitter = rng.uniform(-6, 6, (d, 4))
        base = gts[-1][rng.integers(0, g, d)] if g else np.tile([10, 10, 30, 30], (d, 1))
        dets.append(Detections((base + jitter).astype(np.float32), rng.uniform(0, 1, d).astype(np.float32),
                               list(rng.choice(["car", "person"], d))))
    return gts, labels, dets


@pytest.mark.parametrize("seed", range(5))
def test_fast_ap_equals_reference(seed):
    gts, labels, dets = random_case(np.random.default_rng(seed))
    for cls in ("car", "person"):
        ref_gts = [b[np.asarray(l) == cls] if len(l) else np.zeros((0, 4)) for b, l in zip(gts, labels)]
        ref_dets = [(d.boxes[np.asarray(d.labels) == cls], d.scores[np.asarray(d.labels) == cls]) if d.labels
                    else (np.zeros((0, 4)), np.zeros(0)) for d in dets]
        ref, _ = average_precision(ref_gts, ref_dets, 0.5)
        fast = ap_from([match_image(b, l, d, cls) for b, l, d in zip(gts, labels, dets)])
        assert fast == pytest.approx(ref, nan_ok=True)
    per = image_matches(gts, labels, dets, ("car", "person"))
    assert map50(per) == pytest.approx(evaluate(gts, labels, dets, ("car", "person"))["map50"], nan_ok=True)


def test_bootstrap_interval_contains_estimate_and_paired_zero():
    per = image_matches(*random_case(np.random.default_rng(0), 40), ("car", "person"))
    value, low, high = bootstrap_map50(per, n_boot=200)
    assert low <= value <= high
    delta, dlow, dhigh = bootstrap_map50(per, n_boot=50, reference=per)
    assert delta == 0 and dlow == 0 and dhigh == 0


def test_kid_is_small_for_same_distribution_and_larger_for_shifted():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    assert abs(kid(a, b)) < kid(a, b + 1.0)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, np.nan], [1, 2]))


def test_generative_settings():
    assert set(PROMPTS) == {"rain", "dust"}
    assert IMAGE_GUIDANCE[1] > IMAGE_GUIDANCE[2] > IMAGE_GUIDANCE[3]   # level 3 = strongest edit
    w, h = edit_size(640, 427)
    assert min(w, h) in (512, 504, 520) and w % 8 == 0 and h % 8 == 0
    assert edit_seed("a", "rain", 1) == edit_seed("a", "rain", 1) != edit_seed("a", "rain", 2)


def test_load_coco_keeps_vehicle_scenes_without_crowds(tmp_path):
    from bolero.weather_data import load_coco
    (tmp_path / "annotations").mkdir()
    cats = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 62, "name": "chair"}]
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 80} for i in (1, 2, 3, 4)]
    anns = [
        {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 10], "iscrowd": 0},   # car: kept
        {"image_id": 1, "category_id": 62, "bbox": [0, 0, 5, 5], "iscrowd": 0},      # chair: ignored
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},       # person only: dropped
        {"image_id": 3, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 0},
        {"image_id": 3, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 1},     # crowd: dropped
        {"image_id": 4, "category_id": 1, "bbox": [1, 2, 3, 4], "iscrowd": 0},
        {"image_id": 4, "category_id": 3, "bbox": [5, 5, 10, 10], "iscrowd": 0},
    ]
    (tmp_path / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"categories": cats, "images": images, "annotations": anns}))
    samples = load_coco(tmp_path, n=10)
    assert [s.image_id for s in samples] == ["coco_000000000001", "coco_000000000004"]
    assert samples[0].labels == ["car"] and samples[0].condition == "clean"
    np.testing.assert_allclose(samples[0].boxes, [[10, 10, 30, 20]])
    assert samples[1].labels == ["person", "car"]
