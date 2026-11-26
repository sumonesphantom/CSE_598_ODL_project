import numpy as np
import pandas as pd

from bolero.gallery import confusion_counts, foreground_only, select_cases, transitions
from bolero.perturbations import GRAY


def test_foreground_only_keeps_entity_and_grays_background(image):
    mask = np.zeros(image.shape[:2], bool)
    mask[30:70, 40:90] = True
    out = foreground_only(image, mask)
    assert np.array_equal(out[mask], image[mask])
    assert np.all(out[~mask] == GRAY)
    assert foreground_only(image, None) is image
    assert foreground_only(image, np.zeros_like(mask)) is image


def _table():
    return pd.DataFrame({
        "image_id": list("abcdef"),
        "correct": [False, False, False, True, True, True],
        "confidence": [0.2, 0.9, 0.5, 0.8, 0.7, 0.6],
        "pred_label": ["altar", "monastery", "altar", "church", "church", "church"],
    })


def test_select_cases_orders_failures_by_confidence():
    cases = select_cases(_table(), n_fail=2, n_ok=2, seed=0)
    fails = cases[cases["case"] == "fail"]
    assert list(fails["image_id"]) == ["b", "c"]
    assert (cases["case"] == "ok").sum() == 2
    assert cases[cases["case"] == "ok"]["correct"].all()


def test_confusion_counts_only_counts_failures():
    counts = confusion_counts(_table())
    assert dict(zip(counts["pred_label"], counts["count"])) == {"altar": 2, "monastery": 1}
    assert np.isclose(counts["share_of_failures"].sum(), 1.0)


def test_transitions():
    before = pd.Series([True, True, False, False])
    after = pd.Series([True, False, True, False])
    assert transitions(before, after) == {"fixed": 1, "broken": 1, "still_wrong": 1, "still_right": 1}


def test_text_mask_threshold_and_region_union():
    from bolero.segmentation import Region, box_mask, regions_in_mask, threshold_text_mask

    prob = np.zeros((40, 40), np.float32)
    prob[10:30, 10:30] = 0.9
    guide = threshold_text_mask(prob)
    assert guide.sum() == 400
    assert threshold_text_mask(np.full((4, 4), 0.1, np.float32)) is None
    inside = Region(box_mask((12, 12, 20, 20), (40, 40)), (12, 12, 20, 20), 0.9)
    outside = Region(box_mask((0, 0, 8, 8), (40, 40)), (0, 0, 8, 8), 0.9)
    union = regions_in_mask([inside, outside], guide)
    assert union.sum() == 64 and not union[0:8, 0:8].any()
    assert regions_in_mask([inside], None) is None
