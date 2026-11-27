"""CARLA ground truth: boxes from instance segmentation, true depth in the physics convention."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def instance_image(h=60, w=80):
    """BGRA like CARLA's instance camera: R = semantic tag, G/B = instance id (low / high byte)."""
    img = np.zeros((h, w, 4), np.uint8)
    img[10:30, 10:40, 2], img[10:30, 10:40, 1] = 14, 7          # a car, id 7
    img[35:55, 50:60, 2], img[35:55, 50:60, 1] = 13, 9          # a rider, id 9 -> person
    img[35:55, 62:70, 2], img[35:55, 62:70, 0] = 12, 1          # a pedestrian, id 256 (high byte)
    img[2:4, 2:4, 2], img[2:4, 2:4, 1] = 14, 8                  # a 2x2 car: too small
    img[40:58, 0:8, 2] = 9                                       # vegetation: not a class
    return img


def test_boxes_from_instances():
    carla = pytest.importorskip("carla")  # noqa: F841  (record_carla_weather imports it)
    from record_carla_weather import boxes_from_instances
    boxes, labels = boxes_from_instances(instance_image(), min_box=5, min_pixels=20)
    got = sorted(zip(labels, map(tuple, boxes)))
    assert got == [("car", (10, 10, 40, 30)), ("person", (50, 35, 60, 55)), ("person", (62, 35, 70, 55))]


def test_true_depth_matches_the_physics_convention(tmp_path):
    from run_degradation_compare import true_depth
    (tmp_path / "depth").mkdir()
    metres = np.tile(np.linspace(2, 200, 80, dtype=np.float32), (60, 1))
    cv2.imwrite(str(tmp_path / "depth" / "s.png"), (metres * 100).astype(np.uint16))
    d = true_depth(tmp_path, "s")
    assert d.shape == (60, 80) and d.min() >= 0 and d.max() <= 1
    assert d[0, 0] == pytest.approx(1.0) and d[0, -1] == pytest.approx(0.0)    # near = 1, far = 0
    assert np.all(np.diff(d[0]) <= 1e-6)                                        # monotone with distance
