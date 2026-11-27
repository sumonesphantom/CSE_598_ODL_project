"""The depth step on the car: the obstacle warning and the SAM structure map."""

import numpy as np
import pytest

from bolero.car_depth import DepthObstacle, object_structure
from bolero.car_perception import CameraModel
from bolero.segmentation import Region


def road_scene(camera, obstacle_at=None, a=2.0, b=0.05):
    """Relative inverse depth of a flat road (sky = far), optionally with a 1.6 m tall, 1.8 m wide
    object standing in the lane at `obstacle_at` metres."""
    h, w = camera.height, camera.width
    z = camera.ground_distance(np.arange(h))
    inv = np.where(np.isfinite(z), 1.0 / np.where(np.isfinite(z), z, 1.0), 0.0)
    d = np.repeat((a * inv + b)[:, None], w, axis=1).astype(np.float32)
    if obstacle_at is not None:
        contact = int(np.argmin(np.abs(np.where(np.isfinite(z), z, 1e9) - obstacle_at)))
        f = camera.f
        top = int(contact - 1.6 * f / obstacle_at)
        half = int(0.9 * f / obstacle_at)
        d[top:contact, w // 2 - half: w // 2 + half] = a / obstacle_at + b
    return d


def test_obstacle_warning_finds_a_standing_object_at_its_distance():
    cam = CameraModel(640, 360, 90, 1.4, -2.0)
    ob = DepthObstacle(cam, scale=1.0, min_area=50)
    assert ob(road_scene(cam))["distance"] is None                   # empty road: no warning
    for dist in (8.0, 15.0, 25.0):
        r = ob(road_scene(cam, dist))
        assert r["distance"] == pytest.approx(dist, rel=0.15)
        assert abs(r["lateral"]) < 1.0
    a, b = ob.fit(road_scene(cam))
    assert a == pytest.approx(2.0, rel=0.05) and b == pytest.approx(0.05, abs=0.01)
    assert ob(road_scene(cam, 60.0))["distance"] is None              # beyond the 40 m range


def test_object_structure_keeps_object_sized_masks():
    shape = (40, 60)
    small = np.zeros(shape, bool)
    small[10:20, 10:20] = True
    big = np.ones(shape, bool)                                         # road / sky sized: dropped
    s = object_structure([Region(small, (10, 10, 20, 20), 0.9), Region(big, (0, 0, 60, 40), 0.9)], shape)
    assert s.shape == shape and s.max() == pytest.approx(1.0)
    assert s[15, 15] > 0.9 and s[35, 50] < 0.01
    assert not object_structure([], shape).any()
