"""The driving study's car-side pieces: ground truth, weather on car frames, repair, routing, isolation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bolero.car_perception import CameraModel, ConditionRouter, isolate, smooth_sequence  # noqa: E402
from bolero.detection_eval import Detections  # noqa: E402
from bolero.ego import apply_weather, focal_px, inverse_depth, objects_from_instances  # noqa: E402
from bolero.learned_restoration import dcp_dehaze, guided_filter, weight_path  # noqa: E402


def test_objects_from_instances_distance_lateral_and_ego():
    h, w, fov = 60, 80, 90.0
    inst = np.zeros((h, w, 4), np.uint8)
    inst[20:40, 36:44, 2], inst[20:40, 36:44, 1] = 12, 5       # pedestrian straight ahead, id 5
    inst[20:40, 70:80, 2], inst[20:40, 70:80, 1] = 14, 6       # car to the right, id 6
    inst[50:60, :, 2], inst[50:60, :, 1] = 14, 3               # the car's own body, id 3
    depth = np.full((h, w), 10.0, np.float32)
    objs = {o["id"]: o for o in objects_from_instances(inst, depth, fov, min_box=5, min_pixels=20, exclude_ids=[3])}
    assert set(objs) == {5, 6}
    assert objs[5]["label"] == "person" and objs[5]["distance"] == pytest.approx(10.0)
    assert abs(objs[5]["lateral"]) < 0.2 and objs[5]["in_path"]
    f = focal_px(w, fov)
    assert objs[6]["lateral"] == pytest.approx((74.5 + 0.5 - 40) * 10 / f, rel=0.05)
    assert not objs[6]["in_path"]


def test_inverse_depth_and_weather():
    depth = np.tile(np.linspace(2.0, 200.0, 64, dtype=np.float32), (48, 1))
    inv = inverse_depth(depth)
    assert inv[0, 0] > inv[0, -1] and inv.max() <= 1.0 and inv.min() >= 0.0
    img = np.random.default_rng(0).random((48, 64, 3)).astype(np.float32)
    assert apply_weather(img, depth, "clean", 0, "f") is img
    a = apply_weather(img, depth[::2, ::2], "rain_mist", 2, "f")   # half-resolution depth is upsampled
    b = apply_weather(img, depth[::2, ::2], "rain_mist", 2, "f")
    assert a.shape == img.shape and np.array_equal(a, b)
    assert not np.array_equal(a, apply_weather(img, depth[::2, ::2], "rain_mist", 2, "g"))
    # Haze grows with distance: far columns move further from the clean image than near ones.
    hazed = apply_weather(img, depth, "dust_haze", 3, "f")
    change = np.abs(hazed - img).mean(axis=(0, 2))
    assert change[-8:].mean() > change[:8].mean()


def test_ground_plane_distance_round_trip():
    cam = CameraModel(1280, 720, 90.0, 1.4, -2.0)
    theta = np.deg2rad(2.0)
    for horizontal in (5.0, 12.0, 30.0):
        # A road point `horizontal` m ahead, 1.4 m below the camera, in camera coordinates.
        z = horizontal * np.cos(theta) + 1.4 * np.sin(theta)
        y = 1.4 * np.cos(theta) - horizontal * np.sin(theta)
        v = cam.height / 2 + cam.f * y / z
        assert cam.ground_distance(np.array([v]))[0] == pytest.approx(z, rel=1e-6)
    assert np.isinf(cam.ground_distance(np.array([300.0]))[0])   # above the horizon


def test_isolate_marks_nearest_in_path_object_critical():
    cam = CameraModel(1280, 720, 90.0, 1.4, -2.0)
    boxes = np.array([[600, 300, 680, 470],     # ahead, near (bottom low in the image)
                      [620, 330, 660, 420],     # ahead, further
                      [0, 300, 80, 500],        # far to the left
                      [600, 300, 680, 470]], np.float32)
    dets = Detections(boxes, np.array([0.9, 0.8, 0.9, 0.2], np.float32), ["person", "car", "car", "car"])
    objs = isolate(dets, cam, score=0.5)
    assert len(objs) == 3                                      # the 0.2-score box is dropped
    assert [o["distance"] for o in objs] == sorted(o["distance"] for o in objs)
    crit = [o for o in objs if o["critical"]]
    assert len(crit) == 1 and crit[0]["label"] == "person" and crit[0]["in_path"]
    assert not next(o for o in objs if o["box"][0] == 0)["in_path"]


def test_router_smoothing_removes_flicker():
    rng = np.random.default_rng(0)
    truth = ["clean"] * 20 + ["rain"] * 20
    p_rain = np.r_[rng.uniform(0.3, 0.65, 20), rng.uniform(0.55, 0.9, 20)]
    probs = np.stack([1 - p_rain, p_rain, np.zeros(40)], 1)
    raw = smooth_sequence(probs, 1.0, 0.6, 0.6)
    smooth = smooth_sequence(probs, 0.3, 0.6, 0.45)
    switches = lambda r: sum(a != b for a, b in zip(r, r[1:]))
    assert switches(smooth) < switches(raw)
    assert smooth[-1] == "rain" and smooth[0] == "clean"
    # The raw setting is the detector's own per-frame rule.
    assert raw == ["rain" if p >= 0.6 else "clean" for p in p_rain]
    assert sum(a == b for a, b in zip(smooth, truth)) >= sum(a == b for a, b in zip(raw, truth))


def test_router_switches_between_weathers():
    r = ConditionRouter(detector=None, alpha=1.0, enter=0.6, exit=0.4)
    assert r.step_proba(np.array([0.1, 0.8, 0.1])) == "rain"
    assert r.step_proba(np.array([0.1, 0.5, 0.4])) == "rain"     # above exit: stays
    assert r.step_proba(np.array([0.1, 0.2, 0.7])) == "dust"
    assert r.step_proba(np.array([0.8, 0.1, 0.1])) == "clean"


def test_dcp_dehaze_moves_toward_the_clean_image():
    rng = np.random.default_rng(1)
    clean = np.clip(rng.random((96, 128, 3)) * 0.6 + 0.1, 0, 1).astype(np.float32)
    clean[:10] = 0.02                                          # some dark pixels, as the prior expects
    t = np.linspace(0.9, 0.4, 128, dtype=np.float32)[None, :, None]
    airlight = np.array([0.76, 0.60, 0.38], np.float32)        # yellow-brown, as in the dust effect
    hazed = clean * t + airlight * (1 - t)
    out = dcp_dehaze(hazed, patch=7)
    assert out.shape == hazed.shape and out.dtype == np.float32
    assert np.abs(out - clean).mean() < np.abs(hazed - clean).mean()


def test_guided_filter_keeps_constants():
    g = np.random.default_rng(2).random((40, 50)).astype(np.float32)
    out = guided_filter(g, np.full_like(g, 0.3), r=4)
    assert np.allclose(out, 0.3, atol=1e-4)


def test_weight_hash_is_checked(tmp_path):
    (tmp_path / "restormer_deraining.pth").write_bytes(b"not the real weights")
    with pytest.raises(RuntimeError, match="pinned hash"):
        weight_path("restormer_deraining.pth", root=tmp_path)


def test_routing_schedule_and_measures():
    from run_car_routing import measures, schedule

    s = schedule(0, 120)
    assert len(s) == 120 and s[0] == ("clean", 0) and s[12][1] == 1 and s[-1] == ("clean", 0)
    assert {e for e, _ in schedule(0, 120)} != {e for e, _ in schedule(1, 120)}
    fam = ["clean" if e == "clean" else ("rain" if e.startswith("rain") else "dust") for e, _ in s]
    perfect = measures(fam, fam)
    assert perfect["correct"] == 120 and perfect["clean_false"] == 0 and perfect["switches"] == 4
    assert perfect["delays"] == [0.0, 0.0]
    late = fam[:]
    late[12:14] = ["clean", "clean"]                           # first weather segment caught 1 s late
    assert measures(late, fam)["delays"][0] == 1.0


def test_fast_dcp_matches_full_resolution():
    rng = np.random.default_rng(3)
    import cv2

    clean = cv2.GaussianBlur(rng.random((180, 320, 3)).astype(np.float32), (0, 0), 3)
    t = np.linspace(0.9, 0.4, 320, dtype=np.float32)[None, :, None]
    hazed = clean * t + np.array([0.76, 0.60, 0.38], np.float32) * (1 - t)
    full, fast = dcp_dehaze(hazed), dcp_dehaze(hazed, work_side=160)
    assert fast.shape == full.shape
    assert np.abs(fast - full).mean() < 0.03


def test_async_routing_uses_latest_finished_route():
    from bolero.car_perception import CarPerception
    from bolero.car_perception import CarDetections

    import threading

    release = threading.Event()

    class FakeRouter:
        state = "clean"

        def __call__(self, image):
            release.wait(5)                     # a slow router: finishes only when released
            self.state = "dust" if image.mean() > 0.5 else "clean"
            return self.state

    seen = []
    repairers = {"dust": lambda x: seen.append("dust") or x}
    detector = lambda x, want_masks=False: CarDetections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [])
    pipe = CarPerception(FakeRouter(), repairers, detector, async_route=True)
    bright = np.full((8, 8, 3), 0.9, np.float32)
    first = pipe(bright)["route"]
    assert first == "clean"                     # the first frame does not wait for the router
    release.set()
    pipe.pending.result()
    assert pipe(bright)["route"] == "dust" and seen == ["dust"]
    sync = CarPerception(FakeRouter(), repairers, detector)
    assert sync(bright)["route"] == "dust"
