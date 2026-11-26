"""The driving study's Phase 3-4 pieces: braking policy, steering, run measures, weather v2, repair net."""

import numpy as np
import pytest
import torch

from bolero.driving import BrakePolicy, gap_and_offset, pure_pursuit, run_measures
from bolero.ego import apply_weather
from bolero.task_repair import RepairNet


def obj(distance, in_path=True):
    return {"distance": distance, "in_path": in_path}


def test_policy_needs_two_frames_and_remembers_briefly():
    p = BrakePolicy()
    assert p.observe([obj(20.0)], 0.1, 10.0) is None            # one frame: not yet
    assert p.observe([obj(19.0)], 0.1, 10.0) == 19.0            # confirmed
    d = p.observe([], 0.1, 10.0)                                 # dropped out: remembered, 1 m closer
    assert d == pytest.approx(18.0)
    for _ in range(5):
        d = p.observe([], 0.1, 10.0)
    assert d is None                                             # forgotten after `memory`
    assert p.observe([obj(5.0, in_path=False)], 0.1, 10.0) is None   # not in the path: ignored


def test_policy_brakes_harder_when_closer():
    p = BrakePolicy()
    _, far = p.control(10.0, 60.0)
    _, mid = p.control(10.0, 20.0)
    _, near = p.control(10.0, 5.0)
    assert far == 0.0 and 0.0 < mid < near and near == 1.0
    assert p.control(10.0, None) == p.control(10.0, 500.0)       # nothing ahead: cruise
    thr, brk = p.control(5.0, None)
    assert thr > 0 and brk == 0.0                                # below target speed: accelerate


def test_pure_pursuit_and_gap_geometry():
    assert pure_pursuit([0, 0], 0.0, [10, 0]) == pytest.approx(0.0)
    assert pure_pursuit([0, 0], 0.0, [10, 2]) > 0                # target to the right (+y): steer right
    assert pure_pursuit([0, 0], 90.0, [-2, 10]) > 0              # heading +y, target to its right (-x)
    gap, lat = gap_and_offset([0, 0], 90.0, 2.0, [0.5, 20.0], 0.3)
    assert gap == pytest.approx(17.7) and lat == pytest.approx(-0.5)


def test_run_measures_collision_and_ttc():
    trace = [{"t": 0.05 * k, "speed": 10.0, "brake": 0.0 if k < 10 else 0.5, "gap": 20.0 - 0.5 * k, "lateral": 0.1,
              "hazard_speed": 0.0, "collision": k >= 40} for k in range(45)]
    m = run_measures(trace, 1.3)
    assert m["collision"] and m["impact_speed"] == 10.0
    assert m["min_ttc"] == pytest.approx(0.05, abs=0.06) and m["brake_onset_gap"] == pytest.approx(15.0)
    safe = run_measures([{**r, "collision": False, "lateral": 4.0} for r in trace], 1.3)
    assert not safe["collision"] and np.isnan(safe["min_gap"]) and safe["min_ttc"] == float("inf")


def test_weather_v2_dust_is_uneven_unlike_v1():
    rng = np.random.default_rng(0)
    import cv2

    flat = np.full((90, 160, 3), 0.3, np.float32)               # a flat grey wall 30 m away
    wall = np.full((90, 160), 30.0, np.float32)
    v1 = apply_weather(flat, wall, "dust_haze", 3, "f", model="v1")
    v2 = apply_weather(flat, wall, "dust_haze", 3, "f", model="v2")
    assert v1.std(axis=(0, 1)).max() < 1e-4                      # v1: one transmission, one airlight
    smooth = cv2.GaussianBlur(v2, (0, 0), 4)                     # v2: plumes and a sky-dependent airlight
    assert smooth.std(axis=(0, 1)).max() > 0.02
    img = cv2.GaussianBlur(rng.random((90, 160, 3)).astype(np.float32), (0, 0), 2)
    depth = np.tile(np.linspace(80.0, 5.0, 90, dtype=np.float32)[:, None], (1, 160))
    v2 = apply_weather(img, depth, "dust_haze", 3, "f", model="v2")
    assert v2.shape == img.shape and v2.dtype == np.float32
    assert np.array_equal(v2, apply_weather(img, depth, "dust_haze", 3, "f", model="v2"))
    near, far = np.abs(v2 - img)[-10:].mean(), np.abs(v2 - img)[:10].mean()
    assert far > near                                            # metric depth: far rows fade more
    clean = apply_weather(img, depth, "clean", 0, "f", model="v2")   # the camera still adds noise + JPEG
    assert 0 < np.abs(clean - img).mean() < 0.03
    a = apply_weather(img, depth, "dust_haze", 2, "f", model="v2", t=0.0)
    b = apply_weather(img, depth, "dust_haze", 2, "f", model="v2", t=0.1)
    assert np.abs(a - b).mean() < 0.05                           # plumes drift smoothly in a live drive
    with pytest.raises(ValueError):
        apply_weather(img, depth, "dust_haze", 2, "f", model="v3")


def test_repair_net_starts_as_identity():
    net = RepairNet(8).eval()
    x = torch.rand(1, 3, 37, 61)
    with torch.no_grad():
        assert torch.allclose(net(x), x, atol=1e-6)


def test_route_target_and_run_ids():
    pytest.importorskip("carla")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_car_scenarios import parse_run_id, route_target

    route = np.array([[2.0 * k, 0.0] for k in range(20)])            # straight along +x, a point every 2 m
    target, dev = route_target(route, [3.0, 0.4])
    assert dev == pytest.approx(0.4) and target[0] == pytest.approx(10.0)   # 8 m ahead of the nearest point
    assert route_target(route, [100.0, 0.0])[0][0] == pytest.approx(38.0)   # past the end: the last point
    assert parse_run_id("Town10HD_Opt_L1_pedestrian_cross_dust_storm2_routed") == (
        "Town10HD_Opt", 1, "pedestrian_cross", "dust_storm", 2, "routed")
    assert parse_run_id("Town02_Opt_L0_lead_brake_clean0_gt") == ("Town02_Opt", 0, "lead_brake", "clean", 0, "gt")
    with pytest.raises(ValueError):
        parse_run_id("Town02_Opt_L0_unknown_clean0_gt")
