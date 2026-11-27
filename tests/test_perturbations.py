import numpy as np
import pytest
import torch

from bolero.perturbations import ALL_PERTURBATIONS, HELD_OUT, LEVELS, PERTURBATIONS, rng_for


def test_fourteen_perturbations():
    assert len(PERTURBATIONS) == 14


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_shape_range_and_determinism(name, image, depth):
    p = PERTURBATIONS[name]
    for level in LEVELS:
        a = p(image, level, depth, rng_for("img", name, level, 42))
        b = p(image, level, depth, rng_for("img", name, level, 42))
        assert a.shape == image.shape and a.dtype == np.float32
        assert a.min() >= 0.0 and a.max() <= 1.0
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_severity_increases_distortion(name, image, depth):
    p = PERTURBATIONS[name]
    err = [np.mean((p(image, lvl, depth, rng_for("img", name, lvl, 42)) - image) ** 2) for lvl in LEVELS]
    assert err[0] > 0
    assert err[2] >= err[0]


def test_depth_selective_requires_depth(image):
    with pytest.raises(ValueError):
        PERTURBATIONS["foreground_occlusion"](image, 1)


def test_foreground_occlusion_targets_near_region(image, depth):
    out = PERTURBATIONS["foreground_occlusion"](image, 1, depth)
    changed = np.any(out != image, axis=-1)
    assert changed[depth == 1.0].mean() > 0.9
    assert changed[depth < 0.5].mean() < 0.05


def test_held_out_kept_apart():
    assert "rand_augment" in HELD_OUT
    assert not set(HELD_OUT) & set(PERTURBATIONS)
    assert set(ALL_PERTURBATIONS) == set(PERTURBATIONS) | set(HELD_OUT)


@pytest.mark.parametrize("name", list(HELD_OUT))
def test_held_out_shape_range_and_determinism(name, image, depth):
    p = HELD_OUT[name]
    for level in LEVELS:
        a = p(image, level, depth, rng_for("img", name, level, 42))
        b = p(image, level, depth, rng_for("img", name, level, 42))
        assert a.shape == image.shape and a.dtype == np.float32
        assert a.min() >= 0.0 and a.max() <= 1.0
        np.testing.assert_array_equal(a, b)


def test_rand_augment_varies_across_images_and_leaves_torch_rng_alone(image):
    p = HELD_OUT["rand_augment"]
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    outs = [p(image, 3, rng=rng_for(f"img{i}", "rand_augment", 3, 42)) for i in range(8)]
    assert torch.equal(torch.rand(1), expected)
    assert sum(np.any(o != outs[0]) for o in outs[1:]) > 0
    assert sum(np.mean((o - image) ** 2) > 1e-4 for o in outs) >= 4


# ---- Physics-based weather ---------------------------------------------------

def _scene(seed=0, size=64):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.1, 0.9, size=(size, size, 3)).astype(np.float32)
    depth = np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None], (1, size))  # bottom rows near
    return img, depth


def test_weather_is_opt_in_and_valid():
    from bolero.perturbations import ALL_PERTURBATIONS, ANY_PERTURBATION, WEATHER, rng_for
    assert set(WEATHER).isdisjoint(ALL_PERTURBATIONS)
    assert set(WEATHER) <= set(ANY_PERTURBATION)
    img, depth = _scene()
    for name, p in WEATHER.items():
        diffs = []
        for level in (1, 2, 3):
            a = p(img, level, depth, rng_for("im", name, level, 0))
            b = p(img, level, depth, rng_for("im", name, level, 0))
            assert a.shape == img.shape and a.dtype == np.float32
            assert 0.0 <= a.min() and a.max() <= 1.0
            np.testing.assert_array_equal(a, b)  # deterministic per (image, name, level)
            diffs.append(float(np.abs(a - img).mean()))
        assert diffs[0] > 0.005, name
        assert diffs[2] > diffs[0], name   # stronger at higher severity


def test_scattering_hazes_far_more_than_near():
    from bolero.perturbations import WEATHER
    img, depth = _scene()
    for name in ("rain_mist", "dust_haze"):
        out = WEATHER[name](img, 2, depth)
        far_change = np.abs(out[:8] - img[:8]).mean()     # depth ~ 0: far
        near_change = np.abs(out[-8:] - img[-8:]).mean()  # depth ~ 1: near
        assert far_change > near_change, name


def test_dust_fades_blue_first_and_tints_yellow():
    from bolero.perturbations import WEATHER
    img, depth = _scene()
    out = WEATHER["dust_haze"](img, 3, depth)
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[2] and mean[1] > mean[2]     # yellow-brown cast
    contrast = out.reshape(-1, 3).std(0) / img.reshape(-1, 3).std(0)
    assert contrast[2] < contrast[0]                    # blue loses contrast first


def test_weather_maps_onto_weather_taxonomy():
    from bolero.conditions import condition_label
    from bolero.perturbations import WEATHER
    for name, p in WEATHER.items():
        assert condition_label(name, p.family, "weather") == p.family
