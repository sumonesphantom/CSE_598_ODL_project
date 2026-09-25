import numpy as np
import pytest

from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for


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
