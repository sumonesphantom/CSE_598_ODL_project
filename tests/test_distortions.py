import numpy as np
import pytest

from bolero.distortions import DISTORTIONS
from bolero.perturbations import LEVELS, rng_for


@pytest.mark.parametrize("name", list(DISTORTIONS))
def test_shape_range_and_determinism(name, image):
    d = DISTORTIONS[name]
    for level in LEVELS:
        a = d(image, level, rng=rng_for("img", name, level, 42))
        b = d(image, level, rng=rng_for("img", name, level, 42))
        assert a.shape == image.shape and a.dtype == np.float32
        assert a.min() >= 0.0 and a.max() <= 1.0
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", list(DISTORTIONS))
def test_severity_increases_distortion(name, image):
    d = DISTORTIONS[name]
    err = [np.mean((d(image, lvl, rng=rng_for("img", name, lvl, 42)) - image) ** 2) for lvl in LEVELS]
    assert err[0] > 0
    assert err[2] >= err[0]


def test_radial_warps_leave_corners_alone(image):
    # Outside the inscribed circle swirl, bulge and pinch leave pixels untouched.
    for name in ["swirl", "bulge", "pinch"]:
        out = DISTORTIONS[name](image, 3)
        for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            np.testing.assert_allclose(out[i, j], image[i, j], atol=1e-5)
