import numpy as np
import pytest

from bolero.baselines import BASELINES, DEFAULT_STRENGTH, apply_baseline


def test_defaults_cover_all_baselines():
    assert set(BASELINES) == set(DEFAULT_STRENGTH)


@pytest.mark.parametrize("name", list(BASELINES))
def test_zero_strength_is_identity(name, image):
    np.testing.assert_allclose(apply_baseline(name, image, 0.0), image, atol=1e-6)


@pytest.mark.parametrize("name", list(BASELINES))
def test_output_valid_and_monotone(name, image):
    weak = apply_baseline(name, image, 0.2)
    strong = apply_baseline(name, image, 0.8)
    assert weak.shape == image.shape and 0.0 <= strong.min() and strong.max() <= 1.0
    assert np.abs(strong - image).mean() >= np.abs(weak - image).mean()
