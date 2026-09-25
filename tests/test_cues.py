import numpy as np
import pytest

from bolero.cues import CUES, DEFAULT_STRENGTH, apply_cue


def test_defaults_cover_all_cues():
    assert set(CUES) == set(DEFAULT_STRENGTH)


@pytest.mark.parametrize("name", list(CUES))
def test_zero_strength_is_identity(name, image, depth):
    np.testing.assert_allclose(apply_cue(name, image, depth, 0.0), image, atol=1e-6)


@pytest.mark.parametrize("name", list(CUES))
def test_output_valid_and_monotone(name, image, depth):
    weak = apply_cue(name, image, depth, 0.2)
    strong = apply_cue(name, image, depth, 0.8)
    assert weak.shape == image.shape and 0.0 <= strong.min() and strong.max() <= 1.0
    assert np.abs(strong - image).mean() >= np.abs(weak - image).mean()
