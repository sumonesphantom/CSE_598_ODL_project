import numpy as np
import pytest


@pytest.fixture
def image() -> np.ndarray:
    """A synthetic 'scene': smooth colored background with a bright textured square entity."""
    rng = np.random.default_rng(0)
    h, w = 96, 128
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack([0.2 + 0.5 * xx, 0.3 + 0.4 * yy, 0.5 - 0.3 * xx], axis=-1)
    img[30:70, 40:90] = 0.75 + 0.1 * rng.standard_normal((40, 50, 3))
    return np.clip(img, 0.05, 0.95).astype(np.float32)


@pytest.fixture
def depth() -> np.ndarray:
    """Near (1.0) inside the entity square, receding toward the top of the frame elsewhere."""
    h, w = 96, 128
    d = np.tile(np.linspace(0.0, 0.5, h)[:, None], (1, w))
    d[30:70, 40:90] = 1.0
    return d.astype(np.float32)
