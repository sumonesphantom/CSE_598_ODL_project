import numpy as np
import pytest

from bolero.perturbations import PERTURBATIONS
from bolero.restoration import PLANS, auto_levels, fill_mask, restore


def test_every_perturbation_has_a_plan():
    assert set(PERTURBATIONS) | {"clean"} == set(PLANS)


def test_clean_plan_is_identity(image):
    np.testing.assert_array_equal(restore(image, "clean"), image)


@pytest.mark.parametrize("name", ["low_contrast", "darken", "brighten", "inversion"])
def test_photometric_plans_are_near_inverses(name, image):
    # Blind auto-levels maps the image to the full range, so it inverts these
    # perturbations for images that already span it (as most photos do).
    image = auto_levels(image)
    perturbed = PERTURBATIONS[name](image, 2)
    before = np.abs(perturbed - image).mean()
    after = np.abs(restore(perturbed, name) - image).mean()
    assert after < 0.5 * before


def test_inpainting_reduces_error(image):
    perturbed = PERTURBATIONS["region_removal"](image, 2, rng=np.random.default_rng(1))
    assert fill_mask(perturbed).any()
    assert np.abs(restore(perturbed, "region_removal") - image).mean() < np.abs(perturbed - image).mean()


@pytest.mark.parametrize("name", ["region_removal", "boundary_erase", "foreground_occlusion", "background_removal"])
def test_inpainting_removes_constant_fill(name, image, depth):
    perturbed = PERTURBATIONS[name](image, 2, depth, np.random.default_rng(1))
    assert fill_mask(perturbed).any()
    assert not fill_mask(restore(perturbed, name)).any()


def test_denoise_reduces_error(image):
    perturbed = PERTURBATIONS["gaussian_noise"](image, 3, rng=np.random.default_rng(1))
    assert np.abs(restore(perturbed, "gaussian_noise") - image).mean() < np.abs(perturbed - image).mean()


def test_fill_mask_ignores_clean_image(image):
    assert not fill_mask(image).any()
