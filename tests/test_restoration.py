import json

import numpy as np
import pytest

from bolero.perturbations import PERTURBATIONS
from bolero.restoration import (
    CANDIDATES, PLANS, auto_levels, denoise, denoise_adaptive, fill_mask, inpaint,
    load_plans, restore,
)


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


# ---- dev-selected plans (scripts/tune_plans.py) -----------------------------

def test_candidates_cover_only_real_perturbations():
    assert set(CANDIDATES) <= set(PERTURBATIONS)


def test_every_candidate_set_offers_pass_through():
    # Leaving the image alone must always be selectable: several perturbations
    # destroy information, and no classical plan beats pass-through there.
    for name, variants in CANDIDATES.items():
        assert "pass_through" in variants, name
        assert variants["pass_through"] == []


def test_inpaint_guard_skips_a_hole_too_large_to_invent(image, depth):
    # background_removal level 3 blanks ~70 % of the frame.
    perturbed = PERTURBATIONS["background_removal"](image, 3, depth, np.random.default_rng(1))
    assert fill_mask(perturbed).mean() > 0.4
    np.testing.assert_array_equal(inpaint(perturbed, max_area=0.25), perturbed)


def test_inpaint_guard_still_fills_a_small_hole(image):
    perturbed = PERTURBATIONS["region_removal"](image, 1, rng=np.random.default_rng(1))
    assert fill_mask(perturbed).mean() < 0.25
    assert not np.array_equal(inpaint(perturbed, max_area=0.25), perturbed)


def test_inpaint_default_is_unguarded(image, depth):
    perturbed = PERTURBATIONS["background_removal"](image, 3, depth, np.random.default_rng(1))
    assert not np.array_equal(inpaint(perturbed), perturbed)


def test_stronger_denoise_smooths_more(image):
    perturbed = PERTURBATIONS["gaussian_noise"](image, 3, rng=np.random.default_rng(1))

    def roughness(x):
        return float(np.abs(np.diff(x, axis=0)).mean())

    assert roughness(denoise(perturbed, 0.9)) < roughness(denoise(perturbed, 0.15))


def test_adaptive_denoise_leaves_quiet_images_alone(image):
    np.testing.assert_array_equal(denoise_adaptive(image, threshold=0.10), image)


def test_load_plans_falls_back_and_warns(tmp_path):
    with pytest.warns(UserWarning):
        plans, status = load_plans(tmp_path / "absent.json")
    assert status == "default"
    assert plans == PLANS


def test_load_plans_substitutes_the_selected_candidate(tmp_path):
    path = tmp_path / "plans.json"
    path.write_text(json.dumps({"plans": {"gaussian_noise": "pass_through"}}))
    plans, status = load_plans(path)
    assert status == "tuned"
    assert plans["gaussian_noise"] == []
    assert plans["gaussian_blur"] == PLANS["gaussian_blur"]   # untouched


def test_selected_pass_through_plan_is_identity(image):
    perturbed = PERTURBATIONS["gaussian_noise"](image, 3, rng=np.random.default_rng(1))
    plans = {**PLANS, "gaussian_noise": []}
    np.testing.assert_array_equal(restore(perturbed, "gaussian_noise", plans), perturbed)
