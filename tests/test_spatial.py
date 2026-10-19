import numpy as np
import pandas as pd
import pytest

from bolero.benchmark import corruption_errors, error_table, mean_corruption_error
from bolero.classifier import resolve_backbone
from bolero.config import BACKBONES
from bolero.cues import CUES, apply_cue
from bolero.prompting import SceneEvidence, build_prompt, depth_layout, region_evidence
from bolero.segmentation import Region, box_mask, box_of, select_entity, structure_map


def test_resolve_backbone():
    assert resolve_backbone("resnet50") == BACKBONES["resnet50"]
    assert resolve_backbone("org/custom-model") == "org/custom-model"


def test_box_roundtrip():
    mask = box_mask((10, 5, 30, 25), (40, 50))
    assert box_of(mask) == (10, 5, 30, 25) and mask.sum() == 20 * 20


def test_select_entity_prefers_central_and_skips_whole_frame():
    shape = (100, 100)
    whole = Region(np.ones(shape, bool), (0, 0, 100, 100), 0.99)
    corner = Region(box_mask((0, 0, 20, 20), shape), (0, 0, 20, 20), 0.99)
    center = Region(box_mask((30, 30, 70, 70), shape), (30, 30, 70, 70), 0.9)
    assert select_entity([whole, corner, center], shape) is center
    assert select_entity([whole], shape) is None


def test_structure_map_contract():
    shape = (96, 128)
    s = structure_map(box_mask((40, 30, 90, 70), shape), shape)
    assert s.dtype == np.float32 and s.shape == shape and 0.0 <= s.min() and s.max() <= 1.0
    assert s[50, 65] > 0.9 and s[5, 5] < 0.05
    assert not structure_map(None, shape).any()


@pytest.mark.parametrize("name", list(CUES))
def test_cues_accept_structure_maps(name, image):
    s = structure_map(box_mask((40, 30, 90, 70), image.shape[:2]), image.shape[:2])
    np.testing.assert_allclose(apply_cue(name, image, s, 0.0), image, atol=1e-6)
    out = apply_cue(name, image, s, 0.5)
    assert out.shape == image.shape and 0.0 <= out.min() and out.max() <= 1.0


def test_ce_against_itself_is_one():
    summary = pd.DataFrame({"perturbation": ["blur"] * 3 + ["noise"] * 3, "level": [1, 2, 3] * 2,
                            "acc_pert": [0.7, 0.5, 0.3, 0.6, 0.5, 0.4]})
    errs = error_table(summary)
    ce = corruption_errors(errs, errs, 0.8, 0.8)
    np.testing.assert_allclose(ce["ce"], 1.0)
    np.testing.assert_allclose(ce["relative_ce"], 1.0)
    assert mean_corruption_error(ce, exclude=("noise",))["perturbations"] == 1


def test_ce_rewards_lower_error():
    ref = pd.DataFrame({"perturbation": ["blur"] * 2, "level": [1, 2], "error": [0.4, 0.6]})
    better = ref.assign(error=[0.2, 0.3])
    assert corruption_errors(better, ref, 0.8, 0.8)["ce"].iloc[0] == pytest.approx(0.5)


def test_prompt_contains_evidence(depth):
    mask = depth > 0.9
    ev = SceneEvidence((128, 96), [("golf ball", 0.707), ("tennis ball", 0.1)], "gaussian_blur", 0.8,
                       entity=region_evidence(mask, depth), depth_layout=depth_layout(depth))
    prompt = build_prompt(ev)
    assert "golf ball" in prompt and "gaussian_blur" in prompt and '"box"' in prompt
    assert "damaged" not in prompt.split("EVIDENCE:")[1]  # empty fields are dropped
