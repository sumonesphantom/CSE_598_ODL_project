"""Tests for the decision-engine evidence layer (Phase 1).

These cover serialization of scene evidence, image preparation for the JEV
image bounds, region overlays, question schemas, and the condition sampler.
Decider adapters are tested separately once they land.
"""

from __future__ import annotations

import base64
import io
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from bolero import prompting
from bolero.conditions import TAXONOMIES, iter_samples
from bolero.config import (
    JEV_IMAGE_MAX_PIXELS,
    JEV_IMAGE_MAX_SIDE,
    JEV_IMAGE_MIN_PIXELS,
    JEV_IMAGE_MIN_SIDE,
)
from bolero.data import ImageRecord
from bolero.prompting import (
    SceneEvidence,
    build_jev_payload,
    build_jev_questions,
    draw_regions,
    image_statistics,
    prepare_image,
    region_evidence,
)

EXPECTED_STAT_KEYS = {
    "width",
    "height",
    "luminance_mean",
    "luminance_std",
    "luminance_p01",
    "luminance_p50",
    "luminance_p99",
    "channel_mean_r",
    "channel_mean_g",
    "channel_mean_b",
    "channel_std_r",
    "channel_std_g",
    "channel_std_b",
    "saturation_mean",
    "saturation_low_fraction",
    "laplacian_variance",
    "gradient_mean",
    "gradient_p95",
    "noise_sigma",
    "black_fill_fraction",
    "gray_fill_fraction",
    "blockiness_x",
    "blockiness_y",
    "top_bottom_luminance_diff",
}


def test_image_statistics_keys_and_json(image):
    stats = image_statistics(image)
    assert set(stats) == EXPECTED_STAT_KEYS
    assert json.dumps(stats)
    for key, value in stats.items():
        assert isinstance(value, (int, float)), key
        assert np.isfinite(value), key
    assert stats["width"] == image.shape[1] and stats["height"] == image.shape[0]


def test_to_state_drops_detector_by_default_and_on_request(depth):
    mask = depth > 0.9
    evidence = SceneEvidence(
        (128, 96),
        [("golf ball", 0.707), ("tennis ball", 0.1)],
        perturbation="gaussian_blur",
        perturbation_prob=0.82,
        entity=region_evidence(mask),
        stats={"width": 128, "height": 96},
    )
    without = evidence.to_state(include_detector=False)
    assert "detector" not in without
    assert "detector" not in json.dumps(without)

    with_detector = evidence.to_state(include_detector=True)
    assert with_detector["detector"] == {"label": "gaussian_blur", "probability": 0.82}


def test_to_state_never_leaks_ground_truth(depth):
    mask = depth > 0.9
    evidence = SceneEvidence(
        (128, 96),
        [("golf ball", 0.707)],
        perturbation="gaussian_blur",
        perturbation_prob=0.9,
        entity=region_evidence(mask, label="golf ball"),
        damaged=region_evidence(mask),
    )
    text = json.dumps(evidence.to_state(include_detector=True))
    for forbidden in ("wnid", "class_name", "condition", "severity"):
        assert forbidden not in text


def test_prepare_image_bounds(image):
    rng = np.random.default_rng(0)
    wide = rng.random((100, 1000, 3), dtype=np.float32)
    for src in (image, wide):
        url = prepare_image(src)
        assert url.startswith("data:image/jpeg;base64,")
        raw = base64.b64decode(url.split(",", 1)[1])
        assert raw[:2] == b"\xff\xd8"
        decoded = Image.open(io.BytesIO(raw))
        assert decoded.format == "JPEG"
        width, height = decoded.size
        assert JEV_IMAGE_MIN_SIDE <= width <= JEV_IMAGE_MAX_SIDE
        assert JEV_IMAGE_MIN_SIDE <= height <= JEV_IMAGE_MAX_SIDE
        assert JEV_IMAGE_MIN_PIXELS <= width * height <= JEV_IMAGE_MAX_PIXELS


def test_draw_regions_changes_image(image):
    shape = image.shape[:2]
    entity = prompting.RegionEvidence((10, 10, 60, 60), 0.2, None, "entity")
    damaged = prompting.RegionEvidence((50, 40, 100, 80), 0.1, None, "damaged")
    out = draw_regions(image, entity=entity, damaged=damaged)
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, image)
    assert shape == out.shape[:2]


def test_jev_questions_shape():
    questions = build_jev_questions("family")
    assert set(questions) == {"degraded", "condition", "severity"}
    assert questions["degraded"]["type"] == "noul"
    assert questions["condition"]["type"] == "choice"
    assert questions["severity"]["type"] == "score"

    assert set(questions["condition"]["criteria"]) == set(TAXONOMIES["family"])
    severity = questions["severity"]["criteria"]
    assert len(severity) == 3
    assert severity == ["mild", "moderate", "severe"]


def test_jev_payload_with_and_without_image(image, depth):
    mask = depth > 0.9
    evidence = SceneEvidence(
        (128, 96),
        [("golf ball", 0.7)],
        perturbation="gaussian_blur",
        perturbation_prob=0.8,
        entity=region_evidence(mask),
        stats={"width": 128, "height": 96},
    )
    plain = build_jev_payload(evidence, taxonomy="family")
    assert isinstance(plain["state"], dict)
    assert plain["state"]["evidence"]["image"] == {"width": 128, "height": 96}
    assert set(plain["questions"]) == {"degraded", "condition", "severity"}
    assert "model" not in plain

    with_image = build_jev_payload(evidence, taxonomy="family", image=image)
    assert isinstance(with_image["state"], list)
    assert len(with_image["state"]) == 2
    assert with_image["state"][1]["type"] == "image_url"
    assert with_image["state"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    ("taxonomy", "expected_condition"),
    [("family", "blur"), ("weather", "other")],
)
def test_conditions_iter_samples(tmp_path, taxonomy, expected_condition):
    rng = np.random.default_rng(0)
    array = np.round(rng.random((64, 64, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sample.jpg"
    Image.fromarray(array).save(path, format="JPEG", quality=95)
    record = ImageRecord(image_id="sample", path=path, wnid="n01440764")

    samples = list(
        iter_samples(
            [record],
            taxonomy=taxonomy,
            perturbations=["gaussian_blur"],
            levels=[1],
            max_per_bucket=1,
        )
    )
    assert len(samples) == 2

    clean, degraded = samples
    assert clean.clean and clean.perturbation is None
    assert clean.condition == "clean" and clean.severity is None

    assert not degraded.clean
    assert degraded.perturbation == "gaussian_blur"
    assert degraded.condition == expected_condition
    assert degraded.severity == 1

    buckets = Counter((s.condition, s.severity) for s in samples)
    assert all(count <= 1 for count in buckets.values())
