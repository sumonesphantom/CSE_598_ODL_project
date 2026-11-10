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


def test_to_state_omits_empty_candidates():
    state = SceneEvidence((10, 10), []).to_state()
    assert "classifier_candidates" not in state
    assert state == {"image": {"width": 10, "height": 10}}


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


def test_conditions_iter_samples_empty_list(tmp_path):
    rng = np.random.default_rng(0)
    array = np.round(rng.random((64, 64, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sample.jpg"
    Image.fromarray(array).save(path, format="JPEG", quality=95)
    record = ImageRecord(image_id="sample", path=path, wnid="n01440764")

    samples = list(iter_samples([record], taxonomy="family", perturbations=[], levels=[1]))
    assert [s.condition for s in samples] == ["clean"]
    assert samples[0].clean


# ---------------------------------------------------------------------------
# Decider roster (Checkpoint 3)
# ---------------------------------------------------------------------------

from bolero.conditions import ConditionSample
from bolero.deciders import (
    ConstantDecider,
    JevDecider,
    RFDecider,
    SeverityClassifier,
    build_hf_prompt,
    parse_json_object,
    per_condition_metrics,
    routing_metrics,
    score_rows,
)
from bolero.detection import Decision
from bolero.openrouter import DecisionResponse, OpenRouterError


def make_sample(image, condition="blur", severity=2, perturbation="gaussian_blur"):
    return ConditionSample(
        image_id="img-1",
        wnid="n01440764",
        class_name="tench",
        condition=condition,
        severity=severity,
        image=image,
        clean_image=image,
        perturbation=perturbation,
        level=severity,
    )


def make_evidence():
    return SceneEvidence(
        (128, 96),
        [("tench", 0.7)],
        perturbation="gaussian_blur",
        perturbation_prob=0.8,
        stats={"width": 128, "height": 96},
    )


class FakeDetector:
    classes = np.array(["clean", "gaussian_blur", "gaussian_noise"])

    def __init__(self):
        self.probs = np.array([[0.05, 0.85, 0.10]])

    def predict_proba(self, features):
        return self.probs

    def decide(self, image):
        return Decision("gaussian_blur", "gaussian_blur", 0.7)


class FakeSeverityModel:
    def predict(self, features):
        return 2


def fake_response(answers, **kwargs):
    return DecisionResponse(
        answers=answers,
        model=kwargs.get("model", "typesafe/jev-1.14"),
        provider="test",
        cost=kwargs.get("cost", 0.001),
        original_cost=kwargs.get("cost", 0.001),
        input_tokens=10,
        output_tokens=5,
        cached=kwargs.get("cached", True),
        latency_ms=kwargs.get("latency_ms", 12.3),
        raw={"answers": answers},
    )


class FakeClient:
    def __init__(self, answers=None, error=None):
        self.answers = answers if answers is not None else {
            "degraded": {"noul": 0.9},
            "condition": {
                "choice": "blur",
                "probabilities": {"blur": 0.7, "noise": 0.3},
                "confidence": 0.8,
            },
            "severity": {"score": 1.2},
        }
        self.error = error
        self.calls = []

    def ask(self, model, state, questions):
        self.calls.append({"model": model, "state": state, "questions": questions})
        if self.error is not None:
            raise self.error
        return fake_response(self.answers)


def test_constant_deciders():
    clean = ConstantDecider("always_clean", degraded=False, condition="clean")
    assert clean.name == "always_clean"
    assert clean.available() == (True, "")
    out = clean.decide(make_sample(np.zeros((4, 4, 3), dtype=np.float32)), make_evidence())
    assert out.degraded is False
    assert out.condition == "clean"
    assert out.severity is None
    assert out.parse_ok
    assert out.probabilities == {"clean": 1.0}

    degraded = ConstantDecider("always_degraded", degraded=True, condition="blur", severity=2)
    out = degraded.decide(make_sample(np.zeros((4, 4, 3), dtype=np.float32)), make_evidence())
    assert out.degraded is True
    assert out.condition == "blur"
    assert out.severity == 2
    assert out.p_degraded == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"degraded": true, "condition": "blur"}', {"degraded": True, "condition": "blur"}),
        ('```json\n{"severity": 2}\n```', {"severity": 2}),
        ('Here is the answer:\n{"confidence": 0.4}\nDone.', {"confidence": 0.4}),
    ],
)
def test_parse_json_object_valid(text, expected):
    assert parse_json_object(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "not json", '{"degraded": ', "```json\n{bad}\n```", "[1, 2, 3]", None],
)
def test_parse_json_object_malformed(text):
    assert parse_json_object(text) is None


def test_severity_classifier_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    centers = {1: (0.0, 0.0), 2: (3.0, 0.0), 3: (0.0, 3.0)}
    features, levels = [], []
    for level, center in centers.items():
        features.extend(center + rng.normal(0.0, 0.2, size=(10, 2)))
        levels.extend([level] * 10)
    features = np.asarray(features, dtype=np.float32)
    levels = np.asarray(levels)

    model = SeverityClassifier().fit(features, levels)
    predictions = [model.predict(row) for row in features]
    assert set(predictions) <= {1, 2, 3}
    assert sum(p == level for p, level in zip(predictions, levels)) >= 27

    proba = model.predict_proba(np.array([[3.0, 0.0]], dtype=np.float32))
    assert set(proba) <= {1, 2, 3}
    assert sum(proba.values()) == pytest.approx(1.0)

    path = tmp_path / "severity.joblib"
    model.save(path)
    assert path.exists()
    loaded = SeverityClassifier.load(path)
    assert [loaded.predict(row) for row in features] == predictions


def test_rf_decider_maps_family(image):
    decider = RFDecider(
        detector=FakeDetector(),
        severity_model=FakeSeverityModel(),
    )
    assert decider.available() == (True, "")
    out = decider.decide(make_sample(image), make_evidence())
    assert out.degraded is True
    assert out.condition == "blur"
    assert out.severity == 2
    assert out.confidence == pytest.approx(0.7)
    assert out.p_degraded == pytest.approx(0.95)
    assert out.parse_ok
    assert out.probabilities["clean"] == pytest.approx(0.05)
    assert out.probabilities["blur"] == pytest.approx(0.85)
    assert out.probabilities["noise"] == pytest.approx(0.10)

    unavailable = RFDecider()
    assert unavailable.available()[0] is False


def test_rf_decider_clean_branch(image):
    class CleanDetector(FakeDetector):
        def decide(self, image):
            return Decision("clean", "clean", 0.9)

        def predict_proba(self, features):
            return np.array([[0.9, 0.05, 0.05]])

    out = RFDecider(detector=CleanDetector()).decide(make_sample(image, "clean", None, None), make_evidence())
    assert out.degraded is False
    assert out.condition == "clean"
    assert out.severity is None
    assert out.p_degraded == pytest.approx(0.1)
    assert out.confidence == pytest.approx(0.9)


def test_jev_decider_maps_answers(image):
    client = FakeClient()
    decider = JevDecider("jev", client, "typesafe/jev-1.14")
    assert decider.available() == (True, "")
    out = decider.decide(make_sample(image), make_evidence())
    assert out.degraded is True
    assert out.p_degraded == pytest.approx(0.9)
    assert out.condition == "blur"
    assert out.severity == 2
    assert out.confidence == pytest.approx(0.8)
    assert out.probabilities == {"blur": 0.7, "noise": 0.3}
    assert out.parse_ok
    assert out.cached is True
    assert out.latency_ms == pytest.approx(12.3)
    assert out.cost == pytest.approx(0.001)
    assert client.calls[0]["model"] == "typesafe/jev-1.14"


def test_jev_decider_image_mode_state(image):
    client = FakeClient()
    decider = JevDecider("jev", client, "typesafe/jev-1.14", evidence_mode="image")
    decider.decide(make_sample(image), make_evidence())
    state = client.calls[0]["state"]
    assert isinstance(state, list)
    assert len(state) == 2
    assert "evidence" not in state[0]
    assert state[1]["type"] == "image_url"
    assert state[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_jev_decider_bad_answers(image):
    client = FakeClient(answers={"degraded": {"noul": 0.2}})
    out = JevDecider("jev", client, "typesafe/jev-1.14").decide(make_sample(image), make_evidence())
    assert out.parse_ok is False
    assert out.condition is None
    assert out.severity is None


def test_jev_decider_openrouter_error(image):
    client = FakeClient(error=OpenRouterError("boom", status=500))
    out = JevDecider("jev", client, "typesafe/jev-1.14").decide(make_sample(image), make_evidence())
    assert out.parse_ok is False
    assert "boom" in out.error


def test_routing_metrics_hand_computed():
    rows = [
        {
            "decider": "a", "image_id": "1", "class_name": "tench",
            "true_condition": "blur", "true_degraded": True, "true_severity": 2,
            "pred_condition": "blur", "pred_degraded": True, "pred_severity": 2,
            "p_degraded": 0.9, "confidence": 0.8, "parse_ok": True,
            "latency_ms": 10.0, "cost": 0.1, "cached": False, "error": None,
        },
        {
            "decider": "a", "image_id": "2", "class_name": "tench",
            "true_condition": "noise", "true_degraded": True, "true_severity": 3,
            "pred_condition": "blur", "pred_degraded": False, "pred_severity": 1,
            "p_degraded": 0.3, "confidence": 0.6, "parse_ok": False,
            "latency_ms": 30.0, "cost": 0.2, "cached": False, "error": None,
        },
        {
            "decider": "a", "image_id": "3", "class_name": "tench",
            "true_condition": "clean", "true_degraded": False, "true_severity": None,
            "pred_condition": "clean", "pred_degraded": False, "pred_severity": None,
            "p_degraded": 0.1, "confidence": 0.9, "parse_ok": True,
            "latency_ms": 20.0, "cost": 0.0, "cached": True, "error": None,
        },
    ]
    (metrics,) = routing_metrics(rows)
    assert metrics["decider"] == "a"
    assert metrics["n"] == 3
    assert metrics["degraded_accuracy"] == pytest.approx(2 / 3)
    assert metrics["degraded_recall"] == pytest.approx(1 / 2)
    assert metrics["clean_false_alarm_rate"] == pytest.approx(0.0)
    assert metrics["condition_accuracy"] == pytest.approx(2 / 3)
    assert 0.0 <= metrics["condition_macro_f1"] <= 1.0
    assert metrics["severity_exact"] == pytest.approx(1 / 2)
    assert metrics["severity_within1"] == pytest.approx(1 / 2)
    assert metrics["severity_mae"] == pytest.approx(1.0)
    assert metrics["parse_failure_rate"] == pytest.approx(1 / 3)
    assert metrics["latency_p50_ms"] == pytest.approx(20.0)
    assert metrics["latency_p95_ms"] == pytest.approx(29.0)
    assert metrics["cached_rate"] == pytest.approx(1 / 3)
    assert metrics["total_cost"] == pytest.approx(0.3)

    (per_condition,) = per_condition_metrics([rows[0]])
    assert per_condition["decider"] == "a"
    assert per_condition["true_condition"] == "blur"
    assert per_condition["n"] == 1
    assert per_condition["degraded_recall"] == pytest.approx(1.0)
    assert per_condition["condition_accuracy"] == pytest.approx(1.0)
    assert per_condition["severity_mae"] == pytest.approx(0.0)


def test_score_rows_shape():
    row = {
        "decider": "a", "image_id": "1", "class_name": "tench",
        "true_condition": "blur", "true_degraded": True, "true_severity": 2,
        "pred_condition": "blur", "pred_degraded": True, "pred_severity": 2,
        "p_degraded": 0.9, "confidence": 0.8, "parse_ok": True,
        "latency_ms": 10.0, "cost": 0.1, "cached": False, "error": None,
    }
    (scored,) = score_rows([row])
    assert set(scored) == {
        "decider", "image_id", "true_degraded", "p_degraded", "pred_degraded",
        "degraded_correct", "true_condition", "pred_condition",
        "true_severity", "pred_severity", "parse_ok", "cost", "cached",
    }
    assert scored["degraded_correct"] is True
    assert scored["p_degraded"] == pytest.approx(0.9)


def test_build_hf_prompt_contains_schema_and_evidence():
    prompt = build_hf_prompt(make_evidence(), "family")
    assert "blur" in prompt
    assert "noise" in prompt
    assert '"degraded"' in prompt
    assert '"confidence"' in prompt
    assert "severity" in prompt.lower()
    assert "128" in prompt
    assert "```json" not in prompt
    assert "Example clean" in prompt
    assert "Use severity 1 when the view is clean" in prompt


def test_build_hf_prompt_excludes_detector_by_default():
    assert "gaussian_blur" not in build_hf_prompt(make_evidence(), "family")
    assert "gaussian_blur" in build_hf_prompt(make_evidence(), "family", include_detector=True)


def test_jev_confidence_falls_back_to_max_probability(image):
    answers = {
        "degraded": {"noul": 0.9},
        "condition": {"choice": "blur", "probabilities": {"blur": 0.6, "noise": 0.4}},
        "severity": {"score": 1.2, "confidence": 0.99},
    }
    client = FakeClient(answers=answers)
    out = JevDecider("jev", client, "typesafe/jev-1.14").decide(make_sample(image), make_evidence())
    assert out.confidence == pytest.approx(0.6)
