"""Offline tests for the decider comparison harness (Checkpoint 4).

No network, no real model loads, no model downloads. The harness module lives
under `scripts/`, which is not a package, so it is imported by adding that
directory to `sys.path`.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bolero.config import SEED
from bolero.conditions import ConditionSample, TAXONOMIES
from bolero.deciders import DeciderOutput
from bolero.openrouter import BudgetExceeded
from bolero.perturbations import ALL_PERTURBATIONS

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

EXPECTED_ROW_FIELDS = {
    "decider",
    "image_id",
    "class_name",
    "true_condition",
    "true_degraded",
    "true_severity",
    "pred_condition",
    "pred_degraded",
    "pred_severity",
    "p_degraded",
    "confidence",
    "parse_ok",
    "latency_ms",
    "cost",
    "cached",
    "error",
    "threshold",
}


def load_script():
    if "run_decider_compare" in sys.modules:
        return sys.modules["run_decider_compare"]
    return importlib.import_module("run_decider_compare")


def make_args(tmp_path, **overrides):
    values = dict(
        per_class=1,
        max_per_condition=1,
        taxonomy="family",
        perturbations=list(ALL_PERTURBATIONS),
        levels=[1],
        data_root=None,
        seed=SEED,
        deciders=[],
        evidence_mode="stats",
        include_detector=False,
        include_candidates=False,
        exemplars=0,
        max_calls=5000,
        max_cost=2.0,
        workers=2,
        fit_severity_per_class=0,
        conformal=None,
        out=tmp_path / "out",
        limit=0,
        rf_detector=tmp_path / "missing-detector.joblib",
        severity_model=tmp_path / "missing-severity.joblib",
        hf_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_sample(index, condition="blur", severity=2, perturbation="gaussian_blur"):
    rng = np.random.default_rng(index)
    image = np.clip(rng.random((32, 32, 3)), 0.05, 0.95).astype(np.float32)
    return ConditionSample(
        image_id=f"img-{index}",
        wnid="n01440764",
        class_name="tench",
        condition=condition,
        severity=severity,
        image=image,
        clean_image=image,
        perturbation=perturbation,
        level=severity,
    )


def output(
    degraded=True,
    condition="blur",
    severity=2,
    p_degraded=0.9,
    parse_ok=True,
    cost=0.0,
    cached=False,
):
    return DeciderOutput(
        degraded=degraded,
        condition=condition,
        severity=severity,
        confidence=0.8,
        p_degraded=p_degraded,
        probabilities={condition: 0.8} if condition else {},
        parse_ok=parse_ok,
        latency_ms=1.5,
        cost=cost,
        cached=cached,
        raw=None,
    )


class FakeDecider:
    def __init__(self, name, **kwargs):
        self.name = name
        self.result = output(**kwargs)
        self.calls = 0

    def available(self):
        return True, ""

    def decide(self, sample, evidence):
        self.calls += 1
        return self.result


class FakeHostedDecider(FakeDecider):
    hosted = True


class RaisingDecider(FakeDecider):
    def decide(self, sample, evidence):
        raise BudgetExceeded("out of budget")


# ---------------------------------------------------------------------------
# Row construction and conformal override
# ---------------------------------------------------------------------------


def test_build_row_shape_and_conformal_override():
    module = load_script()
    sample = make_sample(0)

    row = module.build_row("fake", sample, output(degraded=False, p_degraded=0.9))
    assert set(row) == EXPECTED_ROW_FIELDS
    assert row["decider"] == "fake"
    assert row["image_id"] == "img-0"
    assert row["class_name"] == "tench"
    assert row["true_condition"] == "blur"
    assert row["true_degraded"] is True
    assert row["true_severity"] == 2
    assert row["pred_degraded"] is False
    assert row["threshold"] is None

    overridden = module.build_row(
        "fake", sample, output(degraded=False, p_degraded=0.9), threshold=0.6
    )
    assert overridden["pred_degraded"] is True
    assert overridden["threshold"] == 0.6

    below = module.build_row(
        "fake", sample, output(degraded=True, p_degraded=0.4), threshold=0.6
    )
    assert below["pred_degraded"] is False

    # A threshold cannot override a missing probability.
    assert module.build_row(
        "fake", sample, output(p_degraded=None), threshold=0.6
    ) is None


def test_selected_deciders_include_aliases():
    module = load_script()
    args = make_args(Path("/tmp/unused"))
    deciders, skipped = module.build_deciders(args, ["majority", "always_clean"])
    names = [decider.name for decider in deciders]
    assert names == ["majority", "always_clean"]
    assert skipped == []

    majority = deciders[0]
    assert majority.degraded is True
    assert majority.condition == module.majority_condition("family")
    assert majority.severity == 2

    always_clean = deciders[1]
    assert always_clean.degraded is False
    assert always_clean.condition == "clean"

    degraded, _ = module.build_deciders(args, ["always_degraded"])
    assert degraded[0].degraded is True
    assert degraded[0].condition == module.majority_condition("family")
    assert degraded[0].severity == 2


def test_conformal_file_parsing(tmp_path):
    module = load_script()
    path = tmp_path / "conformal.json"
    path.write_text(json.dumps({"jev": 0.4, "rf": 0.55}))

    per_decider, default = module.load_conformal(path)
    assert per_decider == {"jev": 0.4, "rf": 0.55}
    assert default is None

    path.write_text(json.dumps({"threshold": 0.5}))
    per_decider, default = module.load_conformal(path)
    assert per_decider == {}
    assert default == 0.5

    assert module.load_conformal(None) == ({}, None)


def test_depth_cache_version():
    module = load_script()
    assert module.depth_version(Path("/data/imagenette2-320")) == "320"
    assert module.depth_version(Path("/data/imagenette2-160")) == "160"
    assert module.depth_version(None) == "160"
    assert module.depth_version(Path("/tmp/some-other-dir")) == "160"


# ---------------------------------------------------------------------------
# Harness core
# ---------------------------------------------------------------------------


def test_smoke_with_fake_deciders(tmp_path):
    module = load_script()
    args = make_args(tmp_path, workers=1, max_calls=0)
    samples = [
        make_sample(0),
        make_sample(1, condition="clean", severity=None, perturbation=None),
    ]
    deciders = [FakeDecider("alpha"), FakeDecider("beta", degraded=False, condition="clean", severity=None)]

    result = module.run(args, deciders, samples)

    out = args.out
    assert (out / "summary.csv").exists()
    assert (out / "per_condition.csv").exists()
    assert (out / "scores.csv").exists()
    assert (out / "predictions.csv.gz").exists()
    assert (out / "headline.json").exists()
    assert (out / "run.json").exists()

    summary = pd.read_csv(out / "summary.csv")
    assert {"decider", "n", "degraded_accuracy", "clean_false_alarm_rate",
            "condition_accuracy", "severity_mae", "latency_p50_ms", "total_cost"} <= set(summary.columns)
    assert set(summary["decider"]) == {"alpha", "beta"}
    assert len(summary) == 2

    predictions = pd.read_csv(out / "predictions.csv.gz")
    assert EXPECTED_ROW_FIELDS <= set(predictions.columns)
    assert len(predictions) == 4
    assert result["n_samples"] == 2

    scores = pd.read_csv(out / "scores.csv")
    assert {"decider", "true_degraded", "p_degraded", "pred_degraded", "degraded_correct"} <= set(scores.columns)

    headline = json.loads((out / "headline.json").read_text())
    assert headline["n_rows"] == 4
    assert headline["n_samples"] == 2
    assert headline["cached_rate"] == 0.0
    assert len(headline["per_decider"]) == 2
    assert headline["skipped"] == []

    run_json = json.loads((out / "run.json").read_text())
    assert {"config", "started", "finished", "wall_seconds"} <= set(run_json)


def test_max_calls_stops_submission(tmp_path):
    module = load_script()
    args = make_args(tmp_path, workers=1, max_calls=1)
    samples = [make_sample(i) for i in range(4)]
    decider = FakeHostedDecider("api")

    result = module.run(args, [decider], samples)

    assert decider.calls <= 1
    assert result["n_samples"] == 1
    assert result["stopped"] == "max_calls"
    assert (args.out / "predictions.csv.gz").exists()
    assert (args.out / "summary.csv").exists()

    predictions = pd.read_csv(args.out / "predictions.csv.gz")
    assert len(predictions) <= 1


def test_budget_exceeded_writes_partial_results(tmp_path):
    module = load_script()
    args = make_args(tmp_path, workers=1)
    samples = [make_sample(i) for i in range(2)]

    result = module.run(args, [RaisingDecider("noisy")], samples)

    assert result["stopped"] is not None
    assert "budget" in result["stopped"].lower()
    assert (args.out / "summary.csv").exists()
    assert (args.out / "predictions.csv.gz").exists()


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def test_majority_condition_is_most_frequent():
    from collections import Counter

    from bolero.conditions import condition_label

    module = load_script()
    counts = Counter(
        condition_label(name, perturbation.family, "family")
        for name, perturbation in ALL_PERTURBATIONS.items()
    )
    best = module.majority_condition("family")
    assert counts[best] == max(counts.values())
    assert module.majority_condition("weather") == "other"


def test_depth_cache_version_honors_env(monkeypatch):
    module = load_script()
    monkeypatch.setenv("IMAGENETTE_VERSION", "320")
    assert module.depth_version(None) == "320"
    assert module.depth_version(Path("data/imagenette2-320")) == "320"
    assert module.depth_version(Path("data/imagenette2-160")) == "160"


def test_hosted_call_counter_ignores_cache(tmp_path):
    module = load_script()
    args = make_args(tmp_path)
    counter = module._CallCounter()

    cached = FakeHostedDecider("jev", cached=True)
    module._process_sample(make_sample(0), [cached], args, None, None, counter, {}, None)
    assert counter.value == 0

    live = FakeHostedDecider("jev", cached=False)
    module._process_sample(make_sample(1), [live], args, None, None, counter, {}, None)
    assert counter.value == 1


def test_deciders_accept_comma_separated_list(tmp_path):
    module = load_script()
    args = make_args(tmp_path)
    deciders, skipped = module.build_deciders(args, ["majority,always_clean"])
    assert [decider.name for decider in deciders] == ["majority", "always_clean"]
    assert skipped == []
