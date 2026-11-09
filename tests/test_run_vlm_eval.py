"""Offline tests for the VLM occlusion evaluation harness (Checkpoint 6).

No model is loaded and no network call is made: the classifier and VLM client
are injected fakes. The harness module lives under `scripts/`, which is not a
package, so it is imported by adding that directory to `sys.path`.
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
from PIL import Image

from bolero.config import SEED
from bolero.data import ImageRecord
from bolero.prompting import (
    RegionEvidence,
    SceneEvidence,
    image_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script():
    if "run_vlm_eval" in sys.modules:
        return sys.modules["run_vlm_eval"]
    return importlib.import_module("run_vlm_eval")


def make_args(tmp_path, **overrides):
    values = dict(
        per_class=1,
        data_root=tmp_path / "imagenette2-320",
        conditions=["region_removal"],
        levels=[2],
        configs=["none", "text"],
        include_clean=False,
        limit=0,
        sample=[],
        seed=SEED,
        vlm_model="fake-vlm",
        max_new_tokens=64,
        cache_dir=tmp_path / "cache",
        out=tmp_path / "out",
        detector=tmp_path / "detector.joblib",
        depth_root=tmp_path / "depth",
        classifier="resnet50",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_evidence(image, candidates=None):
    height, width = image.shape[:2]
    return SceneEvidence(
        (width, height),
        list(candidates or [("tench", 0.9)]),
        entity=RegionEvidence((40, 30, 90, 70), 0.2, 0.9),
        damaged=RegionEvidence((10, 10, 50, 50), 0.1, 0.4),
        depth_layout={"near_center_x": 0.5, "near_center_y": 0.5, "depth_spread": 0.4},
        stats=image_statistics(image),
    )


def write_jpeg(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path, format="JPEG")
    return path


def make_record(tmp_path: Path, image_id: str, image: np.ndarray) -> ImageRecord:
    path = write_jpeg(tmp_path / "val" / "n01440764" / f"{image_id}.JPEG", image)
    return ImageRecord(image_id=image_id, path=path, wnid="n01440764")


class FakeClassifier:
    def __init__(self, label: str = "tench"):
        self.label = label

    def predict_label(self, image):
        return self.label

    def predict_candidates(self, image):
        return [(self.label, 0.9), ("goldfish", 0.05)]

    def predict(self, images, true_index):
        raise AssertionError("fake classifier should not call predict directly")


CANNED_ANSWER = json.dumps(
    {
        "entity": "tench",
        "damaged_region_content": "tench",
        "relation": "overlaps_entity",
        "confidence": 0.8,
        "rationale": "the damaged box sits on the fish",
    }
)


class FakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def ask(self, images, prompt, cache_key, use_cache=True):
        self.calls.append(
            {"n_images": len(images), "prompt": prompt, "cache_key": cache_key}
        )
        return SimpleNamespace(text=CANNED_ANSWER, latency_ms=1.5, cached=False)


def fake_depth(record, image):
    height, width = image.shape[:2]
    depth = np.tile(np.linspace(0.0, 0.5, height, dtype=np.float32)[:, None], (1, width))
    depth[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.0
    return depth


def row(**overrides):
    values = {
        "image_id": "img-0",
        "class_name": "tench",
        "wnid": "n01440764",
        "true_class": "tench",
        "clean_prediction": "tench",
        "degraded_prediction": "tench",
        "perturbation": "region_removal",
        "level": 2,
        "config": "text",
        "entity_box": "40,30,90,70",
        "damaged_box": "10,10,50,50",
        "damage_overlap": 0.0,
        "parse_ok": True,
        "entity": "tench",
        "damaged_region_content": "tench",
        "relation_answer": "overlaps_entity",
        "confidence": 0.8,
        "rationale": "",
        "entity_hit": True,
        "damage_eligible": True,
        "damage_hit": True,
        "relation_geometric": "overlaps_entity",
        "relation_hit": True,
        "relation_hybrid": "overlaps_entity",
        "relation_hybrid_source": "geometric",
        "utility": True,
        "degraded_correct": True,
        "latency_ms": 1.5,
        "cached": False,
        "text": CANNED_ANSWER,
    }
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_config_names_validate():
    module = load_script()
    module.validate_configs(list(module.CONFIGS))
    module.validate_configs(["none", "text"])

    with pytest.raises(ValueError):
        module.validate_configs(["bogus"])
    with pytest.raises(SystemExit):
        module.parse_args(["--configs", "bogus"])

    args = module.parse_args([])
    assert args.configs == ["none", "text", "overlay", "overlay+text"]
    assert args.include_clean is True


def test_build_config_images_and_prompt(image):
    module = load_script()
    evidence = make_evidence(image)

    images, prompt = module.build_config("none", image, evidence)
    assert len(images) == 1
    assert prompt == module.VLM_INSTRUCTION
    assert "EVIDENCE" not in prompt
    assert np.array_equal(images[0], image)

    images, prompt = module.build_config("text", image, evidence)
    assert len(images) == 1
    assert "EVIDENCE" in prompt
    assert "tench" in prompt
    assert np.array_equal(images[0], image)

    images, prompt = module.build_config("overlay", image, evidence)
    assert len(images) == 1
    assert not np.array_equal(images[0], image)

    images, prompt = module.build_config("overlay+text", image, evidence)
    assert len(images) == 1
    assert "EVIDENCE" in prompt

    images, prompt = module.build_config("overlay+depth", image, evidence)
    assert len(images) == 2
    assert prompt == module.VLM_INSTRUCTION
    assert images[1].shape[:2] == image.shape[:2]

    _, prompt = module.build_config("minus_regions", image, evidence)
    assert '"entity_region"' not in prompt
    assert '"damaged_region"' not in prompt
    assert "tench" in prompt

    _, prompt = module.build_config("minus_candidates", image, evidence)
    assert "classifier_candidates" not in prompt

    _, prompt = module.build_config("minus_depth", image, evidence)
    assert "depth_layout" not in prompt

    detector = SimpleNamespace(
        decide=lambda img: SimpleNamespace(label="region_removal", probability=0.77)
    )
    _, prompt = module.build_config("with_detector", image, evidence, detector=detector)
    assert "region_removal" in prompt

    with pytest.raises(ValueError):
        module.build_config("bogus", image, evidence)
    with pytest.raises(ValueError):
        module.build_config("with_detector", image, evidence, detector=None)


def test_depth_image(image):
    module = load_script()
    depth = np.linspace(0.0, 1.0, image.shape[0] * image.shape[1], dtype=np.float32).reshape(
        image.shape[:2]
    )
    rendered = module.depth_image(depth)
    assert rendered.shape == image.shape
    assert rendered.ndim == 3 and rendered.shape[2] == 3
    assert rendered.dtype in (np.float32, np.uint8)
    assert float(np.asarray(rendered, dtype=np.float32).min()) >= 0.0
    assert float(np.asarray(rendered, dtype=np.float32).max()) <= 1.0


def test_cache_key_stable():
    module = load_script()
    first = module.cache_key("img-1", "region_removal", 2, "text")
    second = module.cache_key("img-1", "region_removal", 2, "text")
    other = module.cache_key("img-1", "region_removal", 2, "overlay")
    assert first == second == "img-1|region_removal|2|text"
    assert other != first


# ---------------------------------------------------------------------------
# Harness core
# ---------------------------------------------------------------------------


def test_core_with_fake_client(tmp_path, image):
    module = load_script()
    args = make_args(tmp_path, conditions=["region_removal"], configs=["none", "text"])
    record = make_record(tmp_path, "img-0", image)
    classifier = FakeClassifier()
    client = FakeClient()

    result = module.run(
        args,
        [record],
        classifier=classifier,
        client=client,
        depth_for=fake_depth,
    )

    out = args.out
    assert (out / "summary.csv").exists()
    assert (out / "predictions.csv.gz").exists()
    assert (out / "per_condition.csv").exists()
    assert (out / "headline.json").exists()
    assert (out / "run.json").exists()

    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["config"]) == {"none", "text"}
    assert len(summary) == 2

    assert len(client.calls) == 2
    keys = {call["cache_key"] for call in client.calls}
    assert keys == {"img-0|region_removal|2|none", "img-0|region_removal|2|text"}

    predictions = pd.read_csv(out / "predictions.csv.gz")
    assert len(predictions) == 2
    expected = {
        "entity_hit",
        "damage_eligible",
        "damage_hit",
        "relation_geometric",
        "relation_hit",
        "utility",
        "degraded_correct",
    }
    assert expected <= set(predictions.columns)
    assert predictions["entity_hit"].all()

    headline = json.loads((out / "headline.json").read_text())
    assert headline["n_rows"] == 2
    assert headline["n_images"] == 1
    assert headline["vlm"] == "fake-vlm"
    assert result["n_rows"] == 2


def test_sample_selection_and_limit(tmp_path, image):
    module = load_script()
    records = [
        make_record(tmp_path, "a", image),
        make_record(tmp_path, "b", image),
        make_record(tmp_path, "c", image),
    ]
    args = make_args(tmp_path, sample=["c"], limit=2)

    selected, skipped = module.resolve_records(
        args, records=records, universe=records
    )
    assert [record.image_id for record in selected] == ["c", "a"]
    assert len(selected) == 2
    assert skipped == []

    args = make_args(tmp_path, sample=["missing"], limit=0)
    selected, skipped = module.resolve_records(
        args, records=records, universe=records
    )
    assert len(selected) == 3
    assert skipped == [{"image_id": "missing", "reason": "not found in val split"}]


def test_summary_metrics():
    module = load_script()
    rows = [
        row(
            entity_hit=True,
            damage_eligible=True,
            damage_hit=True,
            relation_geometric="overlaps_entity",
            relation_hit=True,
            relation_hybrid="overlaps_entity",
            utility=True,
            degraded_correct=True,
            latency_ms=10.0,
        ),
        row(
            entity_hit=False,
            damage_eligible=True,
            damage_hit=False,
            relation_geometric="adjacent",
            relation_hit=False,
            relation_hybrid="adjacent",
            utility=False,
            degraded_correct=False,
            latency_ms=20.0,
        ),
        row(
            entity_hit=True,
            damage_eligible=False,
            damage_hit=False,
            relation_geometric="unclear",
            relation_hit=None,
            relation_hybrid="background",
            relation_hybrid_source="vlm",
            utility=True,
            degraded_correct=True,
            latency_ms=30.0,
        ),
    ]

    summary = module.summary_table(rows)
    assert len(summary) == 1
    entry = summary[0]
    assert entry["config"] == "text"
    assert entry["n"] == 3
    assert entry["parse_rate"] == pytest.approx(1.0)
    assert entry["entity_hit"] == pytest.approx(2 / 3)
    assert entry["damage_n"] == 2
    assert entry["damage_hit"] == pytest.approx(0.5)
    assert entry["relation_n"] == 2
    assert entry["relation_hit"] == pytest.approx(0.5)
    assert entry["relation_hybrid_coverage"] == pytest.approx(1.0)
    assert entry["utility"] == pytest.approx(2 / 3)
    assert entry["degraded_correct"] == pytest.approx(2 / 3)
    assert entry["error_rate"] == pytest.approx(0.0)
    assert entry["latency_p50_ms"] == pytest.approx(20.0)

    per_condition = module.per_condition_table(rows)
    assert len(per_condition) == 1
    assert per_condition[0]["perturbation"] == "region_removal"
    assert per_condition[0]["n"] == 3
    assert per_condition[0]["damage_hit"] == pytest.approx(0.5)
