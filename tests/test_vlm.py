"""Offline tests for the local VLM client and spatial scoring helpers.

No model is loaded and no network call is made: generation is always injected
through a fake ``generate_fn`` and the cache lives under ``tmp_path``.
"""

from __future__ import annotations

import json

import numpy as np

from bolero.perturbations import BLACK, GRAY
from bolero.vlm import VLMClient
from bolero.vlm_scoring import (
    box_iou,
    damaged_mask,
    match_label,
    normalize_text,
    overlap_fraction,
    parse_vlm_answer,
    relation_bucket,
    score_case,
)


def test_normalize_text():
    assert normalize_text("  A Tench, Tinca tinca! ") == "a tench tinca tinca"
    assert normalize_text(None) == ""


def test_match_label_aliases():
    assert match_label("a springer spaniel running", "English springer")
    assert match_label("a large garbage truck", "garbage truck")
    assert not match_label("a church", "chain saw")
    assert not match_label(None, "tench")
    assert not match_label("", "tench")


def test_match_label_imagenet_synonyms():
    assert match_label("a tench swimming", "tench, Tinca tinca")


def test_box_geometry():
    box = (10, 10, 50, 50)
    assert box_iou(box, box) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert overlap_fraction((12, 12, 18, 18), box) == 1.0
    assert box_iou((0, 0, 0, 0), box) == 0.0


def test_relation_bucket():
    shape = (320, 320)
    entity = (10, 10, 110, 110)
    assert relation_bucket(entity, (30, 30, 130, 130), shape) == "overlaps_entity"
    assert relation_bucket(entity, (130, 10, 230, 110), shape) == "adjacent"
    assert relation_bucket(entity, (250, 250, 300, 300), shape) == "background"
    assert relation_bucket(None, (30, 30, 130, 130), shape) == "unclear"
    assert relation_bucket(entity, (0, 0, 0, 0), shape) == "unclear"


def test_damaged_mask():
    clean = np.full((64, 64, 3), 0.25, dtype=np.float32)

    black = clean.copy()
    black[10:30, 10:30] = BLACK
    mask = damaged_mask("region_removal", black, clean, None)
    assert mask is not None and mask.dtype == bool
    assert mask[10:30, 10:30].all()
    assert not mask[40:50, 40:50].any()

    gray = clean.copy()
    gray[5:20, 30:50] = GRAY
    mask = damaged_mask("foreground_occlusion", gray, clean, None)
    assert mask is not None and mask[5:20, 30:50].all()
    assert not mask[30:40, 0:10].any()

    assert damaged_mask("gaussian_blur", clean, clean, None) is None


def test_parse_vlm_answer():
    plain = json.dumps(
        {
            "entity": "tench",
            "damaged_region_content": "water",
            "relation": "adjacent",
            "confidence": 0.8,
            "rationale": "the fish is beside it",
        }
    )
    parsed = parse_vlm_answer(plain)
    assert parsed == {
        "entity": "tench",
        "damaged_region_content": "water",
        "relation": "adjacent",
        "confidence": 0.8,
        "rationale": "the fish is beside it",
    }

    fenced = parse_vlm_answer('```json\n{"entity": "tench", "relation": "background"}\n```')
    assert fenced is not None and fenced["relation"] == "background"

    unknown = parse_vlm_answer('{"entity": "tench", "relation": "on_top_of"}')
    assert unknown is not None and unknown["relation"] == "unclear"

    assert parse_vlm_answer("no json here") is None
    assert parse_vlm_answer(None) is None


def test_score_case():
    answer = parse_vlm_answer(
        json.dumps(
            {
                "entity": "tench",
                "damaged_region_content": "tench",
                "relation": "overlaps_entity",
                "confidence": 0.9,
                "rationale": "the box sits on the fish",
            }
        )
    )
    assert answer is not None

    scored = score_case(
        answer,
        true_class="tench",
        clean_prediction="tench",
        degraded_prediction="tench",
        entity_box=(10, 10, 110, 110),
        damaged_box=(40, 40, 140, 140),
        shape=(320, 320),
    )
    assert scored["entity_hit"] is True
    assert scored["damage_eligible"] is True
    assert scored["damage_hit"] is True
    assert scored["relation_geometric"] == "overlaps_entity"
    assert scored["relation_hit"] is True
    assert scored["relation_hybrid"] == "overlaps_entity"
    assert scored["relation_hybrid_source"] == "geometric"
    assert scored["utility"] is True
    assert scored["degraded_correct"] is True


def test_score_case_ineligible_damage():
    answer = parse_vlm_answer('{"entity": "tench", "damaged_region_content": "tench"}')
    assert answer is not None
    scored = score_case(
        answer,
        true_class="tench",
        clean_prediction="tench",
        degraded_prediction="church",
        entity_box=(10, 10, 110, 110),
        damaged_box=(100, 100, 130, 130),
        shape=(320, 320),
    )
    assert scored["damage_eligible"] is False
    assert scored["damage_hit"] is False
    assert scored["relation_geometric"] == "adjacent"
    assert scored["relation_hybrid_source"] == "geometric"
    assert scored["degraded_correct"] is False


def test_vlm_client_available_and_cache(tmp_path):
    calls = []

    def fake_generate(images, prompt):
        calls.append((images, prompt))
        return '{"entity": "tench"}'

    client = VLMClient(generate_fn=fake_generate, cache_dir=tmp_path)
    ok, reason = client.available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)

    first = client.ask([np.zeros((16, 16, 3), dtype=np.float32)], "prompt", "case-1")
    assert first.text == '{"entity": "tench"}'
    assert first.cached is False
    assert first.raw is None
    assert first.latency_ms >= 0.0
    assert len(calls) == 1

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    assert payload["text"] == first.text
    assert "api_key" not in cache_files[0].read_text().lower()

    second = client.ask([np.zeros((16, 16, 3), dtype=np.float32)], "prompt", "case-1")
    assert second.cached is True
    assert second.text == first.text
    assert len(calls) == 1


def test_vlm_client_multi_image_prompt(tmp_path):
    seen = {}

    def fake_generate(images, prompt):
        seen["n_images"] = len(images)
        seen["prompt"] = prompt
        return "ok"

    client = VLMClient(generate_fn=fake_generate, cache_dir=tmp_path)
    images = [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.ones((8, 8, 3), dtype=np.float32),
    ]
    response = client.ask(images, "two views", "multi")
    assert response.text == "ok"
    assert seen["n_images"] == 2
    assert seen["prompt"] == "two views"
