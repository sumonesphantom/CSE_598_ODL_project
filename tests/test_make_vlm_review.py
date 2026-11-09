"""Offline tests for the 30-case VLM human review workflow (Checkpoint 7).

No model is loaded and no network call is made: clean images and depth maps are
synthetic, and the depth map is injected so `DepthCache` never builds an
estimator. The script lives under `scripts/`, which is not a package, so it is
imported by adding that directory to `sys.path`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bolero.config import SEED
from bolero.data import to_uint8

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script():
    if "make_vlm_review" in sys.modules:
        return sys.modules["make_vlm_review"]
    return importlib.import_module("make_vlm_review")


PERTURBATIONS = [
    "region_removal",
    "foreground_occlusion",
    "boundary_erase",
    "background_removal",
]
CLASSES = ["tench", "church", "parachute"]


def prediction_row(**overrides):
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
        "parse_ok": True,
        "entity": "tench",
        "damaged_region_content": "tench",
        "relation_answer": "overlaps_entity",
        "confidence": 0.8,
        "entity_hit": True,
        "damage_eligible": True,
        "damage_hit": True,
        "relation_geometric": "overlaps_entity",
        "relation_hit": True,
    }
    values.update(overrides)
    return values


def synthetic_predictions() -> pd.DataFrame:
    """4 perturbations x 3 classes x 2 configs, two levels, plus clean controls."""
    rows: list[dict] = []
    for perturbation in PERTURBATIONS:
        for index, class_name in enumerate(CLASSES):
            image_id = f"{perturbation}-{index}"
            for config in ("none", "text"):
                for level in (1, 2):
                    rows.append(
                        prediction_row(
                            image_id=image_id,
                            class_name=class_name,
                            perturbation=perturbation,
                            level=level,
                            config=config,
                        )
                    )
                rows.append(
                    prediction_row(
                        image_id=image_id,
                        class_name=class_name,
                        perturbation="clean",
                        level=0,
                        config=config,
                    )
                )
    return pd.DataFrame(rows)


def write_jpeg(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path, format="JPEG")
    return path


def make_scene(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """A temp data root with one Imagenette-style val image and a depth map."""
    data_root = tmp_path / "imagenette2-320"
    height, width = 240, 480
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height, width])[:, None, None]
    image = np.stack([0.2 + 0.5 * xx, 0.3 + 0.4 * yy, 0.5 - 0.3 * xx], axis=-1)
    image = np.clip(image, 0.05, 0.95).astype(np.float32)
    write_jpeg(data_root / "val" / "n01440764" / "img-0.JPEG", image)

    depth = np.tile(np.linspace(0.0, 0.5, height, dtype=np.float32)[:, None], (1, width))
    depth[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.0
    return data_root, image, depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------


def test_select_cases_stratified():
    module = load_script()
    frame = synthetic_predictions()

    selected = module.select_cases(frame, cases=8, configs=["none", "text"], seed=SEED)

    assert len({row["image_id"] for row in selected}) <= 8
    assert {row["perturbation"] for row in selected} == set(PERTURBATIONS)
    assert all(row["perturbation"] != "clean" for row in selected)
    assert all(int(row["level"]) == 2 for row in selected)

    by_image: dict[str, list[str]] = {}
    for row in selected:
        by_image.setdefault(row["image_id"], []).append(row["config"])
    for configs in by_image.values():
        assert sorted(configs) == ["none", "text"]
        assert len(configs) == 2
    assert len(selected) == 2 * len(by_image)

    again = module.select_cases(frame, cases=8, configs=["none", "text"], seed=SEED)
    assert selected == again


# ---------------------------------------------------------------------------
# Packet building
# ---------------------------------------------------------------------------


def test_review_rows_and_images(tmp_path):
    module = load_script()
    data_root, _, depth = make_scene(tmp_path)
    out = tmp_path / "review"
    row = prediction_row()

    result = module.build_review(
        [row],
        out=out,
        data_root=data_root,
        depth_root=tmp_path / "depth",
        depth_for=lambda record, image: depth,
    )

    case_id = "img-0_text"
    for suffix in ("clean", "degraded", "overlay"):
        path = out / f"{case_id}_{suffix}.jpg"
        assert path.is_file()
        with Image.open(path) as thumb:
            assert thumb.width <= 320

    review = pd.read_csv(out / "review.csv")
    assert list(review.columns) == module.REVIEW_FIELDS
    assert review.loc[0, "case_id"] == case_id
    for column in ("clean_image", "degraded_image", "overlay_image"):
        relative = review.loc[0, column]
        assert not Path(relative).is_absolute()
        assert (out / relative).is_file()

    assert (out / "contact_sheet.html").is_file()
    assert result["n_images"] == 1
    assert result["n_rows"] == 1


def test_contact_sheet_escapes_text(tmp_path):
    module = load_script()
    data_root, _, depth = make_scene(tmp_path)
    out = tmp_path / "review"
    row = prediction_row(
        config="none",
        entity="<script>alert(1)</script> tench",
        damaged_region_content="<script>bad()</script>",
    )

    module.build_review(
        [row],
        out=out,
        data_root=data_root,
        depth_root=tmp_path / "depth",
        depth_for=lambda record, image: depth,
    )

    html = (out / "contact_sheet.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def write_review_csv(path: Path, rows: list[dict]) -> Path:
    columns = [
        "config",
        "manual_entity_correct",
        "auto_entity_hit",
        "manual_damage_correct",
        "auto_damage_hit",
        "manual_relation_correct",
        "auto_relation_hit",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_score_review_agreement(tmp_path):
    module = load_script()
    entity_manual = [1, 0, 1, 0]
    entity_auto = [1, 0, 1, 0]
    damage_manual = [1, 1, 0, 0]
    damage_auto = [1, 0, 1, 0]
    relation_manual = [1, 1, None, None]
    relation_auto = [1, 1, None, None]

    rows = [
        {
            "config": "none",
            "manual_entity_correct": entity_manual[i],
            "auto_entity_hit": entity_auto[i],
            "manual_damage_correct": damage_manual[i],
            "auto_damage_hit": damage_auto[i],
            "manual_relation_correct": relation_manual[i],
            "auto_relation_hit": relation_auto[i],
        }
        for i in range(4)
    ]
    path = write_review_csv(tmp_path / "review.csv", rows)

    result = module.score_review(path)

    entity = result["fields"]["entity"]
    assert entity["n"] == 4
    assert entity["agreement"] == pytest.approx(1.0)
    assert entity["kappa"] == pytest.approx(1.0)

    damage = result["fields"]["damage"]
    assert damage["n"] == 4
    assert damage["agreement"] == pytest.approx(0.5)
    assert damage["kappa"] == pytest.approx(0.0)
    assert damage["manual_correct"] == 2
    assert damage["auto_correct"] == 2

    relation = result["fields"]["relation"]
    assert relation["n"] == 2
    assert relation["agreement"] == pytest.approx(1.0)

    assert result["overall"]["n"] == 10
    assert result["overall"]["agreement"] == pytest.approx(0.8)
    assert result["per_config"]["none"]["n"] == 10
    assert (path.parent / "agreement.json").is_file()


def test_blank_manual_values_are_skipped(tmp_path):
    module = load_script()
    rows = [
        {
            "config": "none",
            "manual_entity_correct": "1",
            "auto_entity_hit": 1,
            "manual_damage_correct": "",
            "auto_damage_hit": 1,
            "manual_relation_correct": "",
            "auto_relation_hit": 1,
        },
        {
            "config": "none",
            "manual_entity_correct": " 1 ",
            "auto_entity_hit": 1,
            "manual_damage_correct": None,
            "auto_damage_hit": 0,
            "manual_relation_correct": None,
            "auto_relation_hit": None,
        },
        {
            "config": "none",
            "manual_entity_correct": "",
            "auto_entity_hit": 0,
            "manual_damage_correct": "",
            "auto_damage_hit": 0,
            "manual_relation_correct": "",
            "auto_relation_hit": 0,
        },
        {
            "config": "none",
            "manual_entity_correct": "0",
            "auto_entity_hit": 0,
            "manual_damage_correct": "",
            "auto_damage_hit": 0,
            "manual_relation_correct": "",
            "auto_relation_hit": 0,
        },
    ]
    path = write_review_csv(tmp_path / "review.csv", rows)

    result = module.score_review(path)

    entity = result["fields"]["entity"]
    assert entity["n"] == 3
    assert entity["agreement"] == pytest.approx(1.0)
    assert entity["manual_correct"] == 2
    assert entity["auto_correct"] == 2

    assert result["fields"]["damage"]["n"] == 0
    assert result["fields"]["damage"]["agreement"] is None
    assert result["fields"]["relation"]["n"] == 0
