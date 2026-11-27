"""Stage 6 - evaluate a local VLM on occlusion images.

For every Imagenette val image, every selected occlusion condition and level,
and every evidence configuration, this harness asks Qwen2.5-VL to name the main
entity, describe what the damaged region contains, and state the spatial
relation between the two. Answers are scored against the true class and against
geometry derived from the clean depth map.

The entity region used for scoring and for the overlay is an *idealized* region:
it is computed once from the clean depth map with `near_mask(depth, 0.4)` and
does not depend on what the recognizer or detector believe. That keeps the
score honest (it does not reward the VLM for agreeing with a wrong entity box),
but it also means the box marks the nearest depth region rather than a true
instance segmentation.

Configurations:
  none              degraded image, base instruction
  text              degraded image + structured evidence text
  overlay           degraded image with entity/damaged boxes drawn
  overlay+text      overlay image + evidence text
  overlay+depth     overlay image + clean depth map rendered as grayscale RGB
  minus_candidates  text config with evidence.candidates = []
  minus_depth       text config with evidence.depth_layout = {}
  minus_regions     text config with evidence.entity = evidence.damaged = None
  with_detector     text config with the 320 detector's label and probability

Outputs (in --out):
  summary.csv          per-config metrics
  per_condition.csv    per-config x perturbation metrics
  predictions.csv.gz   one row per image x condition x level x config
  headline.json        config, headline numbers, per-config summary, skips
  run.json             config plus timestamps and wall time

Tests inject a fake classifier and a fake VLM client, so nothing here loads a
model or makes a network call unless `main()` builds the real clients.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from bolero.classifier import FrozenClassifier
from bolero.config import CACHE_DIR, DATA_DIR, REPO_ROOT, RESULTS_DIR, SEED, load_env
from bolero.data import ImageRecord, list_images, load_image
from bolero.depth import DepthCache, near_mask
from bolero.detection import PerturbationDetector
from bolero.perturbations import ALL_PERTURBATIONS, rng_for
from bolero.prompting import (
    SceneEvidence,
    build_prompt,
    depth_layout,
    draw_regions,
    image_statistics,
    region_evidence,
)
from bolero.records import save
from bolero.vlm import VLMClient
from bolero.vlm_scoring import (
    damaged_mask,
    overlap_fraction,
    parse_vlm_answer,
    score_case,
)

VLM_INSTRUCTION = (
    "You are given an image that may have a damaged or occluded region, plus optional\n"
    "structured evidence. Identify the main entity, describe what the damaged region\n"
    "most likely contains, and state its spatial relation to the entity. Do not invent\n"
    "color or texture. Answer as one JSON object with keys: entity, damaged_region_content,\n"
    "relation (one of overlaps_entity, adjacent, background, unclear), confidence (0-1),\n"
    "rationale."
)

CONFIGS = (
    "none",
    "text",
    "overlay",
    "overlay+text",
    "overlay+depth",
    "minus_candidates",
    "minus_depth",
    "minus_regions",
    "with_detector",
)
CLEAN_CONFIGS = ("none", "text")
TEXT_CONFIGS = {
    "text",
    "overlay+text",
    "minus_candidates",
    "minus_depth",
    "minus_regions",
    "with_detector",
}
OVERLAY_CONFIGS = {"overlay", "overlay+text", "overlay+depth"}

ROW_FIELDS = [
    "image_id",
    "class_name",
    "wnid",
    "true_class",
    "clean_prediction",
    "degraded_prediction",
    "perturbation",
    "level",
    "config",
    "entity_box",
    "damaged_box",
    "damage_overlap",
    "parse_ok",
    "entity",
    "damaged_region_content",
    "relation_answer",
    "confidence",
    "rationale",
    "entity_hit",
    "damage_eligible",
    "damage_hit",
    "relation_geometric",
    "relation_hit",
    "relation_hybrid",
    "relation_hybrid_source",
    "utility",
    "degraded_correct",
    "latency_ms",
    "cached",
    "error",
    "text",
]

SUMMARY_FIELDS = [
    "config",
    "n",
    "parse_rate",
    "entity_hit",
    "damage_n",
    "damage_hit",
    "relation_n",
    "relation_hit",
    "relation_hybrid_coverage",
    "utility",
    "degraded_correct",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cached_rate",
]

PER_CONDITION_FIELDS = [
    "config",
    "perturbation",
    "n",
    "entity_hit",
    "damage_hit",
    "relation_hit",
    "utility",
]


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def validate_configs(names: Sequence[str]) -> None:
    """Raise ValueError for unknown evidence configurations."""
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise ValueError(
            f"unknown config(s) {unknown}; choose from {list(CONFIGS)}"
        )


def cache_key(image_id: str, name: str, level: int, config: str) -> str:
    """Stable VLM cache key for one (image, condition, level, config) case."""
    return f"{image_id}|{name}|{level}|{config}"


def depth_image(depth: np.ndarray) -> np.ndarray:
    """Render a depth map as a 3-channel float32 RGB image in [0, 1]."""
    depth = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    return np.repeat(depth[..., None], 3, axis=-1).astype(np.float32)


def _evidence_variant(
    config: str,
    evidence: SceneEvidence,
    image: np.ndarray,
    detector: Any | None,
) -> SceneEvidence:
    """Apply the evidence ablation or detector enrichment for `config`."""
    if config == "minus_candidates":
        return replace(evidence, candidates=[])
    if config == "minus_depth":
        return replace(evidence, depth_layout={})
    if config == "minus_regions":
        return replace(evidence, entity=None, damaged=None)
    if config == "with_detector":
        if detector is None:
            raise ValueError("config 'with_detector' requires a detector")
        decision = detector.decide(image)
        return replace(
            evidence,
            perturbation=str(decision.label),
            perturbation_prob=float(decision.probability),
        )
    return evidence


def build_config(
    config: str,
    image: np.ndarray,
    evidence: SceneEvidence,
    depth: np.ndarray | None = None,
    detector: Any | None = None,
) -> tuple[list[np.ndarray], str]:
    """Build the image list and prompt for one evidence configuration."""
    validate_configs([config])
    variant = (
        _evidence_variant(config, evidence, image, detector)
        if config in TEXT_CONFIGS
        else evidence
    )

    base = image
    if config in OVERLAY_CONFIGS:
        base = draw_regions(image, variant.entity, variant.damaged)

    if config == "none":
        return [image], VLM_INSTRUCTION
    if config in OVERLAY_CONFIGS and config != "overlay+text":
        images = [base]
        if config == "overlay+depth":
            if depth is None:
                depth = np.zeros(image.shape[:2], dtype=np.float32)
            images.append(depth_image(depth))
        return images, VLM_INSTRUCTION
    if config in ("text", "overlay+text"):
        return [base], build_prompt(variant, instruction=VLM_INSTRUCTION)
    # Remaining configs are text ablations of the degraded image.
    return [image], build_prompt(variant, instruction=VLM_INSTRUCTION)


# ---------------------------------------------------------------------------
# Classifier adapter
# ---------------------------------------------------------------------------


class ClassifierAdapter:
    """Frozen ImageNet-1k classifier with single-image convenience methods."""

    def __init__(self, classifier: FrozenClassifier) -> None:
        self.classifier = classifier

    def _topk(self, image: np.ndarray) -> list[tuple[str, float]]:
        preds = self.classifier.predict([image], np.zeros(1, dtype=int))
        return self.classifier.candidates(preds, 0)

    def predict_label(self, image: np.ndarray) -> str:
        return self._topk(image)[0][0]

    def predict_candidates(self, image: np.ndarray) -> list[tuple[str, float]]:
        return self._topk(image)


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------


def select_records(
    records: Sequence[ImageRecord],
    sample_ids: Sequence[str] = (),
    limit: int = 0,
    universe: Sequence[ImageRecord] | None = None,
) -> tuple[list[ImageRecord], list[dict]]:
    """Forced sample ids first, then the sampled records, deduplicated and limited."""
    known = {record.image_id: record for record in (universe or records)}
    selected: list[ImageRecord] = []
    seen: set[str] = set()
    skipped: list[dict] = []

    for image_id in sample_ids or ():
        record = known.get(str(image_id))
        if record is None:
            skipped.append({"image_id": str(image_id), "reason": "not found in val split"})
            continue
        if record.image_id not in seen:
            seen.add(record.image_id)
            selected.append(record)

    for record in records:
        if record.image_id not in seen:
            seen.add(record.image_id)
            selected.append(record)

    if limit and limit > 0:
        selected = selected[: int(limit)]
    return selected, skipped


def resolve_records(
    args: Any,
    records: Sequence[ImageRecord] | None = None,
    universe: Sequence[ImageRecord] | None = None,
) -> tuple[list[ImageRecord], list[dict]]:
    """Load the val split (unless injected) and apply --sample and --limit."""
    if records is None:
        records = list_images(
            "val", root=args.data_root, per_class=args.per_class, seed=args.seed
        )
    if universe is None:
        sample_ids = list(getattr(args, "sample", []) or [])
        universe = (
            list_images("val", root=args.data_root, per_class=None, seed=args.seed)
            if sample_ids
            else records
        )
    return select_records(
        records,
        getattr(args, "sample", []) or [],
        getattr(args, "limit", 0) or 0,
        universe,
    )


# ---------------------------------------------------------------------------
# One VLM case
# ---------------------------------------------------------------------------


def _box_text(box) -> str | None:
    if box is None:
        return None
    return ",".join(str(int(value)) for value in box)


def ask_case(
    *,
    record: ImageRecord,
    true_class: str,
    clean_prediction: str,
    degraded_prediction: str,
    image: np.ndarray,
    evidence: SceneEvidence,
    entity_box,
    damaged_box,
    name: str,
    level: int,
    config: str,
    client: Any,
    depth: np.ndarray | None,
    detector: Any | None,
) -> dict:
    """Ask the VLM one case and score the answer against class and geometry."""
    images, prompt = build_config(
        config, image, evidence, depth=depth, detector=detector
    )
    error = None
    try:
        response = client.ask(
            images, prompt, cache_key=cache_key(record.image_id, name, level, config)
        )
        text = response.text
        latency_ms = float(getattr(response, "latency_ms", 0.0))
        cached = bool(getattr(response, "cached", False))
    except Exception as exc:  # one failed case must not end the run
        text = ""
        latency_ms = 0.0
        cached = False
        error = str(exc)
    answer = parse_vlm_answer(text)
    parse_ok = answer is not None
    if answer is None:
        answer = {
            "entity": "",
            "damaged_region_content": "",
            "relation": "unclear",
            "confidence": None,
            "rationale": "",
        }

    scored = score_case(
        answer,
        true_class=true_class,
        clean_prediction=clean_prediction,
        degraded_prediction=degraded_prediction,
        entity_box=entity_box,
        damaged_box=damaged_box,
        shape=image.shape[:2],
    )

    return {
        "image_id": record.image_id,
        "class_name": record.class_name,
        "wnid": record.wnid,
        "true_class": true_class,
        "clean_prediction": clean_prediction,
        "degraded_prediction": degraded_prediction,
        "perturbation": name,
        "level": int(level),
        "config": config,
        "entity_box": _box_text(entity_box),
        "damaged_box": _box_text(damaged_box),
        "damage_overlap": round(float(overlap_fraction(damaged_box, entity_box)), 4),
        "parse_ok": bool(parse_ok),
        "entity": answer["entity"],
        "damaged_region_content": answer["damaged_region_content"],
        "relation_answer": answer["relation"],
        "confidence": answer["confidence"],
        "rationale": answer["rationale"],
        **scored,
        "latency_ms": latency_ms,
        "cached": cached,
        "error": error,
        "text": text,
    }


def evaluate_record(
    record: ImageRecord,
    clean: np.ndarray,
    depth: np.ndarray,
    args: Any,
    classifier: Any,
    client: Any,
    detector: Any | None = None,
) -> list[dict]:
    """All rows for one image: degraded conditions plus optional clean controls."""
    height, width = clean.shape[:2]
    entity = region_evidence(near_mask(depth, 0.4), depth)
    clean_candidates = list(classifier.predict_candidates(clean))
    clean_prediction = clean_candidates[0][0] if clean_candidates else ""
    true_class = record.class_name
    rows: list[dict] = []

    for name in args.conditions:
        perturbation = ALL_PERTURBATIONS[name]
        for level in args.levels:
            level = int(level)
            degraded = perturbation(
                clean,
                level,
                depth if perturbation.needs_depth else None,
                rng_for(record.image_id, name, level, args.seed),
            )
            mask = damaged_mask(name, degraded, clean, depth)
            damaged = (
                region_evidence(mask, depth)
                if mask is not None and bool(mask.any())
                else None
            )
            evidence = SceneEvidence(
                (width, height),
                list(classifier.predict_candidates(degraded)),
                entity=entity,
                damaged=damaged,
                depth_layout=depth_layout(depth),
                stats=image_statistics(degraded),
            )
            degraded_prediction = evidence.candidates[0][0] if evidence.candidates else ""
            for config in args.configs:
                rows.append(
                    ask_case(
                        record=record,
                        true_class=true_class,
                        clean_prediction=clean_prediction,
                        degraded_prediction=degraded_prediction,
                        image=degraded,
                        evidence=evidence,
                        entity_box=entity.box,
                        damaged_box=damaged.box if damaged is not None else None,
                        name=name,
                        level=level,
                        config=config,
                        client=client,
                        depth=depth,
                        detector=detector,
                    )
                )

    if getattr(args, "include_clean", False):
        evidence = SceneEvidence(
            (width, height),
            list(clean_candidates),
            entity=entity,
            damaged=None,
            depth_layout=depth_layout(depth),
            stats=image_statistics(clean),
        )
        for config in CLEAN_CONFIGS:
            rows.append(
                ask_case(
                    record=record,
                    true_class=true_class,
                    clean_prediction=clean_prediction,
                    degraded_prediction=clean_prediction,
                    image=clean,
                    evidence=evidence,
                    entity_box=entity.box,
                    damaged_box=None,
                    name="clean",
                    level=0,
                    config=config,
                    client=client,
                    depth=depth,
                    detector=detector,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def _mean(values: Iterable[Any]) -> float:
    kept = [
        value
        for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    ]
    return float(np.mean(kept)) if kept else float("nan")


def _bools(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _config_order(df: pd.DataFrame) -> list[str]:
    present = set(df["config"])
    return [config for config in CONFIGS if config in present]


def summary_table(rows: Sequence[dict]) -> list[dict]:
    """Per-config metrics: parse/entity/damage/relation rates and latency."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summary: list[dict] = []
    for config in _config_order(df):
        sub = df[df["config"] == config]
        eligible = sub[_bools(sub["damage_eligible"])]
        relations = sub[sub["relation_hit"].notna()]
        live = sub[~_bools(sub["cached"])]
        latencies = pd.to_numeric(live["latency_ms"], errors="coerce").to_numpy(dtype=float)
        summary.append(
            {
                "config": config,
                "n": int(len(sub)),
                "parse_rate": float(_bools(sub["parse_ok"]).mean()),
                "entity_hit": float(_bools(sub["entity_hit"]).mean()),
                "damage_n": int(len(eligible)),
                "damage_hit": _mean(eligible["damage_hit"]) if len(eligible) else float("nan"),
                "relation_n": int(len(relations)),
                "relation_hit": _mean(relations["relation_hit"]) if len(relations) else float("nan"),
                "relation_hybrid_coverage": float(
                    sub["relation_hybrid"].fillna("unclear").ne("unclear").mean()
                ),
                "utility": float(_bools(sub["utility"]).mean()),
                "degraded_correct": float(_bools(sub["degraded_correct"]).mean()),
                "error_rate": float(
                    (sub["error"] if "error" in sub else pd.Series([None] * len(sub)))
                    .notna()
                    .mean()
                ),
                "latency_p50_ms": (
                    float(np.percentile(latencies, 50)) if len(latencies) else float("nan")
                ),
                "latency_p95_ms": (
                    float(np.percentile(latencies, 95)) if len(latencies) else float("nan")
                ),
                "cached_rate": float(_bools(sub["cached"]).mean()),
            }
        )
    return summary


def per_condition_table(rows: Sequence[dict]) -> list[dict]:
    """Per-config x perturbation metrics."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    perturbations = list(dict.fromkeys(df["perturbation"]))
    table: list[dict] = []
    for config in _config_order(df):
        for perturbation in perturbations:
            sub = df[(df["config"] == config) & (df["perturbation"] == perturbation)]
            if sub.empty:
                continue
            eligible = sub[_bools(sub["damage_eligible"])]
            relations = sub[sub["relation_hit"].notna()]
            table.append(
                {
                    "config": config,
                    "perturbation": perturbation,
                    "n": int(len(sub)),
                    "entity_hit": float(_bools(sub["entity_hit"]).mean()),
                    "damage_hit": _mean(eligible["damage_hit"]) if len(eligible) else float("nan"),
                    "relation_hit": _mean(relations["relation_hit"]) if len(relations) else float("nan"),
                    "utility": float(_bools(sub["utility"]).mean()),
                }
            )
    return table


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return _jsonable(vars(value))
    except TypeError:
        return str(value)


def _print_table(summary: Sequence[dict]) -> None:
    if not summary:
        print("no rows")
        return
    header = (
        f"{'config':<18} {'n':>5} {'parse':>7} {'entity':>7} {'dmg_n':>6} "
        f"{'dmg':>7} {'rel_n':>6} {'rel':>7} {'hybrid':>7} {'util':>7} {'p50':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{str(row['config']):<18} {int(row['n']):>5} "
            f"{float(row['parse_rate']):>7.3f} {float(row['entity_hit']):>7.3f} "
            f"{int(row['damage_n']):>6} {float(row['damage_hit']):>7.3f} "
            f"{int(row['relation_n']):>6} {float(row['relation_hit']):>7.3f} "
            f"{float(row['relation_hybrid_coverage']):>7.3f} {float(row['utility']):>7.3f} "
            f"{float(row['latency_p50_ms']):>8.1f}"
        )


def _write_outputs(
    args: Any,
    rows: list[dict],
    n_images: int,
    skipped: Sequence[dict],
    started: datetime,
    wall_seconds: float,
) -> dict:
    out = Path(getattr(args, "out", RESULTS_DIR / "vlm"))
    out.mkdir(parents=True, exist_ok=True)

    summary = summary_table(rows)
    per_condition = per_condition_table(rows)

    save(pd.DataFrame(summary, columns=SUMMARY_FIELDS), out / "summary.csv")
    save(pd.DataFrame(per_condition, columns=PER_CONDITION_FIELDS), out / "per_condition.csv")
    save(pd.DataFrame(rows, columns=ROW_FIELDS), out / "predictions.csv.gz")

    live = [row for row in rows if not row.get("cached")]
    generation_seconds = float(
        sum(float(row.get("latency_ms") or 0.0) for row in live) / 1000.0
    )
    config = _jsonable(vars(args))

    finished = datetime.now(timezone.utc)
    headline = {
        "config": config,
        "vlm": getattr(args, "vlm_model", None),
        "n_images": int(n_images),
        "n_rows": len(rows),
        "summary": summary,
        "skipped": list(skipped),
        "generation_seconds": generation_seconds,
        "generation_ms": generation_seconds * 1000.0,
        "generated_calls": len(live),
        "cached_rate": (
            float(np.mean([bool(row.get("cached")) for row in rows])) if rows else 0.0
        ),
    }
    (out / "headline.json").write_text(json.dumps(_jsonable(headline), indent=2))

    run_info = {
        "config": config,
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "wall_seconds": float(wall_seconds),
    }
    (out / "run.json").write_text(json.dumps(_jsonable(run_info), indent=2))

    _print_table(summary)
    return {
        "rows": rows,
        "summary": summary,
        "per_condition": per_condition,
        "n_images": int(n_images),
        "n_rows": len(rows),
        "skipped": list(skipped),
    }


# ---------------------------------------------------------------------------
# Harness core
# ---------------------------------------------------------------------------


def run(
    args: Any,
    records: Sequence[ImageRecord],
    *,
    classifier: Any,
    client: Any,
    depth_for: Callable[[ImageRecord, np.ndarray], np.ndarray] | None = None,
    detector: Any | None = None,
    skipped: Sequence[dict] | None = None,
) -> dict:
    """Evaluate the VLM on `records` with injected classifier and client."""
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()

    cache = None if depth_for is not None else DepthCache(root=Path(args.depth_root))

    rows: list[dict] = []
    n_images = 0
    for record in records:
        clean = load_image(record.path)
        depth = (
            depth_for(record, clean)
            if depth_for is not None
            else cache.get(record, clean)
        )
        rows.extend(
            evaluate_record(
                record, clean, depth, args, classifier, client, detector=detector
            )
        )
        n_images += 1

    return _write_outputs(
        args,
        rows,
        n_images,
        list(skipped or []),
        started,
        time.perf_counter() - start_time,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument(
        "--data-root", type=Path, default=DATA_DIR / "imagenette2-320"
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["region_removal", "foreground_occlusion", "boundary_erase", "background_removal"],
    )
    parser.add_argument("--levels", nargs="*", type=int, default=[2])
    parser.add_argument(
        "--configs", nargs="*", default=["none", "text", "overlay", "overlay+text"]
    )
    parser.add_argument(
        "--include-clean", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--cache-dir", type=Path, default=RESULTS_DIR / "vlm" / "cache"
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vlm")
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("results/deciders/detector320/detector.joblib"),
    )
    parser.add_argument("--depth-root", type=Path, default=CACHE_DIR / "depth-320")
    parser.add_argument("--classifier", default="resnet50")

    args = parser.parse_args(argv)
    args.configs = [
        piece.strip()
        for value in args.configs
        for piece in str(value).split(",")
        if piece.strip()
    ]
    args.conditions = [
        piece.strip()
        for value in args.conditions
        for piece in str(value).split(",")
        if piece.strip()
    ]
    args.sample = [
        piece.strip()
        for value in args.sample
        for piece in str(value).split(",")
        if piece.strip()
    ]
    try:
        validate_configs(args.configs)
    except ValueError as exc:
        parser.error(str(exc))
    bad_conditions = [name for name in args.conditions if name not in ALL_PERTURBATIONS]
    if bad_conditions:
        parser.error(
            f"unknown condition(s) {bad_conditions}; choose from {list(ALL_PERTURBATIONS)}"
        )
    bad_levels = [level for level in args.levels if level not in (1, 2, 3)]
    if bad_levels:
        parser.error(f"levels must be 1, 2, or 3; got {bad_levels}")
    return args


def _resolve_detector_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    load_env()

    records, skipped = resolve_records(args)
    for entry in skipped:
        print(f"skipping {entry['image_id']}: {entry['reason']}", flush=True)

    detector = None
    if "with_detector" in args.configs:
        path = _resolve_detector_path(Path(args.detector))
        if not path.exists():
            raise SystemExit(f"{path} not found. Run scripts/train_detector.py first.")
        detector = PerturbationDetector.load(path)

    classifier = ClassifierAdapter(FrozenClassifier(args.classifier))
    client = VLMClient(
        model=args.vlm_model,
        cache_dir=args.cache_dir,
        max_new_tokens=args.max_new_tokens,
    )
    result = run(
        args,
        records,
        classifier=classifier,
        client=client,
        detector=detector,
        skipped=skipped,
    )
    print(f"wrote {result['n_rows']} rows for {result['n_images']} images to {args.out}")


if __name__ == "__main__":
    main()
