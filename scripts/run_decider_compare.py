"""Stage 4 - decider comparison: ask every router the same three questions.

Samples Imagenette val images, degrades them with `conditions.iter_samples`,
builds `SceneEvidence` from image statistics (plus optional detector output and
classifier candidates), and asks one or more deciders whether the view is
degraded, which condition it is, and how severe. Writes routing metrics for the
report and the per-image scores the conformal calibration step consumes.

Outputs (in --out):
  summary.csv          per-decider routing table (routing_metrics)
  per_condition.csv    per-decider x true condition table
  scores.csv           conformal handoff schema (score_rows)
  predictions.csv.gz   one row per sample x decider
  headline.json        config, headline numbers, per-decider summary, skips
  run.json             config plus timestamps and wall time

This is an offline-friendly harness: tests inject fake deciders and samples, and
nothing here makes a network call unless a hosted decider is selected and asked.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import requests

import _bootstrap  # noqa: F401
from bolero.classifier import FrozenClassifier
from bolero.conditions import TAXONOMIES, ConditionSample, condition_label, iter_samples
from bolero.config import (
    CACHE_DIR,
    DECISION_MODELS,
    JEV_MODEL,
    RESULTS_DIR,
    SEED,
    load_env,
)
from bolero.data import list_images
from bolero.deciders import (
    ConstantDecider,
    DeciderOutput,
    HFTextDecider,
    JevDecider,
    RFDecider,
    SeverityClassifier,
    per_condition_metrics,
    routing_metrics,
    score_rows,
)
from bolero.depth import DepthCache
from bolero.detection import PerturbationDetector, extract_features
from bolero.openrouter import BudgetExceeded, CostGuard, DecisionsClient
from bolero.perturbations import ALL_PERTURBATIONS, ANY_PERTURBATION, LEVELS
from bolero.prompting import SceneEvidence, image_statistics
from bolero.records import save

DEFAULT_DECIDERS = ["majority", "rf", "jev"]
EVIDENCE_MODES = ("stats", "stats+image", "image")
HOSTED_ALIASES = set(DECISION_MODELS)

ROW_FIELDS = [
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
]

SUMMARY_FIELDS = [
    "decider",
    "n",
    "degraded_accuracy",
    "degraded_recall",
    "clean_false_alarm_rate",
    "condition_accuracy",
    "condition_macro_f1",
    "severity_exact",
    "severity_within1",
    "severity_mae",
    "parse_failure_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cached_rate",
    "total_cost",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def is_hosted(decider: Any) -> bool:
    return bool(getattr(decider, "hosted", False))


class _CallCounter:
    """Thread-safe count of attempted hosted decision calls."""

    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def add(self) -> None:
        with self._lock:
            self.value += 1


class ThreadLocalClient:
    """One DecisionsClient per worker thread, all sharing one CostGuard budget."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()

    def _client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._factory()
            self._local.client = client
        return client

    def ask(self, model: str, state: Any, questions: dict, **kwargs: Any) -> Any:
        return self._client().ask(model, state, questions, **kwargs)


def depth_version(root: Path | None) -> str:
    """Cache version for a data root, so 320 depth maps never mix with 160."""
    if root is None:
        return os.environ.get("IMAGENETTE_VERSION", "160")
    name = Path(root).name
    if name.startswith("imagenette2-"):
        version = name.split("-")[-1]
        if version:
            return version
    return "160"


def majority_condition(taxonomy: str) -> str:
    """The most frequent degraded condition in the perturbation suite."""
    counts: Counter[str] = Counter()
    for name, perturbation in ALL_PERTURBATIONS.items():
        counts[condition_label(name, perturbation.family, taxonomy)] += 1
    return counts.most_common(1)[0][0]


def build_depth_cache(root: Path | None) -> DepthCache:
    return DepthCache(root=CACHE_DIR / f"depth-{depth_version(root)}")


def load_conformal(path: Path | None) -> tuple[dict[str, float], float | None]:
    """Parse a conformal JSON file: per-decider thresholds plus an optional global."""
    if path is None:
        return {}, None
    data = json.loads(Path(path).read_text())
    default = float(data["threshold"]) if "threshold" in data else None
    per_decider = {
        str(name): float(value) for name, value in data.items() if name != "threshold"
    }
    return per_decider, default


def build_row(
    decider_name: str,
    sample: ConditionSample,
    result: DeciderOutput,
    threshold: float | None = None,
) -> dict | None:
    """One prediction row; applies the conformal override when a threshold is set."""
    pred_degraded = result.degraded
    if threshold is not None:
        if result.p_degraded is None:
            return None
        pred_degraded = float(result.p_degraded) >= float(threshold)
    return {
        "decider": decider_name,
        "image_id": sample.image_id,
        "class_name": sample.class_name,
        "true_condition": sample.condition,
        "true_degraded": not sample.clean,
        "true_severity": sample.severity,
        "pred_condition": result.condition,
        "pred_degraded": pred_degraded,
        "pred_severity": result.severity,
        "p_degraded": result.p_degraded,
        "confidence": result.confidence,
        "parse_ok": result.parse_ok,
        "latency_ms": result.latency_ms,
        "cost": result.cost,
        "cached": result.cached,
        "error": result.error,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Decider roster
# ---------------------------------------------------------------------------


def build_client_proxy(args: Any, guard: CostGuard) -> ThreadLocalClient:
    cache_dir = Path(getattr(args, "out", RESULTS_DIR / "deciders")) / "cache"

    def factory() -> DecisionsClient:
        return DecisionsClient(
            session=requests.Session(), guard=guard, cache_dir=cache_dir
        )

    return ThreadLocalClient(factory)


def build_deciders(
    args: Any,
    aliases: Sequence[str] | None = None,
    exemplars: Any | None = None,
) -> tuple[list[Any], list[dict]]:
    """Resolve CLI aliases into decider instances, recording unavailable ones."""
    raw_aliases = list(aliases if aliases is not None else getattr(args, "deciders", DEFAULT_DECIDERS))
    aliases = [
        piece.strip()
        for alias in raw_aliases
        for piece in str(alias).split(",")
        if piece.strip()
    ]
    taxonomy = getattr(args, "taxonomy", "family")
    degraded_label = majority_condition(taxonomy)
    evidence_mode = getattr(args, "evidence_mode", "stats")
    include_detector = bool(getattr(args, "include_detector", False))

    deciders: list[Any] = []
    skipped: list[dict] = []
    guard = CostGuard(getattr(args, "max_cost", 2.0))
    proxy: ThreadLocalClient | None = None

    for alias in aliases:
        if alias == "majority":
            decider = ConstantDecider(
                name="majority", degraded=True, condition=degraded_label, severity=2
            )
        elif alias == "always_clean":
            decider = ConstantDecider(name="always_clean", degraded=False, condition="clean")
        elif alias == "always_degraded":
            decider = ConstantDecider(
                name="always_degraded",
                degraded=True,
                condition=degraded_label,
                severity=2,
            )
        elif alias == "rf":
            detector_path = Path(
                getattr(args, "rf_detector", RESULTS_DIR / "detector" / "detector.joblib")
            )
            severity_path = Path(
                getattr(args, "severity_model", RESULTS_DIR / "deciders" / "severity.joblib")
            )
            if not detector_path.exists():
                skipped.append(
                    {"decider": alias, "reason": f"detector not found: {detector_path}"}
                )
                continue
            kwargs: dict[str, Any] = {"detector_path": detector_path}
            if severity_path.exists():
                kwargs["severity_path"] = severity_path
            try:
                decider = RFDecider(name="rf", taxonomy=taxonomy, **kwargs)
            except Exception as exc:
                skipped.append({"decider": alias, "reason": f"could not load detector: {exc}"})
                continue
        elif alias == "hf":
            decider = HFTextDecider(
                name="hf",
                model=getattr(args, "hf_model", "Qwen/Qwen2.5-1.5B-Instruct"),
                taxonomy=taxonomy,
                include_detector=include_detector,
            )
        elif alias == "jev" or alias in HOSTED_ALIASES:
            if not os.environ.get("OPENROUTER_API_KEY"):
                skipped.append(
                    {"decider": alias, "reason": "OPENROUTER_API_KEY not set"}
                )
                continue
            if proxy is None:
                proxy = build_client_proxy(args, guard)
            model = JEV_MODEL if alias == "jev" else DECISION_MODELS[alias]
            decider = JevDecider(
                name=alias,
                client=proxy,
                model=model,
                taxonomy=taxonomy,
                evidence_mode=evidence_mode,
                include_detector=include_detector,
                exemplars=exemplars,
            )
            decider.hosted = True
        else:
            skipped.append({"decider": alias, "reason": "unknown decider alias"})
            continue

        available, reason = decider.available()
        if not available:
            skipped.append({"decider": alias, "reason": reason or "unavailable"})
            continue
        deciders.append(decider)

    return deciders, skipped


def describe_decider(decider: Any) -> dict:
    info: dict[str, Any] = {
        "name": decider.name,
        "type": type(decider).__name__,
        "hosted": is_hosted(decider),
    }
    for attribute in ("model", "model_name", "taxonomy", "evidence_mode"):
        if hasattr(decider, attribute):
            value = getattr(decider, attribute)
            if isinstance(value, PurePath):
                value = str(value)
            info[attribute] = value
    return info


# ---------------------------------------------------------------------------
# Evidence and exemplars
# ---------------------------------------------------------------------------


def build_evidence(
    sample: ConditionSample,
    args: Any,
    detector: PerturbationDetector | None = None,
    classifier: FrozenClassifier | None = None,
) -> SceneEvidence:
    stats = image_statistics(sample.image)
    height, width = sample.image.shape[:2]
    candidates: list[tuple[str, float]] = []
    perturbation: str | None = None
    perturbation_prob: float | None = None

    if bool(getattr(args, "include_candidates", False)) and classifier is not None:
        preds = classifier.predict([sample.image], np.zeros(1, dtype=int))
        candidates = classifier.candidates(preds, 0)

    if bool(getattr(args, "include_detector", False)) and detector is not None:
        decision = detector.decide(sample.image)
        perturbation = decision.label
        perturbation_prob = decision.probability

    return SceneEvidence(
        (width, height),
        candidates,
        perturbation,
        perturbation_prob,
        stats=stats,
    )


def build_exemplars(
    args: Any,
    n: int | None = None,
    depth_cache: DepthCache | None = None,
) -> list[dict]:
    """One train-split example per condition, balanced and never from val."""
    n = int(n if n is not None else getattr(args, "exemplars", 0))
    if n <= 0:
        return []
    cache = depth_cache or build_depth_cache(getattr(args, "data_root", None))
    records = list_images(
        "train",
        root=getattr(args, "data_root", None),
        per_class=2,
        seed=getattr(args, "seed", SEED),
    )
    samples = iter_samples(
        records,
        getattr(args, "taxonomy", "family"),
        getattr(args, "perturbations", list(ALL_PERTURBATIONS)),
        [2],
        max_per_bucket=1,
        depth_for=cache.get,
    )

    by_condition: dict[str, dict] = {}
    for sample in samples:
        if sample.condition in by_condition:
            continue
        stats = image_statistics(sample.image)
        height, width = sample.image.shape[:2]
        by_condition[sample.condition] = {
            "evidence": {
                "image_statistics": stats,
                "image": {"width": int(width), "height": int(height)},
            },
            "answer": {
                "condition": sample.condition,
                "severity": sample.severity,
            },
        }
        if len(by_condition) >= n:
            break

    ordered = list(by_condition.values())
    if not ordered:
        return []
    return [ordered[index % len(ordered)] for index in range(n)]


# ---------------------------------------------------------------------------
# Severity fit
# ---------------------------------------------------------------------------


def fit_severity_model(args: Any, depth_cache: DepthCache | None = None) -> Path:
    """Fit the severity random forest on the train split and save it."""
    cache = depth_cache or build_depth_cache(getattr(args, "data_root", None))
    records = list_images(
        "train",
        root=getattr(args, "data_root", None),
        per_class=3,
        seed=getattr(args, "seed", SEED),
    )
    features: list[np.ndarray] = []
    levels: list[int] = []
    for sample in iter_samples(
        records,
        getattr(args, "taxonomy", "family"),
        None,
        LEVELS,
        max_per_bucket=int(getattr(args, "fit_severity_per_class", 10)),
        depth_for=cache.get,
    ):
        if sample.clean:
            continue
        features.append(extract_features(sample.image))
        levels.append(int(sample.level))
    if not features:
        raise RuntimeError("no train samples available to fit the severity model")

    model = SeverityClassifier().fit(np.asarray(features), np.asarray(levels))
    path = Path(getattr(args, "severity_model", RESULTS_DIR / "deciders" / "severity.joblib"))
    model.save(path)
    return path


# ---------------------------------------------------------------------------
# Harness core
# ---------------------------------------------------------------------------


def _error_output(exc: Exception) -> DeciderOutput:
    return DeciderOutput(
        None, None, None, None, None, {}, False, 0.0, 0.0, False, None, error=str(exc)
    )


def _process_sample(
    sample: ConditionSample,
    deciders: Sequence[Any],
    args: Any,
    detector: PerturbationDetector | None,
    classifier: FrozenClassifier | None,
    counter: _CallCounter,
    thresholds: dict[str, float],
    default_threshold: float | None,
) -> list[dict]:
    evidence = build_evidence(sample, args, detector, classifier)
    rows: list[dict] = []
    for decider in deciders:
        threshold = thresholds.get(decider.name, default_threshold)
        try:
            result = decider.decide(sample, evidence)
        except (BudgetExceeded, KeyboardInterrupt):
            raise
        except Exception as exc:  # defensive: one bad decider must not kill the run
            result = _error_output(exc)
        # Count only calls that actually reached the network; cache hits are free.
        if is_hosted(decider) and not result.cached:
            counter.add()
        row = build_row(decider.name, sample, result, threshold)
        if row is not None:
            rows.append(row)
    return rows


def _execute(
    args: Any,
    deciders: Sequence[Any],
    samples: Iterable[ConditionSample],
    detector: PerturbationDetector | None,
    classifier: FrozenClassifier | None,
    counter: _CallCounter,
    thresholds: dict[str, float],
    default_threshold: float | None,
) -> tuple[list[dict], int, str | None]:
    workers = 1 if bool(getattr(args, "include_candidates", False)) else max(
        1, int(getattr(args, "workers", 4))
    )
    # A local HF model is one shared MPS object; do not call it from several threads.
    if any(isinstance(decider, HFTextDecider) for decider in deciders):
        workers = 1
    max_calls = int(getattr(args, "max_calls", 5000) or 0)
    limit = int(getattr(args, "limit", 0) or 0)
    calls_per_sample = sum(1 for decider in deciders if is_hosted(decider))

    rows: list[dict] = []
    n_samples = 0
    reserved = 0
    stopped: str | None = None
    iterator = iter(samples)

    def process(sample: ConditionSample) -> list[dict]:
        return _process_sample(
            sample,
            deciders,
            args,
            detector,
            classifier,
            counter,
            thresholds,
            default_threshold,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight: dict[Any, ConditionSample] = {}

        def submit_next() -> bool:
            nonlocal reserved, n_samples, stopped
            if stopped is not None:
                return False
            if limit and n_samples >= limit:
                return False
            if max_calls and reserved + calls_per_sample > max_calls:
                stopped = "max_calls"
                return False
            try:
                sample = next(iterator)
            except StopIteration:
                return False
            reserved += calls_per_sample
            n_samples += 1
            inflight[pool.submit(process, sample)] = sample
            if n_samples % 100 == 0:
                print(
                    f"JCODE_PROGRESS {json.dumps({'current': n_samples, 'message': f'{len(rows)} rows'})}",
                    flush=True,
                )
            return True

        for _ in range(workers):
            if not submit_next():
                break

        while inflight:
            try:
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                stopped = "interrupted"
                break
            for future in done:
                inflight.pop(future, None)
                try:
                    rows.extend(future.result())
                except BudgetExceeded as exc:
                    stopped = f"budget_exceeded: {exc}"
                except KeyboardInterrupt:
                    stopped = "interrupted"
                except Exception as exc:  # keep partial results on unexpected crashes
                    stopped = f"error: {exc}"
            if stopped is not None:
                break
            submit_next()
        for future in inflight:
            future.cancel()

    return rows, n_samples, stopped


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
    return str(value)


def _print_table(summary: Sequence[dict]) -> None:
    if not summary:
        print("no rows")
        return
    header = (
        f"{'decider':<18} {'n':>5} {'degraded_acc':>12} {'clean_fa':>9} "
        f"{'condition_acc':>13} {'severity_mae':>12} {'latency_p50':>11} {'cost':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{str(row['decider']):<18} {int(row['n']):>5} "
            f"{float(row['degraded_accuracy']):>12.3f} "
            f"{float(row['clean_false_alarm_rate']):>9.3f} "
            f"{float(row['condition_accuracy']):>13.3f} "
            f"{float(row['severity_mae']):>12.3f} "
            f"{float(row['latency_p50_ms']):>11.1f} "
            f"{float(row['total_cost']):>9.5f}"
        )


def _write_outputs(
    args: Any,
    rows: list[dict],
    n_samples: int,
    stopped: str | None,
    skipped: Sequence[dict],
    deciders: Sequence[Any],
    counter: _CallCounter,
    started: datetime,
    wall_seconds: float,
) -> dict:
    out = Path(getattr(args, "out", RESULTS_DIR / "deciders"))
    out.mkdir(parents=True, exist_ok=True)

    summary = routing_metrics(rows)
    per_condition = per_condition_metrics(rows)
    scores = score_rows(rows)

    save(pd.DataFrame(summary, columns=SUMMARY_FIELDS), out / "summary.csv")
    save(pd.DataFrame(per_condition), out / "per_condition.csv")
    save(pd.DataFrame(scores), out / "scores.csv")
    save(pd.DataFrame(rows, columns=ROW_FIELDS), out / "predictions.csv.gz")

    total_cost = float(sum(float(row.get("cost") or 0.0) for row in rows))
    cached_rate = float(np.mean([bool(row.get("cached")) for row in rows])) if rows else 0.0
    parse_failure_rate = (
        float(np.mean([not bool(row.get("parse_ok")) for row in rows])) if rows else 0.0
    )

    config = _jsonable(vars(args))
    config["deciders_resolved"] = [describe_decider(decider) for decider in deciders]

    finished = datetime.now(timezone.utc)
    headline = {
        "config": config,
        "n_rows": len(rows),
        "n_samples": n_samples,
        "total_cost": total_cost,
        "cached_rate": cached_rate,
        "parse_failure_rate": parse_failure_rate,
        "per_decider": summary,
        "skipped": list(skipped),
        "stopped": stopped,
        "hosted_calls": counter.value,
    }
    (out / "headline.json").write_text(json.dumps(_jsonable(headline), indent=2))

    run_info = {
        "config": config,
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "wall_seconds": wall_seconds,
        "stopped": stopped,
    }
    (out / "run.json").write_text(json.dumps(_jsonable(run_info), indent=2))

    _print_table(summary)
    return {
        "rows": rows,
        "summary": summary,
        "per_condition": per_condition,
        "scores": scores,
        "n_samples": n_samples,
        "hosted_calls": counter.value,
        "stopped": stopped,
        "skipped": list(skipped),
    }


def run(
    args: Any,
    deciders: Sequence[Any],
    samples: Iterable[ConditionSample],
    *,
    detector: PerturbationDetector | None = None,
    classifier: FrozenClassifier | None = None,
    skipped: Sequence[dict] | None = None,
) -> dict:
    """Run the comparison on already-built samples and deciders (test entry point)."""
    out = Path(getattr(args, "out", RESULTS_DIR / "deciders"))
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()

    thresholds, default_threshold = load_conformal(getattr(args, "conformal", None))
    counter = _CallCounter()
    skipped = list(skipped or [])

    try:
        rows, n_samples, stopped = _execute(
            args,
            deciders,
            samples,
            detector,
            classifier,
            counter,
            thresholds,
            default_threshold,
        )
    except BudgetExceeded as exc:
        rows, n_samples, stopped = [], 0, f"budget_exceeded: {exc}"
    except KeyboardInterrupt:
        rows, n_samples, stopped = [], 0, "interrupted"

    return _write_outputs(
        args,
        rows,
        n_samples,
        stopped,
        skipped,
        deciders,
        counter,
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
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--max-per-condition", type=int, default=20)
    parser.add_argument("--taxonomy", choices=sorted(TAXONOMIES), default="family")
    parser.add_argument(
        "--perturbations", nargs="*", default=list(ALL_PERTURBATIONS),
        choices=list(ANY_PERTURBATION),
    )
    parser.add_argument("--levels", nargs="*", type=int, default=list(LEVELS))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deciders", nargs="*", default=list(DEFAULT_DECIDERS))
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="stats+image",
                        help="stats = text features only; stats+image = features plus the image (default); image = image only")
    parser.add_argument("--include-detector", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--exemplars", type=int, choices=(0, 3, 6), default=0)
    parser.add_argument("--max-calls", type=int, default=5000)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fit-severity-per-class", type=int, default=10)
    parser.add_argument("--conformal", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "deciders")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--rf-detector", type=Path, default=RESULTS_DIR / "detector" / "detector.joblib"
    )
    parser.add_argument(
        "--severity-model", type=Path, default=RESULTS_DIR / "deciders" / "severity.joblib"
    )
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    load_env()
    if args.include_candidates:
        args.workers = 1

    depth_cache = build_depth_cache(args.data_root)

    if args.fit_severity_per_class > 0 and not args.severity_model.exists():
        print(f"fitting severity model on train split -> {args.severity_model}")
        fit_severity_model(args, depth_cache)

    detector = None
    if args.include_detector or "rf" in args.deciders:
        if args.rf_detector.exists():
            detector = PerturbationDetector.load(args.rf_detector)
        elif args.include_detector:
            raise SystemExit(
                f"{args.rf_detector} not found. Run scripts/train_detector.py first."
            )

    exemplars = build_exemplars(args, args.exemplars, depth_cache) if args.exemplars else None
    deciders, skipped = build_deciders(args, exemplars=exemplars)
    for entry in skipped:
        print(f"skipping {entry['decider']}: {entry['reason']}", flush=True)

    classifier = None
    if args.include_candidates:
        classifier = FrozenClassifier()

    records = list_images(
        "val", root=args.data_root, per_class=args.per_class, seed=args.seed
    )
    samples = iter_samples(
        records,
        args.taxonomy,
        args.perturbations,
        args.levels,
        max_per_bucket=args.max_per_condition,
        depth_for=depth_cache.get,
    )
    if args.limit:
        from itertools import islice

        samples = islice(samples, args.limit)

    result = run(
        args,
        deciders,
        samples,
        detector=detector,
        classifier=classifier,
        skipped=skipped,
    )
    if result["stopped"]:
        print(f"stopped early: {result['stopped']}")
    print(f"wrote {len(result['rows'])} rows to {args.out}")


if __name__ == "__main__":
    main()
