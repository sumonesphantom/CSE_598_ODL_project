"""Decision adapters and routing metrics for the decider comparison.

Every decider answers the same three questions from `conditions.py`: is the view
degraded, which condition label best describes it, and how severe is it. The
roster spans fixed baselines, a signature-detector adapter, the hosted Decisions
API models, and a local prompt-and-parse language model. All of them return a
`DeciderOutput`, so the benchmark can score them with one row schema.

The metrics helpers at the bottom turn those rows into routing tables for the
report and into per-image scores for the conformal calibration handoff.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .conditions import TAXONOMIES, ConditionSample, condition_label
from .config import SEED
from .detection import CLEAN, PerturbationDetector, extract_features
from .openrouter import DecisionResponse, OpenRouterError
from .perturbations import ALL_PERTURBATIONS
from .prompting import SceneEvidence, build_jev_questions, prepare_image


@dataclass
class DeciderOutput:
    degraded: bool | None
    condition: str | None
    severity: int | None
    confidence: float | None
    p_degraded: float | None          # P(degraded) for the conformal export
    probabilities: dict[str, float]   # condition label -> probability
    parse_ok: bool
    latency_ms: float
    cost: float
    cached: bool
    raw: dict | str | None
    error: str | None = None


@runtime_checkable
class Decider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def decide(self, sample: ConditionSample, evidence: SceneEvidence) -> DeciderOutput: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """Finite float from a JSON-style scalar, or None. Rejects bools."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _nested(answers: dict, key: str) -> dict:
    value = answers.get(key)
    return value if isinstance(value, dict) else {}


def _class_condition(cls: str, taxonomy: str) -> str:
    if cls == CLEAN:
        return "clean"
    perturbation = ALL_PERTURBATIONS.get(cls)
    if perturbation is not None:
        return condition_label(cls, perturbation.family, taxonomy)
    return cls


def _clean_probability(classes: Sequence[str], probabilities: np.ndarray | None) -> float | None:
    if probabilities is None:
        return None
    total = 0.0
    found = False
    for cls, prob in zip(classes, probabilities):
        if cls == CLEAN:
            total += float(prob)
            found = True
    return total if found else None


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class ConstantDecider:
    """Always return the same fixed answer; backs the trivial baselines."""

    def __init__(
        self,
        name: str,
        degraded: bool,
        condition: str,
        severity: int | None = None,
        confidence: float = 1.0,
    ) -> None:
        self.name = name
        self.degraded = degraded
        self.condition = condition
        self.severity = severity
        self.confidence = float(confidence)

    def available(self) -> tuple[bool, str]:
        return True, ""

    def decide(self, sample: ConditionSample, evidence: SceneEvidence) -> DeciderOutput:
        return DeciderOutput(
            degraded=self.degraded,
            condition=self.condition,
            severity=self.severity,
            confidence=self.confidence,
            p_degraded=1.0 if self.degraded else 0.0,
            probabilities={self.condition: self.confidence} if self.condition else {},
            parse_ok=True,
            latency_ms=0.0,
            cost=0.0,
            cached=False,
            raw=None,
        )


# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------


class SeverityClassifier:
    """Random forest over detector features predicting a 1..3 severity level."""

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, features: np.ndarray, levels: Sequence[int]) -> "SeverityClassifier":
        self.model.fit(np.asarray(features, dtype=np.float32), np.asarray(levels))
        return self

    def predict(self, features: np.ndarray) -> int:
        row = np.atleast_2d(np.asarray(features, dtype=np.float32))
        return int(self.model.predict(row)[0])

    def predict_proba(self, features: np.ndarray) -> dict[int, float]:
        row = np.atleast_2d(np.asarray(features, dtype=np.float32))
        probabilities = self.model.predict_proba(row)[0]
        return {int(cls): float(prob) for cls, prob in zip(self.model.classes_, probabilities)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "SeverityClassifier":
        classifier = cls()
        classifier.model = joblib.load(Path(path))
        return classifier


# ---------------------------------------------------------------------------
# Signature detector adapter
# ---------------------------------------------------------------------------


class RFDecider:
    """Map the hand-crafted perturbation detector (plus severity model) to answers."""

    def __init__(
        self,
        name: str = "rf",
        taxonomy: str = "family",
        detector: Any | None = None,
        severity_model: Any | None = None,
        detector_path: str | Path | None = None,
        severity_path: str | Path | None = None,
    ) -> None:
        if taxonomy not in TAXONOMIES:
            raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
        if detector is None and detector_path is not None:
            detector = PerturbationDetector.load(Path(detector_path))
        if severity_model is None and severity_path is not None:
            severity_model = SeverityClassifier.load(Path(severity_path))
        self.name = name
        self.taxonomy = taxonomy
        self.detector = detector
        self.severity_model = severity_model

    def available(self) -> tuple[bool, str]:
        if self.detector is None:
            return False, "no perturbation detector loaded"
        return True, ""

    def decide(self, sample: ConditionSample, evidence: SceneEvidence) -> DeciderOutput:
        if self.detector is None:
            return DeciderOutput(
                None, None, None, None, None, {}, False, 0.0, 0.0, False, None,
                error="no perturbation detector loaded",
            )

        started = time.perf_counter()
        try:
            features = extract_features(sample.image)
            decision = self.detector.decide(sample.image)
            classes = [str(cls) for cls in getattr(self.detector, "classes", [])]
            raw_proba = np.asarray(self.detector.predict_proba(features), dtype=float)[0]
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = (time.perf_counter() - started) * 1000.0
            return DeciderOutput(
                None, None, None, None, None, {}, False, elapsed, 0.0, False, None,
                error=str(exc),
            )
        elapsed = (time.perf_counter() - started) * 1000.0

        probabilities: dict[str, float] = {}
        for cls, prob in zip(classes, raw_proba):
            label = _class_condition(cls, self.taxonomy)
            probabilities[label] = probabilities.get(label, 0.0) + float(prob)
        probabilities.setdefault("clean", 0.0)

        clean_probability = _clean_probability(classes, raw_proba)
        p_degraded = 1.0 if clean_probability is None else 1.0 - clean_probability

        if decision.label == CLEAN:
            degraded = False
            condition = "clean"
            severity = None
            confidence = clean_probability if clean_probability is not None else float(decision.probability)
        else:
            degraded = True
            perturbation = ALL_PERTURBATIONS.get(decision.predicted)
            family = perturbation.family if perturbation is not None else None
            condition = condition_label(decision.predicted, family, self.taxonomy)
            severity = (
                int(self.severity_model.predict(features))
                if self.severity_model is not None
                else None
            )
            confidence = float(decision.probability)

        return DeciderOutput(
            degraded=degraded,
            condition=condition,
            severity=severity,
            confidence=confidence,
            p_degraded=p_degraded,
            probabilities=probabilities,
            parse_ok=True,
            latency_ms=elapsed,
            cost=0.0,
            cached=False,
            raw={
                "label": decision.label,
                "predicted": decision.predicted,
                "probability": float(decision.probability),
            },
        )


# ---------------------------------------------------------------------------
# Hosted Decisions API adapter
# ---------------------------------------------------------------------------

EVIDENCE_MODES = ("stats", "stats+image", "image")


class JevDecider:
    """Ask an OpenRouter Decisions API model the structured three questions."""

    def __init__(
        self,
        name: str,
        client: Any,
        model: str,
        taxonomy: str = "family",
        evidence_mode: str = "stats",
        include_detector: bool = False,
        exemplars: Any | None = None,
    ) -> None:
        if taxonomy not in TAXONOMIES:
            raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"unknown evidence_mode {evidence_mode!r}; choose from {EVIDENCE_MODES}")
        self.name = name
        self.client = client
        self.model = model
        self.taxonomy = taxonomy
        self.evidence_mode = evidence_mode
        self.include_detector = include_detector
        self.exemplars = exemplars

    def available(self) -> tuple[bool, str]:
        return True, ""

    def _state(self, sample: ConditionSample, evidence: SceneEvidence) -> dict | list:
        if self.evidence_mode in ("stats", "stats+image"):
            state: dict = {"evidence": evidence.to_state(include_detector=self.include_detector)}
        else:
            state = {}
        if self.exemplars is not None:
            state["exemplars"] = self.exemplars
        if self.evidence_mode in ("stats+image", "image"):
            return [state, {"type": "image_url", "image_url": {"url": prepare_image(sample.image)}}]
        return state

    def decide(self, sample: ConditionSample, evidence: SceneEvidence) -> DeciderOutput:
        state = self._state(sample, evidence)
        questions = build_jev_questions(self.taxonomy)
        started = time.perf_counter()
        try:
            response: DecisionResponse = self.client.ask(self.model, state, questions)
        except OpenRouterError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            return DeciderOutput(
                None, None, None, None, None, {}, False, elapsed, 0.0, False, None,
                error=str(exc),
            )

        answers = response.answers if isinstance(response.answers, dict) else {}
        degraded_answer = _nested(answers, "degraded")
        condition_answer = _nested(answers, "condition")
        severity_answer = _nested(answers, "severity")

        noul = _number(degraded_answer.get("noul"))
        degraded = None if noul is None else noul >= 0.5
        p_degraded = noul

        raw_condition = condition_answer.get("choice")
        condition = (
            raw_condition
            if isinstance(raw_condition, str) and raw_condition in TAXONOMIES[self.taxonomy]
            else None
        )

        probabilities: dict[str, float] = {}
        raw_probabilities = condition_answer.get("probabilities")
        if isinstance(raw_probabilities, dict):
            for label, value in raw_probabilities.items():
                probability = _number(value)
                if isinstance(label, str) and probability is not None:
                    probabilities[label] = probability

        confidence = _number(condition_answer.get("confidence"))
        if confidence is None and probabilities:
            confidence = max(probabilities.values())

        score = _number(severity_answer.get("score"))
        severity = None if score is None else min(3, max(1, int(round(score)) + 1))

        parse_ok = noul is not None and condition is not None and score is not None

        return DeciderOutput(
            degraded=degraded,
            condition=condition,
            severity=severity,
            confidence=confidence,
            p_degraded=p_degraded,
            probabilities=probabilities,
            parse_ok=parse_ok,
            latency_ms=float(response.latency_ms),
            cost=float(response.cost),
            cached=bool(response.cached),
            raw=response.raw,
        )


# ---------------------------------------------------------------------------
# Local prompt-and-parse baseline
# ---------------------------------------------------------------------------


def build_hf_prompt(evidence: SceneEvidence, taxonomy: str = "family", include_detector: bool = False) -> str:
    """Text-only prompt for a local instruction model, asking for one JSON object."""
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
    labels = "\n".join(f"- {name}: {description}" for name, description in TAXONOMIES[taxonomy].items())
    evidence_json = json.dumps(evidence.to_state(include_detector=include_detector), indent=2, sort_keys=True)
    return (
        "You are given structured evidence about an image that may be degraded. "
        "Using only the evidence below, decide whether the view is degraded, name the "
        "single best condition label, and rate severity.\n\n"
        f"Allowed condition labels:\n{labels}\n\n"
        "Severity levels:\n"
        "1 = mild\n"
        "2 = moderate\n"
        "3 = severe\n"
        "Use severity 1 when the view is clean.\n\n"
        "Evidence (JSON):\n"
        f"{evidence_json}\n\n"
        "Answer with one JSON object and nothing else. Use exactly this schema:\n"
        '{"degraded": true, "condition": "blur", "severity": 1, "confidence": 0.0}\n'
        '"degraded" must be a boolean, "condition" one of the allowed labels, '
        '"severity" an integer from 1 to 3, and "confidence" a number from 0 to 1.\n'
        'Example clean: {"degraded": false, "condition": "clean", "severity": 1, "confidence": 0.9}\n'
        'Example degraded: {"degraded": true, "condition": "blur", "severity": 3, "confidence": 0.8}'
    )


class HFTextDecider:
    """Local text-only instruction model: prompt with evidence, parse JSON out."""

    def __init__(
        self,
        name: str = "hf:Qwen/Qwen2.5-1.5B-Instruct",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        taxonomy: str = "family",
        max_new_tokens: int = 96,
        device: str | None = None,
        include_detector: bool = False,
    ) -> None:
        if taxonomy not in TAXONOMIES:
            raise ValueError(f"unknown taxonomy {taxonomy!r}; choose from {sorted(TAXONOMIES)}")
        self.name = name
        self.model_name = model
        self.taxonomy = taxonomy
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.include_detector = include_detector
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def available(self) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .config import get_device

        device = get_device(self.device)
        dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
            .to(device)
            .eval()
        )

    def _error(self, started: float, message: str) -> DeciderOutput:
        return DeciderOutput(
            None, None, None, None, None, {}, False,
            (time.perf_counter() - started) * 1000.0, 0.0, False, None,
            error=message,
        )

    def decide(self, sample: ConditionSample, evidence: SceneEvidence) -> DeciderOutput:
        started = time.perf_counter()
        try:
            self._ensure_loaded()
            import torch
        except Exception as exc:
            return self._error(started, f"model load failed: {exc}")

        prompt = build_hf_prompt(evidence, self.taxonomy, include_detector=self.include_detector)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
            output_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            return self._error(started, f"generation failed: {exc}")

        elapsed = (time.perf_counter() - started) * 1000.0
        parsed = parse_json_object(output_text)
        if parsed is None:
            return DeciderOutput(
                None, None, None, None, None, {}, False, elapsed, 0.0, False,
                output_text, error="could not parse model output as a JSON object",
            )

        raw_degraded = parsed.get("degraded")
        degraded = raw_degraded if isinstance(raw_degraded, bool) else None

        raw_condition = parsed.get("condition")
        condition = (
            raw_condition
            if isinstance(raw_condition, str) and raw_condition in TAXONOMIES[self.taxonomy]
            else None
        )

        raw_severity = _number(parsed.get("severity"))
        severity = int(round(raw_severity)) if raw_severity is not None else None
        if severity is not None and not 1 <= severity <= 3:
            severity = None

        # A prompt-and-parse pipeline has to reconcile the clean case: a clean
        # answer has no condition label and no meaningful severity.
        if degraded is False:
            condition = "clean"
            severity = severity or 1

        confidence = _number(parsed.get("confidence"))
        parse_ok = (
            degraded is not None
            and condition is not None
            and (severity is not None or degraded is False)
        )

        if degraded is None:
            p_degraded = None
        else:
            p_degraded = 1.0 if degraded else 0.0
        probabilities = {condition: confidence} if condition is not None and confidence is not None else {}

        return DeciderOutput(
            degraded=degraded,
            condition=condition,
            severity=severity,
            confidence=confidence,
            p_degraded=p_degraded,
            probabilities=probabilities,
            parse_ok=parse_ok,
            latency_ms=elapsed,
            cost=0.0,
            cached=False,
            raw=parsed,
        )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(text: str | None) -> dict | None:
    """Extract one JSON object from plain, fenced, or prose-wrapped text."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = [text.strip(), *_FENCE.findall(text)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _is_true(value: Any) -> bool:
    return value is True


def routing_metrics(rows: Sequence[dict]) -> list[dict]:
    """Per-decider routing table, sorted by decider name.

    Accuracy and recall count a missing prediction (None) as wrong, including in
    the denominator. Severity exact and within-1 likewise count missing
    predictions as wrong; severity MAE only averages rows where both the true
    and predicted severity are present (missing values are excluded from MAE).
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("decider", "")), []).append(row)

    results: list[dict] = []
    for name in sorted(grouped):
        group = grouped[name]
        n = len(group)

        degraded_accuracy = _mean([
            row.get("pred_degraded") is not None
            and row.get("pred_degraded") == row.get("true_degraded")
            for row in group
        ])
        degraded_rows = [row for row in group if row.get("true_degraded") is True]
        degraded_recall = _mean([_is_true(row.get("pred_degraded")) for row in degraded_rows])
        clean_rows = [row for row in group if row.get("true_degraded") is False]
        clean_false_alarm_rate = _mean([_is_true(row.get("pred_degraded")) for row in clean_rows])

        condition_accuracy = _mean([
            row.get("pred_condition") is not None
            and row.get("pred_condition") == row.get("true_condition")
            for row in group
        ])
        y_true = [str(row.get("true_condition")) for row in group]
        y_pred = [
            str(row["pred_condition"]) if row.get("pred_condition") is not None else "<missing>"
            for row in group
        ]
        condition_macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        severity_rows = [row for row in group if row.get("true_severity") is not None]
        severity_exact = _mean([
            row.get("pred_severity") is not None
            and row.get("pred_severity") == row.get("true_severity")
            for row in severity_rows
        ])
        severity_within1 = _mean([
            row.get("pred_severity") is not None
            and abs(int(row["pred_severity"]) - int(row["true_severity"])) <= 1
            for row in severity_rows
        ])
        severity_errors = [
            abs(int(row["pred_severity"]) - int(row["true_severity"]))
            for row in severity_rows
            if row.get("pred_severity") is not None
        ]
        severity_mae = _mean(severity_errors)

        parse_failure_rate = _mean([not bool(row.get("parse_ok")) for row in group])

        latencies = [
            float(row["latency_ms"]) for row in group if not bool(row.get("cached"))
        ]
        if latencies:
            p50, p95 = np.percentile(np.asarray(latencies, dtype=float), [50, 95])
        else:
            p50, p95 = 0.0, 0.0

        results.append({
            "decider": name,
            "n": n,
            "degraded_accuracy": degraded_accuracy,
            "degraded_recall": degraded_recall,
            "clean_false_alarm_rate": clean_false_alarm_rate,
            "condition_accuracy": condition_accuracy,
            "condition_macro_f1": condition_macro_f1,
            "severity_exact": severity_exact,
            "severity_within1": severity_within1,
            "severity_mae": severity_mae,
            "parse_failure_rate": parse_failure_rate,
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "cached_rate": _mean([bool(row.get("cached")) for row in group]),
            "total_cost": float(sum(float(row.get("cost") or 0.0) for row in group)),
        })
    return results


def per_condition_metrics(rows: Sequence[dict]) -> list[dict]:
    """One row per decider x true condition, sorted by decider then condition.

    `degraded_recall` is computed only within that condition's truly degraded
    rows; conditions with no degraded rows (e.g. clean) report 0.0. Severity MAE
    uses rows where both true and predicted severity are present.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("decider", "")), str(row.get("true_condition", "")))
        grouped.setdefault(key, []).append(row)

    results: list[dict] = []
    for decider, true_condition in sorted(grouped):
        group = grouped[(decider, true_condition)]
        degraded_rows = [row for row in group if row.get("true_degraded") is True]
        results.append({
            "decider": decider,
            "true_condition": true_condition,
            "n": len(group),
            "degraded_recall": _mean([_is_true(row.get("pred_degraded")) for row in degraded_rows]),
            "condition_accuracy": _mean([
                row.get("pred_condition") is not None
                and row.get("pred_condition") == row.get("true_condition")
                for row in group
            ]),
            "severity_mae": _mean([
                abs(int(row["pred_severity"]) - int(row["true_severity"]))
                for row in group
                if row.get("true_severity") is not None and row.get("pred_severity") is not None
            ]),
        })
    return results


def score_rows(rows: Sequence[dict]) -> list[dict]:
    """Project full result rows onto the conformal-calibration handoff schema."""
    scored: list[dict] = []
    for row in rows:
        pred_degraded = row.get("pred_degraded")
        true_degraded = row.get("true_degraded")
        scored.append({
            "decider": row.get("decider"),
            "image_id": row.get("image_id"),
            "true_degraded": true_degraded,
            "p_degraded": row.get("p_degraded"),
            "pred_degraded": pred_degraded,
            "degraded_correct": pred_degraded is not None and pred_degraded == true_degraded,
            "true_condition": row.get("true_condition"),
            "pred_condition": row.get("pred_condition"),
            "true_severity": row.get("true_severity"),
            "pred_severity": row.get("pred_severity"),
            "parse_ok": bool(row.get("parse_ok")),
            "cost": row.get("cost"),
            "cached": bool(row.get("cached")),
        })
    return scored
