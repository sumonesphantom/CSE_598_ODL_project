"""Generate a consolidated, read-only Bolero benchmark report.

Reads existing result artifacts and writes:
    results/CONSOLIDATED_BENCHMARK_REPORT.md

Does not run experiments, load models, or modify source result files.
Uses only the Python standard library.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "CONSOLIDATED_BENCHMARK_REPORT.md"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path.relative_to(ROOT)}: {exc}")
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        print(f"Warning: could not read {path.relative_to(ROOT)}: {exc}")
        return []


def number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(str(value).replace(",", "").replace("%", ""))
        return None if result != result else result
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    n = number(value)
    return "N/A" if n is None else f"{n:.{digits}f}"


def pct(value: Any, digits: int = 2) -> str:
    n = number(value)
    return "N/A" if n is None else f"{100 * n:.{digits}f}%"


def pp(value: Any, digits: int = 2) -> str:
    n = number(value)
    return "N/A" if n is None else f"{n:.{digits}f} pp"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data available._"

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def weighted_mean(rows: list[dict[str, str]], field: str) -> float | None:
    numerator = denominator = 0.0
    for row in rows:
        value, weight = number(row.get(field)), number(row.get("n"))
        if value is None:
            continue
        weight = weight if weight is not None and weight > 0 else 1.0
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def pooled_transition_rates(rows: list[dict[str, str]]) -> tuple[float | None, float | None]:
    """Pool transition counts over the actual before-correct/before-wrong denominators."""
    fixed = broken = initially_wrong = initially_correct = 0.0
    for row in rows:
        n = number(row.get("n"))
        before_acc = number(row.get("acc_perturbed"))
        if n is None or before_acc is None:
            continue
        fixed += number(row.get("wrong_to_correct")) or 0.0
        broken += number(row.get("correct_to_wrong")) or 0.0
        initially_correct += n * before_acc
        initially_wrong += n * (1.0 - before_acc)
    fix_rate = fixed / initially_wrong if initially_wrong > 0 else None
    break_rate = broken / initially_correct if initially_correct > 0 else None
    return fix_rate, break_rate


def restoration_summary(path: Path) -> list[dict[str, Any]]:
    """Aggregate non-clean rows independently by method AND depth source."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if row.get("perturbation", "").lower() == "clean":
            continue
        key = (row.get("method", "unknown"), row.get("depth_source", "unknown"))
        groups[key].append(row)

    summaries = []
    for (method, depth_source), group in sorted(groups.items()):
        fix_rate, break_rate = pooled_transition_rates(group)
        summaries.append({
            "method": method,
            "depth_source": depth_source,
            "n": sum(number(row.get("n")) or 0 for row in group),
            "conditions": len(group),
            "acc_perturbed": weighted_mean(group, "acc_perturbed"),
            "acc_restored": weighted_mean(group, "acc_restored"),
            "delta_pp": weighted_mean(group, "delta_pp"),
            "fix_rate": fix_rate,
            "break_rate": break_rate,
            "restore_ms": weighted_mean(group, "restore_ms"),
        })
    return summaries


def restoration_table(path: Path) -> str:
    rows = []
    for item in restoration_summary(path):
        rows.append([
            item["method"],
            item["depth_source"],
            f'{item["n"]:,.0f}',
            item["conditions"],
            pct(item["acc_perturbed"]),
            pct(item["acc_restored"]),
            pp(item["delta_pp"]),
            pct(item["fix_rate"]),
            pct(item["break_rate"]),
            fmt(item["restore_ms"], 2),
        ])
    return markdown_table(
        [
            "Method", "Depth source", "Evaluated samples", "Conditions",
            "Perturbed accuracy", "Restored accuracy", "Change",
            "Pooled fix rate", "Pooled break rate", "Restore ms",
        ],
        rows,
    )


def json_summary(path: Path, fields: list[tuple[str, str]]) -> str:
    data = read_json(path)
    if data is None:
        return f"_Not available: `{rel(path)}`._"
    rows = [[label, data[key]] for label, key in fields if key in data]
    return markdown_table(["Metric", "Value"], rows)


def selected_strengths() -> str:
    data = read_json(RESULTS / "tuning" / "strengths.json")
    if not data:
        return "_Strength tuning metadata not available._"
    strengths = data.get("strengths", {})
    constraints = data.get("constraint_met", {})
    rows = [
        [method, value, "Yes" if constraints.get(method) else "No"]
        for method, value in sorted(strengths.items())
    ]
    heading = (
        f'Tuning split: **{data.get("split", "unknown")}**; '
        f'{data.get("images", "unknown")} images; '
        f'{data.get("per_class", "unknown")} per class; '
        f'seed {data.get("seed", "unknown")}. '
        f'Maximum allowed clean accuracy drop: '
        f'{data.get("max_clean_drop_pp", "unknown")} pp.\n\n'
    )
    return heading + markdown_table(
        ["Method", "Selected strength", "Constraint met"], rows
    )


def selected_plans() -> str:
    data = read_json(RESULTS / "tuning" / "plans.json")
    if not data:
        return "_Plan tuning metadata not available._"
    rows = [[name, variant] for name, variant in sorted(data.get("plans", {}).items())]
    heading = (
        f'Tuning split: **{data.get("split", "unknown")}**; '
        f'{data.get("images", "unknown")} images; '
        f'{data.get("per_class", "unknown")} per class; '
        f'seed {data.get("seed", "unknown")}. '
        f'Minimum gain criterion: {data.get("min_gain_pp", "unknown")} pp.\n\n'
    )
    return heading + markdown_table(["Perturbation", "Selected plan"], rows)


def coverage_table(path: Path) -> str:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        groups[(row.get("method", "unknown"), row.get("depth_source", "unknown"))].append(row)
    rows = []
    for (method, depth_source), group in sorted(groups.items()):
        samples = sum(number(row.get("n")) or 0 for row in group)
        perturbations = sorted({row.get("perturbation", "") for row in group})
        rows.append([
            method, depth_source, len(group), f"{samples:,.0f}",
            ", ".join(p for p in perturbations if p),
        ])
    return markdown_table(
        ["Method", "Depth source", "Rows", "Evaluated samples", "Perturbations represented"],
        rows,
    )


def artifact_inventory() -> str:
    paths = [
        "audit/headline_metrics.csv", "diagnostic/headline.json",
        "diagnostic/summary.csv", "detector/headline.json",
        "detector/eval_summary.csv", "detector_retrained/headline.json",
        "detector_retrained/eval_summary.csv", "tuning/strengths.json",
        "tuning/summary.csv", "tuning/plans.json", "tuning/plan_summary.csv",
        "restoration/run.json", "restoration/summary.csv",
        "restoration/plan_validation_50_retry/run.json",
        "restoration/plan_validation_50_retry/summary.csv",
        "cue_sweep/summary.csv", "distortions/summary.csv",
    ]
    rows = [
        [f"`results/{item}`",
         "Available" if (RESULTS / item).is_file() else "Not found"]
        for item in paths
    ]
    return markdown_table(["Artifact", "Status"], rows)


def build_report() -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    audit = read_csv(RESULTS / "audit" / "headline_metrics.csv")
    audit_map = {
        row.get("metric", "").strip().lower(): row.get("value", "")
        for row in audit
    }
    diagnostic = read_json(RESULTS / "diagnostic" / "headline.json")
    detector_old = read_json(RESULTS / "detector" / "headline.json")
    detector_new = read_json(RESULTS / "detector_retrained" / "headline.json")
    original_run = read_json(RESULTS / "restoration" / "run.json")
    plan_dir = RESULTS / "restoration" / "plan_validation_50_retry"
    plan_run = read_json(plan_dir / "run.json")

    add("# Bolero: Consolidated Benchmark Report")
    add()
    add("**Generated from existing experiment artifacts.**")
    add()
    add(
        "This report consolidates the available audit, diagnostic, detector, "
        "tuning, restoration, cue-sweep, and distortion results. These artifacts "
        "come from distinct experiment stages and are not one single end-to-end run."
    )
    add()
    add("> Negative and neutral outcomes are retained. This report does not assume "
        "that every restoration method improves recognition.")
    add()
    add("## 1. Research objective")
    add()
    add(
        "The project investigates whether depth-aware, context-preserving image "
        "processing can help a classifier retain recognition performance under "
        "visual perturbations, and whether a detector can route selected inputs "
        "through a restoration stage."
    )
    add()
    add("## 2. Experimental organization")
    add()
    add(markdown_table(
        ["Stage", "Data / configuration", "Primary artifact"],
        [
            ["Initial audit", "3,925 images", "`results/audit/`"],
            ["Perturbation diagnostic", "3,925 images; 14 perturbations; 3 levels",
             "`results/diagnostic/`"],
            ["Detector evaluation", "18,000 train; 9,000 validation samples",
             "`results/detector/` and `results/detector_retrained/`"],
            ["Strength tuning", "100 training images; 10 per class",
             "`results/tuning/strengths.json`"],
            ["Plan tuning", "200 training images; 20 per class",
             "`results/tuning/plans.json`"],
            ["Original restoration", "500 validation images; 50 per class",
             "`results/restoration/`"],
            ["Tuned-plan validation", "500 validation images; 50 per class",
             "`results/restoration/plan_validation_50_retry/`"],
        ],
    ))
    add()
    add("**Counting note:** In restoration tables, evaluated samples are the sum of "
        "the row-level `n` values across perturbation/severity conditions. They are "
        "not unique images. The same validation image is evaluated in multiple "
        "conditions and, in the original benchmark, under multiple depth sources.")
    add()
    add("## 3. Initial Bolero audit")
    add()
    if audit_map:
        labels = [
            "images", "accuracy, original", "accuracy, bolero", "accuracy change",
            "relative error increase", "predictions changed",
            "mean confidence, original", "mean confidence, bolero",
            "mean confidence change",
        ]
        add(markdown_table(
            ["Metric", "Result"],
            [[label, audit_map[label]] for label in labels if label in audit_map],
        ))
    else:
        add("_Audit headline metrics were not found._")
    add()
    add(
        "The audit indicates that the evaluated Bolero transformation did not "
        "preserve the original classifier's accuracy in this run. It motivates "
        "more selective processing rather than establishing an overall gain."
    )
    add()
    add("Source: `results/audit/headline_metrics.csv`.")
    add()
    add("## 4. Perturbation diagnostic")
    add()
    if diagnostic:
        add(markdown_table(
            ["Metric", "Result"],
            [
                ["Images", diagnostic.get("images", "N/A")],
                ["Records", diagnostic.get("records", "N/A")],
                ["Perturbations", diagnostic.get("perturbations", "N/A")],
                ["Severity levels", diagnostic.get("levels", "N/A")],
                ["Clean accuracy", pct(diagnostic.get("clean_accuracy"))],
                ["Perturbed accuracy", pct(diagnostic.get("perturbed_accuracy"))],
                ["Prediction changed rate", pct(diagnostic.get("prediction_changed_rate"))],
                ["Correct to wrong", diagnostic.get("correct_to_wrong", "N/A")],
                ["Wrong to correct", diagnostic.get("wrong_to_correct", "N/A")],
                ["Held-out rand_augment accuracy",
                 pct(diagnostic.get("held_out_accuracy", {}).get("rand_augment"))],
                ["Wall time (minutes)", fmt(diagnostic.get("wall_minutes"), 2)],
            ],
        ))
    else:
        add("_Diagnostic headline metadata was not found._")
    add()
    add(
        "Correct-to-wrong and wrong-to-correct counts show that perturbations "
        "can both break previously correct predictions and occasionally correct "
        "previously incorrect ones."
    )
    add()
    add("Source: `results/diagnostic/headline.json` and `summary.csv`.")
    add()
    add("## 5. Detector evaluation")
    add()
    detector_rows = []
    for label, data in [
        ("Original detector", detector_old),
        ("Retrained detector", detector_new),
    ]:
        if data:
            detector_rows.append([
                label, data.get("train_samples", "N/A"),
                data.get("val_samples", "N/A"), pct(data.get("top1_accuracy")),
                pct(data.get("routed_correctly")),
                pct(data.get("clean_false_alarm_rate")),
                pct(data.get("perturbed_detected_rate")),
            ])
    add(markdown_table(
        ["Detector", "Train samples", "Validation samples", "Top-1 accuracy",
         "Routed correctly", "Clean false-alarm rate", "Perturbed detection rate"],
        detector_rows,
    ))
    add()
    add(
        "Top-1 accuracy, routing correctness, clean false-alarm rate, and "
        "perturbed detection rate are distinct measurements. The retrained "
        "detector is the one named in the newer plan-validation metadata."
    )
    add()
    add("Sources: `results/detector/` and `results/detector_retrained/`.")
    add()
    add("## 6. Restoration parameter tuning")
    add()
    add("### 6.1 Strength tuning")
    add()
    add(selected_strengths())
    add()
    add(
        "Strength selection used training data, a candidate grid, and a maximum "
        "permitted clean-accuracy drop. These are tuning outputs, not independent "
        "validation results."
    )
    add()
    add("Source: `results/tuning/strengths.json` and `summary.csv`.")
    add()
    add("### 6.2 Restoration plan tuning")
    add()
    add(selected_plans())
    add()
    add(
        "Plan selection is separate from strength selection. Pass-through "
        "selections mean that no restoration operation was retained for those "
        "perturbations; they are not successful image reconstructions."
    )
    add()
    add("Source: `results/tuning/plans.json` and `plan_summary.csv`.")
    add()
    add("## 7. Original broad restoration benchmark")
    add()
    if original_run:
        add(markdown_table(
            ["Metadata", "Value"],
            [
                ["Split", original_run.get("split", "N/A")],
                ["Images", original_run.get("images", "N/A")],
                ["Images per class", original_run.get("per_class", "N/A")],
                ["Seed", original_run.get("seed", "N/A")],
                ["Strength status", original_run.get("strength_status", "N/A")],
                ["Depth sources", ", ".join(original_run.get("depth_sources", []))],
            ],
        ))
    add()
    add("### Results by method and depth source")
    add()
    add(restoration_table(RESULTS / "restoration" / "summary.csv"))
    add()
    add(
        "Each cue-based method is reported separately for `perturbed`, `clean`, "
        "and `mismatched` depth. Baseline methods and detector/oracle controls "
        "use `unused` because they do not consume a depth source. Clean "
        "perturbation rows are excluded from these restoration-effect aggregates."
    )
    add()
    add(
        "**Aggregation:** Accuracy and timing are weighted by each row's sample "
        "count. Fix rate is pooled as total wrong-to-correct transitions divided "
        "by the total number initially incorrect. Break rate is pooled as total "
        "correct-to-wrong transitions divided by the total number initially "
        "correct. These denominators differ, so averaging row-level rates by `n` "
        "would not produce pooled transition rates."
    )
    add()
    add(
        "The sample count is repeated condition-level evaluation volume, not a "
        "count of distinct validation images. The original broad benchmark is "
        "kept separate from the tuned-plan validation because their method sets "
        "and configurations differ."
    )
    add()
    add("Source: `results/restoration/run.json` and `results/restoration/summary.csv`.")
    add()
    add("## 8. Tuned-plan validation benchmark")
    add()
    if plan_run:
        variants = plan_run.get("plan_variants", {})
        add(markdown_table(
            ["Metadata", "Value"],
            [
                ["Split", plan_run.get("split", "N/A")],
                ["Images", plan_run.get("images", "N/A")],
                ["Images per class", plan_run.get("per_class", "N/A")],
                ["Seed", plan_run.get("seed", "N/A")],
                ["Backbone", plan_run.get("backbone", "N/A")],
                ["Strength status", plan_run.get("strength_status", "N/A")],
                ["Plan status", plan_run.get("plan_status", "N/A")],
                ["Detector", plan_run.get("detector", "N/A")],
                ["Depth sources in run configuration",
                 ", ".join(plan_run.get("depth_sources", []))],
            ],
        ))
        add()
        add("### Plan variants used")
        add()
        add(markdown_table(
            ["Perturbation", "Variant"],
            [[key, value] for key, value in sorted(variants.items())],
        ))
    add()
    add("### Results by method")
    add()
    add(restoration_table(plan_dir / "summary.csv"))
    add()
    add("### Actual evaluation coverage")
    add()
    add(coverage_table(plan_dir / "summary.csv"))
    add()
    add(
        "Although the run metadata lists `perturbed`, `clean`, and `mismatched` "
        "as configured depth sources, the saved validation summary has "
        "`depth_source=unused` for all methods. Therefore this validation output "
        "does not provide a comparison across those three depth sources."
    )
    add()
    add(
        "The `none` and `detector_classical` methods each cover clean plus 15 "
        "perturbations. `oracle_classical` has fewer rows and omits clean and "
        "`rand_augment`, so its aggregate has a different condition set and "
        "should not be treated as a strictly matched overall comparison."
    )
    add()
    add(
        "This validation run uses the selected strength and plan configurations "
        "and the retrained detector. It remains independent of the original "
        "broad restoration benchmark."
    )
    add()
    add("Source: `results/restoration/plan_validation_50_retry/run.json` and `summary.csv`.")
    add()
    add("## 9. Cue and distortion experiments")
    add()
    cue_path = RESULTS / "cue_sweep" / "summary.csv"
    distortion_path = RESULTS / "distortions" / "summary.csv"
    cue_rows, distortion_rows = read_csv(cue_path), read_csv(distortion_path)
    add(markdown_table(
        ["Experiment", "Rows available", "Artifact"],
        [
            ["Cue sweep", len(cue_rows), f"`{rel(cue_path)}`"],
            ["Distortion evaluation", len(distortion_rows), f"`{rel(distortion_path)}`"],
        ],
    ))
    add()
    add(
        "These experiments provide additional evidence about individual cues "
        "and distortion families. Their own sample sizes and evaluation "
        "protocols apply; they are not automatically comparable to the main "
        "restoration benchmarks."
    )
    add()
    add("## 10. Interpretation and limitations")
    add()
    add("The available results support the following cautious conclusions:")
    add()
    add("1. **Perturbations affect recognition.** The diagnostic shows a substantial "
        "difference between clean and perturbed accuracy.")
    add("2. **Detector routing is imperfect.** Clean false alarms and missed "
        "perturbations remain relevant when deciding whether to invoke restoration.")
    add("3. **Restoration is not universally beneficial.** Some transformations "
        "can damage useful classifier cues, producing neutral or negative outcomes.")
    add("4. **Selective processing is a meaningful design direction.** Pass-through "
        "plan selections indicate that leaving an input unchanged can be preferable "
        "to applying a harmful operation.")
    add("5. **The two restoration runs answer different questions.** They must "
        "remain separate in headline claims and comparisons.")
    add()
    add("### Limitations")
    add()
    add("- The results use finite validation samples and a defined set of controlled "
        "perturbations.")
    add("- Different runs vary in method coverage, detector version, tuning "
        "procedure, and condition coverage.")
    add("- Pass-through is not evidence of successful reconstruction.")
    add("- Classifier accuracy alone does not establish object-level robustness "
        "or performance under real weather and surveillance conditions.")
    add("- Saved outputs alone do not establish full pipeline reproducibility; "
        "source code, model versions, dependencies, and data provenance must "
        "also be retained.")
    add()
    add("## 11. Excluded and pending work")
    add()
    add("The following smoke-test directories are excluded from headline benchmark aggregation:")
    add()
    add("- `results/restoration/detector_smoke/`")
    add("- `results/restoration/plan_smoke/`")
    add()
    add(
        "NAFNet adapter smoke tests, if present, demonstrate execution rather "
        "than comparative restoration effectiveness. Weather-specific "
        "diagnostics, real-world weather evaluation, generative degradation "
        "comparisons, object-detection baselines, and security-oriented weather "
        "attacks remain pending until their evaluation artifacts are available."
    )
    add()
    add("## 12. Artifact inventory")
    add()
    add(artifact_inventory())
    add()
    add("## 13. Reproducibility notes")
    add()
    add("- All artifact paths in this report are repository-relative.")
    add("- The generator reads existing files only; it does not load model weights "
        "or execute experiment scripts.")
    add("- `results/SUMMARY.md` is not overwritten.")
    add("- Smoke-test outputs are excluded from the main restoration tables.")
    add("- Missing artifacts are marked unavailable rather than replaced with "
        "inferred values.")
    add()
    add("---")
    add()
    add(f"_Generated from the current contents of `{rel(RESULTS)}/` by "
        f"`scripts/generate_report.py`._")
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Report written: {rel(OUTPUT)}")
    print(f"Report size: {len(report):,} characters")


if __name__ == "__main__":
    main()
