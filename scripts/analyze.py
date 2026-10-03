"""Stage 4 - figures and a markdown summary from whichever stages have been run.

Reads results/{diagnostic,detector,tuning,restoration,cue_sweep}/ and writes
results/figures/*.png and results/SUMMARY.md.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from bolero.config import RESULTS_DIR
from bolero.perturbations import HELD_OUT
from bolero.plotting import (
    DIVERGING, INK_PRIMARY, SEQUENTIAL, apply_style, diverging_norm, strip_axes,
)

METHOD_ORDER = ["oracle_classical", "detector_classical"]


def annotate(ax, values: np.ndarray, fmt: str, cmap, norm) -> None:
    for (i, j), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        r, g, b, _ = cmap(norm(v))
        dark_cell = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                color="white" if dark_cell else INK_PRIMARY)


def heatmap(values: pd.DataFrame, title: str, cbar_label: str, path: Path, diverging: bool, fmt: str) -> None:
    arr = values.to_numpy(dtype=float)
    cmap = DIVERGING if diverging else SEQUENTIAL
    norm = diverging_norm(np.nan_to_num(arr)) if diverging else plt.Normalize(0, max(1e-6, np.nanmax(arr)))
    fig, ax = plt.subplots(figsize=(1.1 + 0.9 * arr.shape[1], 0.9 + 0.36 * arr.shape[0]))
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(arr.shape[1]), [str(c) for c in values.columns], rotation=35, ha="right")
    ax.set_yticks(range(arr.shape[0]), [str(i) for i in values.index])
    ax.grid(False)
    strip_axes(ax, keep=())
    annotate(ax, arr, fmt, cmap, norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def cell(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def method_label(method: str, source: str) -> str:
    return method if source in ("unused", "perturbed") else f"{method} [{source} depth]"


def main_methods(summary: pd.DataFrame) -> list[str]:
    """Deployable methods: no oracle depth, no control depth, no detector+cue combos."""
    present = set(summary["method"])
    ordered = [m for m in METHOD_ORDER if m in present]
    ordered += sorted(m for m in present if m.startswith("baseline:"))
    ordered += sorted(m for m in present if m.startswith("cue:"))
    return ordered


def pooled(summary: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate per-condition rows, pooling counts so rates are over all images."""
    s = summary.assign(n_wrong=summary["n"] * (1 - summary["acc_perturbed"]),
                       n_correct=summary["n"] * summary["acc_perturbed"],
                       agree_before_n=summary["n"] * summary["agree_clean_before"],
                       agree_after_n=summary["n"] * summary["agree_clean_after"])
    g = s.groupby(by, sort=False).agg(
        n=("n", "sum"), acc_perturbed=("acc_perturbed", "mean"), acc_restored=("acc_restored", "mean"),
        delta_pp=("delta_pp", "mean"), wrong_to_correct=("wrong_to_correct", "sum"),
        correct_to_wrong=("correct_to_wrong", "sum"), n_wrong=("n_wrong", "sum"),
        n_correct=("n_correct", "sum"), agree_before_n=("agree_before_n", "sum"),
        agree_after_n=("agree_after_n", "sum"), back_to_clean_pred=("back_to_clean_pred", "sum"),
        back_to_clean_pred_wrong=("back_to_clean_pred_wrong", "sum"), restore_ms=("restore_ms", "mean"),
    ).reset_index()
    g["fix_rate"] = g["wrong_to_correct"] / g["n_wrong"]
    g["break_rate"] = g["correct_to_wrong"] / g["n_correct"]
    g["agree_clean_before"] = g["agree_before_n"] / g["n"]
    g["agree_clean_after"] = g["agree_after_n"] / g["n"]
    return g.drop(columns=["n_wrong", "n_correct", "agree_before_n", "agree_after_n"])


def restoration_section(rest: Path, figures: Path) -> list[str]:
    summary = pd.read_csv(rest / "summary.csv")
    run = json.loads((rest / "run.json").read_text()) if (rest / "run.json").exists() else {}
    held_out = summary[summary["perturbation"].isin(HELD_OUT)]
    perturbed = summary[(summary["perturbation"] != "clean") & ~summary["perturbation"].isin(HELD_OUT)]
    on_clean = summary[summary["perturbation"] == "clean"]
    main = main_methods(summary)
    is_main = summary["method"].isin(main) & summary["depth_source"].isin(["unused", "perturbed"])
    report = [
        "## Stage 3: restoration", "",
        f"Evaluated on {run.get('images', '?')} val images. Strengths: {run.get('strength_status', '?')}.",
        "Depth for every cue in the main results is re-estimated on the perturbed input. "
        "Accuracy, fixes and breaks are against the ground-truth label; "
        "`agree_clean_*` is agreement with the classifier's prediction on the clean image.",
        "",
    ]

    # Accuracy change per perturbation, deployable methods only.
    gain = (summary[is_main & (summary["method"] != "none")]
            .groupby(["perturbation", "method"])["delta_pp"].mean()
            .unstack("method").reindex(columns=[m for m in main if m != "none"]))
    gain = gain.reindex(["clean"] + [p for p in gain.index if p != "clean"])
    heatmap(gain, "Accuracy change from restoration (mean over severities)",
            "Δ top-1 accuracy vs perturbed input (pp)",
            figures / "restoration_gain.png", diverging=True, fmt=".1f")
    report += ["Row `clean` shows what each method does to unperturbed inputs (the cost of a false alarm).",
               "", "![](figures/restoration_gain.png)", ""]

    # Overall, over all perturbed conditions.
    overall = pooled(perturbed, ["method", "depth_source"])
    overall.insert(0, "label", [method_label(m, s) for m, s in zip(overall["method"], overall["depth_source"])])
    cols = ["label", "acc_perturbed", "acc_restored", "delta_pp", "fix_rate", "break_rate",
            "wrong_to_correct", "correct_to_wrong", "agree_clean_before", "agree_clean_after",
            "back_to_clean_pred", "back_to_clean_pred_wrong", "restore_ms"]
    save_csv(overall, rest / "overall.csv")
    report += [
        "Across all perturbations and severities. `back_to_clean_pred_wrong` counts images that returned "
        "to the clean prediction but are still wrong, because the clean prediction was wrong too.", "",
        md_table(overall[overall["method"] != "none"][cols]), "",
        "On clean inputs:", "",
        md_table(on_clean[on_clean["method"] != "none"]
                 .assign(label=lambda d: [method_label(m, s) for m, s in zip(d["method"], d["depth_source"])])
                 [["label", "delta_pp", "changed_rate", "correct_to_wrong", "wrong_to_correct"]]
                 .rename(columns={"delta_pp": "delta_pp_vs_clean"})),
        "",
    ]

    if len(held_out):
        stress = pooled(held_out, ["perturbation", "method", "depth_source"])
        stress = stress[stress["method"].isin(main) & stress["depth_source"].isin(["unused", "perturbed"])]
        stress.insert(1, "label", [method_label(m, s) for m, s in zip(stress["method"], stress["depth_source"])])
        save_csv(stress, rest / "held_out.csv")
        report += [
            "### Held-out stress test", "",
            f"Perturbations never seen by the detector or strength tuning ({', '.join(HELD_OUT)}). "
            "They are excluded from the pooled table above. There is no oracle plan for them.", "",
            md_table(stress[["perturbation", "label", "acc_perturbed", "acc_restored", "delta_pp",
                             "fix_rate", "break_rate"]]), "",
        ]

    # Fix and break rates by perturbation x severity.
    by_cond = summary[is_main & (summary["method"] != "none") & (summary["perturbation"] != "clean")].copy()
    by_cond["condition"] = by_cond["perturbation"] + " L" + by_cond["level"].astype(str)
    for metric, title in [("fix_rate", "Fixed: wrong -> correct (share of wrong inputs)"),
                          ("break_rate", "Broken: correct -> wrong (share of correct inputs)")]:
        table = by_cond.pivot_table(index="condition", columns="method", values=metric, sort=False)
        table = table.reindex(columns=[m for m in main if m in table.columns])
        heatmap(table, title, metric.replace("_", " "), figures / f"restoration_{metric}.png",
                diverging=False, fmt=".2f")
    report += ["Fixes and breaks by perturbation and severity (per-condition counts are in "
               "`restoration/summary.csv`):", "",
               "![](figures/restoration_fix_rate.png)", "", "![](figures/restoration_break_rate.png)", ""]

    # Does depth information matter? Same cue, three depth maps.
    cue_rows = perturbed[perturbed["method"].str.startswith("cue:")]
    if cue_rows["depth_source"].nunique() > 1:
        by_source = cue_rows.groupby(["method", "depth_source"])["delta_pp"].mean().unstack("depth_source")
        by_source = by_source.reindex(columns=[c for c in ["perturbed", "clean", "mismatched"]
                                               if c in by_source.columns])
        heatmap(by_source, "Cue accuracy change by depth source", "Δ top-1 accuracy vs perturbed input (pp)",
                figures / "cue_depth_source.png", diverging=True, fmt=".2f")
        report += [
            "### Does the depth map matter?", "",
            "Each cue with depth re-estimated on the perturbed input (deployable), the clean image's depth "
            "(idealized) and an unrelated image's depth (control). If `mismatched` matches `perturbed`, "
            "the cue's effect does not come from depth information.", "",
            "![](figures/cue_depth_source.png)", "", md_table(by_source.reset_index()), "",
        ]

    # Detector restoration followed by each cue: which components add value?
    combos = overall[overall["method"].str.startswith("detector+cue:")]
    if len(combos):
        det = overall[overall["method"] == "detector_classical"]
        report += ["### Components: detector restoration alone vs. followed by each cue", "",
                   md_table(pd.concat([det, combos])[["label", "acc_restored", "delta_pp", "fix_rate",
                                                      "break_rate"]]), ""]

    report += depth_reliability_section(rest)
    return report


def depth_reliability_section(rest: Path) -> list[str]:
    """Fix and break rates of each cue, binned by how well perturbed depth matches clean depth."""
    path = rest / "records.csv.gz"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "depth_rank_corr" not in df:
        return []
    keys = ["image_id", "perturbation", "level"]
    df = df[(df["perturbation"] != "clean") & ~df["perturbation"].isin(HELD_OUT)]
    ref = df[df["method"] == "none"][keys + ["out_correct", "depth_rank_corr"]]
    ref = ref.rename(columns={"out_correct": "before_ok"}).dropna(subset=["depth_rank_corr"])
    ref["depth_bin"] = pd.qcut(ref["depth_rank_corr"], 4, duplicates="drop")
    cues = df[df["method"].str.startswith("cue:") & (df["depth_source"] == "perturbed")]
    m = cues[keys + ["method", "out_correct"]].merge(ref, on=keys)
    m["fixed"] = ~m["before_ok"] & m["out_correct"]
    m["broken"] = m["before_ok"] & ~m["out_correct"]
    rows = []
    for (method, bin_), g in m.groupby(["method", "depth_bin"], observed=True):
        rows.append({"method": method, "depth_rank_corr": str(bin_), "n": len(g),
                     "fix_rate": g["fixed"].sum() / max(1, (~g["before_ok"]).sum()),
                     "break_rate": g["broken"].sum() / max(1, g["before_ok"].sum()),
                     "delta_pp": 100.0 * (g["out_correct"].mean() - g["before_ok"].mean())})
    table = pd.DataFrame(rows)
    save_csv(table, rest / "depth_reliability.csv")
    bins = ref.groupby("depth_bin", observed=True).agg(acc_perturbed=("before_ok", "mean"),
                                                       n=("before_ok", "size")).reset_index()
    return [
        "### Do failures coincide with unreliable depth?", "",
        "Depth reliability is the Spearman correlation between depth estimated on the perturbed input "
        "and on the clean image (quartile bins over all perturbed inputs).", "",
        md_table(bins.assign(depth_bin=bins["depth_bin"].astype(str))), "",
        md_table(table), "",
    ]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    apply_style()
    figures = args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report = ["# Results summary", "", "Generated by `scripts/analyze.py`.", ""]

    # ---- Stage 1: diagnostic ----
    diag = args.results / "diagnostic"
    if (diag / "summary.csv").exists():
        head = json.loads((diag / "headline.json").read_text())
        summary = pd.read_csv(diag / "summary.csv")
        drop = summary.pivot(index="perturbation", columns="level", values="delta_pp")
        drop = drop.loc[drop.mean(axis=1).sort_values().index]
        drop.columns = [f"level {c}" for c in drop.columns]
        heatmap(drop, "Accuracy change under perturbation", "Δ top-1 accuracy vs clean (pp)",
                figures / "diagnostic_accuracy_change.png", diverging=True, fmt=".1f")
        report += [
            "## Stage 1: diagnostic sweep", "",
            f"- {head['records']:,} records from {head['images']:,} images, "
            f"{head['perturbations']} perturbations x {head['levels']} severities",
            f"- Clean accuracy {head['clean_accuracy']:.1%}; perturbed accuracy {head['perturbed_accuracy']:.1%}",
            f"- Prediction changed in {head['prediction_changed_rate']:.2%} of records; "
            f"{head['correct_to_wrong']:,} correct->wrong, {head['wrong_to_correct']:,} wrong->correct",
            "", "![](figures/diagnostic_accuracy_change.png)", "",
            md_table(summary[["perturbation", "level", "acc_pert", "delta_pp", "changed_rate",
                              "correct_to_wrong", "wrong_to_correct", "psnr", "ssim"]]),
            "",
        ]

    # ---- Stage 2: detector ----
    det = args.results / "detector"
    if (det / "eval_predictions.csv").exists():
        head = json.loads((det / "headline.json").read_text())
        evals = pd.read_csv(det / "eval_predictions.csv")
        order = ["clean"] + [p for p in evals["true"].unique() if p != "clean"]
        confusion = pd.crosstab(evals["true"], evals["predicted"], normalize="index")
        confusion = confusion.reindex(index=order, columns=order, fill_value=0.0)
        heatmap(confusion, "Detector confusion (val)", "fraction of true class",
                figures / "detector_confusion.png", diverging=False, fmt=".2f")
        report += [
            "## Stage 2: perturbation detector", "",
            f"- Top-1 accuracy {head['top1_accuracy']:.1%} over {head['val_samples']:,} val samples",
            f"- Clean false-alarm rate {head['clean_false_alarm_rate']:.1%} at threshold {head['threshold']}",
            f"- Perturbed inputs routed to restoration: {head['perturbed_detected_rate']:.1%}",
            "", "![](figures/detector_confusion.png)", "",
        ]

    # ---- Strength tuning (dev split) ----
    tuning = args.results / "tuning"
    if (tuning / "strengths.json").exists():
        meta = json.loads((tuning / "strengths.json").read_text())
        tsum = pd.read_csv(tuning / "summary.csv")
        chosen = pd.DataFrame([{"method": m, "strength": s, "constraint_met": meta["constraint_met"][m]}
                               for m, s in meta["strengths"].items()])
        chosen = chosen.merge(tsum, on=["method", "strength"])
        report += [
            "## Strength selection (development data)", "",
            f"Strengths were chosen on {meta['images']} `{meta['split']}`-split images, never on the val "
            f"images used below. Rule: best mean accuracy over all perturbed conditions, among strengths "
            f"that cost at most {meta['max_clean_drop_pp']} pp on clean images.", "",
            md_table(chosen[["method", "strength", "acc_perturbed", "gain_pp", "clean_drop_pp",
                             "constraint_met"]]),
            "",
        ]

    # ---- Stage 3: restoration ----
    rest = args.results / "restoration"
    if (rest / "summary.csv").exists():
        report += restoration_section(rest, figures)

    # ---- Cue sweep ----
    sweep = args.results / "cue_sweep"
    if (sweep / "summary.csv").exists():
        summary = pd.read_csv(sweep / "summary.csv")
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for cue, g in summary.groupby("cue", sort=False):
            ax.plot(g["strength"], 100 * g["changed_rate"], marker="o", label=cue.replace("_", " "))
        ax.set_xlabel("cue strength")
        ax.set_ylabel("predictions changed vs clean (%)")
        ax.set_title("Clean-image stability of each contextual cue")
        ax.legend(loc="upper left")
        strip_axes(ax)
        fig.savefig(figures / "cue_sweep.png")
        plt.close(fig)
        pivot = summary.pivot(index="strength", columns="cue", values="changed_rate").reset_index()
        report += ["## Cue strength sweep (clean images)", "", "![](figures/cue_sweep.png)", "",
                   "Fraction of predictions changed vs. the clean image:", "", md_table(pivot), ""]

    (args.results / "SUMMARY.md").write_text("\n".join(report))
    print(f"Wrote {args.results / 'SUMMARY.md'} and {figures}/")


if __name__ == "__main__":
    main()
