"""Render example images from a finished restoration run, labeled with the classifier's predictions.

Picks real cases from results/restoration/records.csv.gz, regenerates the images
(perturbation, restoration, depth, cue) and labels each panel with the prediction
recorded in that run, so the figures match the reported numbers exactly.

Outputs (in --out):
  examples_fixed.png         rows the oracle restoration fixed (wrong -> right)
  examples_broken.png        rows the oracle restoration broke (right -> wrong)
  pipeline_walkthrough.png   one image through every stage: perturb, detect, restore, depth, cue

Run after scripts/run_restoration.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoConfig

import _bootstrap  # noqa: F401
from bolero.baselines import apply_baseline
from bolero.config import CLASSIFIER_MODEL, RESULTS_DIR
from bolero.cues import apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator, boundary_strength
from bolero.perturbations import ALL_PERTURBATIONS, rng_for
from bolero.plotting import CATEGORICAL, INK_PRIMARY, apply_style
from bolero.restoration import restore

RIGHT, WRONG = CATEGORICAL[0], CATEGORICAL[7]  # blue / red, same as the gain/loss palette


def pick(rec: pd.DataFrame, perturbation: str, fixed: bool) -> pd.Series | None:
    """Most severe (image, level) where the oracle fixed (or broke) a prediction."""
    key = ["image_id", "level"]
    cond = rec[rec["perturbation"] == perturbation]
    none = cond[cond["method"] == "none"].set_index(key)["out_correct"]
    oracle = cond[cond["method"] == "oracle_classical"].set_index(key)["out_correct"]
    hits = (oracle & ~none) if fixed else (none & ~oracle)
    clean_ok = set(rec[(rec["perturbation"] == "clean") & (rec["method"] == "none") & rec["out_correct"]]["image_id"])
    hits = hits[hits & hits.index.get_level_values("image_id").isin(clean_ok)]  # clean must be right, or the story muddles
    if hits.empty:
        return None
    image_id, level = sorted(hits.index, key=lambda k: -k[1])[0]
    return pd.Series({"image_id": image_id, "perturbation": perturbation, "level": level})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RESULTS_DIR / "restoration")
    parser.add_argument("--fixed", nargs="*", default=["brighten", "gaussian_blur", "inversion"])
    parser.add_argument("--broken", nargs="*", default=["gaussian_noise", "background_removal", "region_removal"])
    parser.add_argument("--cue", default="entity_background_separation")
    parser.add_argument("--baseline", default="autocontrast")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    args = parser.parse_args()

    run = json.loads((args.run / "run.json").read_text())
    rec = pd.read_csv(args.run / "records.csv.gz")
    records = {r.image_id: r for r in list_images(run["split"], per_class=run["per_class"], seed=run["seed"])}
    names = {int(k): v.split(",")[0] for k, v in AutoConfig.from_pretrained(CLASSIFIER_MODEL).id2label.items()}
    depth_cache, estimator = DepthCache(), DepthEstimator(device=args.device)
    cue_key, base_key = f"cue:{args.cue}", f"baseline:{args.baseline}"

    def outcome(image_id, perturbation, level, method, depth_source="unused"):
        return rec[(rec["image_id"] == image_id) & (rec["perturbation"] == perturbation) & (rec["level"] == level)
                  & (rec["method"] == method) & (rec["depth_source"] == depth_source)].iloc[0]

    def label(row) -> tuple[str, str]:
        ok = bool(row["out_correct"])
        return f"{names[int(row['out_pred'])]} {'✓' if ok else '✗'}", RIGHT if ok else WRONG

    def build(case: pd.Series) -> tuple[list[np.ndarray], list[str], list[tuple[str, str]]]:
        record, name, level = records[case["image_id"]], case["perturbation"], int(case["level"])
        clean = load_image(record.path)
        x = ALL_PERTURBATIONS[name](clean, level, depth_cache.get(record, clean),
                                    rng_for(record.image_id, name, level, run["seed"]))
        depth = estimator(x)  # cues see depth estimated on the perturbed input, as in the main setting
        images = [clean, x, restore(x, name),
                  apply_baseline(args.baseline, x, run["strengths"][base_key]),
                  apply_cue(args.cue, x, depth, run["strengths"][cue_key])]
        titles = ["clean", f"{name} L{level}", "oracle restoration", f"{args.baseline}", f"{args.cue} cue"]
        labels = [label(outcome(record.image_id, "clean", 0, "none")),
                  label(outcome(record.image_id, name, level, "none")),
                  label(outcome(record.image_id, name, level, "oracle_classical")),
                  label(outcome(record.image_id, name, level, base_key)),
                  label(outcome(record.image_id, name, level, cue_key, "perturbed"))]
        return images, titles, labels

    def grid(cases: list[pd.Series], title: str, path: Path) -> None:
        rows = [build(c) for c in cases]
        fig, axes = plt.subplots(len(rows), 5, figsize=(13, 2.9 * len(rows)), squeeze=False, layout="constrained")
        for i, (images, titles, labels) in enumerate(rows):
            truth = records[cases[i]["image_id"]].class_name
            for j, (im, t, (lab, color)) in enumerate(zip(images, titles, labels)):
                ax = axes[i, j]
                ax.imshow(np.clip(im, 0, 1))
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                ax.set_title(t, fontsize=10, loc="center", color=INK_PRIMARY)
                ax.set_xlabel(lab, fontsize=10, color=color, fontweight="bold")
            axes[i, 0].set_ylabel(f"true: {truth}", fontsize=10)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrote {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()
    fixed = [c for c in (pick(rec, p, True) for p in args.fixed) if c is not None]
    broken = [c for c in (pick(rec, p, False) for p in args.broken) if c is not None]
    if fixed:
        grid(fixed, "Restoration recovers damage that can be undone", args.out / "examples_fixed.png")
    if broken:
        grid(broken, "Restoration hurts when it has to guess missing content", args.out / "examples_broken.png")

    # Walkthrough: first fixed case through the detector-routed pipeline.
    case = (fixed or broken)[0]
    record, name, level = records[case["image_id"]], case["perturbation"], int(case["level"])
    clean = load_image(record.path)
    x = ALL_PERTURBATIONS[name](clean, level, depth_cache.get(record, clean),
                                rng_for(record.image_id, name, level, run["seed"]))
    det = outcome(record.image_id, name, level, "detector_classical")
    routed = restore(x, det["detected"])
    depth = estimator(x)  # run_restoration applies combo cues with depth from the perturbed input
    cued = apply_cue(args.cue, routed, depth, run["strengths"][cue_key])
    combo = outcome(record.image_id, name, level, f"detector+cue:{args.cue}", "perturbed")
    panels = [
        (clean, "1. clean input", label(outcome(record.image_id, "clean", 0, "none"))),
        (x, f"2. perturbed ({name} L{level})", label(outcome(record.image_id, name, level, "none"))),
        (routed, f"3. detector: {det['detected']} (p={det['detected_prob']:.2f})\n→ classical restoration", label(det)),
        (depth, "4. depth (Depth-Anything-V2)\nestimated on perturbed image", None),
        (boundary_strength(depth), "5. depth boundaries", None),
        (cued, f"6. {args.cue} cue\n→ frozen ResNet-50", label(combo)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.6), layout="constrained")
    for ax, (im, t, lab) in zip(axes, panels):
        ax.imshow(im, cmap=None if im.ndim == 3 else "inferno")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        ax.set_title(t, fontsize=9.5, loc="center", color=INK_PRIMARY)
        if lab:
            ax.set_xlabel(lab[0], fontsize=10, color=lab[1], fontweight="bold")
    fig.suptitle(f"One image through the pipeline (true class: {record.class_name})", fontsize=13, fontweight="bold")
    path = args.out / "pipeline_walkthrough.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
