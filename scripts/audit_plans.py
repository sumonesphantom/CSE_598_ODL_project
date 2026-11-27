"""Audit the classical restoration plans on development data: which ones help?

Stage 3 found that routing more images to restoration (the conformal gate) lowered
accuracy, because several plans hurt even when the perturbation is known. This
script measures each plan with the *true* perturbation (the oracle setting) on
Imagenette **train** images that the perturbation detector was not trained on, so
the decision is never made on the val (evaluation) split.

Rule: a plan is skipped (pass-through) when its mean accuracy change over all
levels is below --min-gain-pp. The detector routes by perturbation name, not by
level, so the decision is per perturbation.

Outputs (in --out):
  summary.csv       per perturbation x level: accuracy before/after, delta, fixes, breaks
  per_plan.csv      per perturbation: mean delta over levels, decision
  skip_plans.json   {"skip": [...], ...} for run_restoration.py --skip-plans
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import save
from bolero.restoration import PLANS, restore


def dev_records(detector_per_class: int, per_class: int, seed: int):
    """Train images disjoint from the detector's training sample, `per_class` per class."""
    used = {r.image_id for r in list_images("train", per_class=detector_per_class, seed=seed)}
    rng = np.random.default_rng(seed + 200)
    available = [r for r in list_images("train") if r.image_id not in used]
    out = []
    for wnid in sorted({r.wnid for r in available}):
        pool = [r for r in available if r.wnid == wnid]
        out += [pool[i] for i in np.sort(rng.choice(len(pool), size=min(per_class, len(pool)), replace=False))]
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per perturbation x level: plan vs. no plan on the same images."""
    wide = df.pivot_table(index=["image_id", "perturbation", "level"], columns="method",
                          values="correct", aggfunc="first").reset_index()
    wide["fix"] = ~wide["none"].astype(bool) & wide["oracle"].astype(bool)
    wide["brk"] = wide["none"].astype(bool) & ~wide["oracle"].astype(bool)
    g = wide.groupby(["perturbation", "level"], sort=False)
    out = g.agg(n=("image_id", "size"), acc_none=("none", "mean"), acc_plan=("oracle", "mean"),
                fix_rate=("fix", "mean"), break_rate=("brk", "mean")).reset_index()
    out["delta_pp"] = 100 * (out["acc_plan"] - out["acc_none"])
    return out


def decide(summary: pd.DataFrame, min_gain_pp: float) -> pd.DataFrame:
    per = summary.groupby("perturbation", sort=False).agg(
        mean_delta_pp=("delta_pp", "mean"), worst_level_pp=("delta_pp", "min"),
        best_level_pp=("delta_pp", "max")).reset_index()
    per["has_plan"] = per["perturbation"].map(lambda p: bool(PLANS.get(p)))
    per["skip"] = per["has_plan"] & (per["mean_delta_pp"] < min_gain_pp)
    return per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=10, help="train images per class")
    parser.add_argument("--detector-train-per-class", type=int, default=40,
                        help="must match train_detector.py --train-per-class")
    parser.add_argument("--min-gain-pp", type=float, default=0.0,
                        help="skip a plan whose mean accuracy change is below this (percentage points)")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "restoration_audit")
    args = parser.parse_args()

    classifier = FrozenClassifier(args.backbone, device=args.device)
    predictor = BatchPredictor(classifier, args.batch_size)
    depth_cache = DepthCache()
    records = dev_records(args.detector_train_per_class, args.per_class, args.seed)
    for record in tqdm(records, desc="train images"):
        clean = load_image(record.path)
        depth = depth_cache.get(record, clean)
        base = {"image_id": record.image_id, "class_name": record.class_name, "label_index": record.label_index}
        for name, perturb in PERTURBATIONS.items():
            if not PLANS.get(name):
                continue
            for level in LEVELS:
                x = perturb(clean, level, depth, rng_for(record.image_id, name, level, args.seed))
                meta = {**base, "perturbation": name, "level": level}
                predictor.add({**meta, "method": "none"}, x)
                predictor.add({**meta, "method": "oracle"}, restore(x, name))
    predictor.flush()

    df = pd.DataFrame(predictor.rows)
    summary = summarize(df)
    per = decide(summary, args.min_gain_pp)
    save(df, args.out / "records.csv.gz")
    save(summary, args.out / "summary.csv")
    save(per, args.out / "per_plan.csv")
    (args.out / "skip_plans.json").write_text(json.dumps({
        "skip": per.loc[per["skip"], "perturbation"].tolist(),
        "keep": per.loc[~per["skip"] & per["has_plan"], "perturbation"].tolist(),
        "rule": f"mean oracle delta over levels < {args.min_gain_pp} pp",
        "split": "train", "images": len(records), "seed": args.seed, "backbone": args.backbone,
    }, indent=2))
    with pd.option_context("display.width", 200):
        print(per.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
