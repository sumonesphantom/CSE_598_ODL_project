"""Choose a classical restoration plan per perturbation on development data.

The hand-set plans in `restoration.PLANS` were never validated. Two of them
cost far more accuracy than the perturbation they undo: non-local-means
denoising loses 16 pp on noisy inputs because it strips the fine texture the
classifier reads, and Telea inpainting loses 15 pp on `background_removal`
because it is asked to invent up to 70 % of the frame from the rest.

This script scores every candidate in `restoration.CANDIDATES` on perturbed
*train*-split images and keeps the best one per perturbation, so the choice is
made on dev data rather than by hand (README known gap #3). `pass_through` is
always a candidate: where a perturbation destroys information instead of
transforming it, no classical inverse exists and leaving the image alone is
the honest plan.

A non-trivial plan must beat `pass_through` by at least --min-gain-pp to be
adopted. Repairs cost latency and can only be justified by a real improvement,
and at these sample sizes a sub-point win is noise.

Outputs (in --out):
  plan_summary.csv   per perturbation x candidate x level: dev accuracy
  plans.json         the selected plan per perturbation (read by run_restoration.py)
"""

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
from bolero.restoration import CANDIDATES, restore

PASS_THROUGH = "pass_through"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", help="Development split (never the evaluation split)")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--perturbations", nargs="*", default=list(CANDIDATES), choices=list(CANDIDATES))
    parser.add_argument("--min-gain-pp", type=float, default=0.5,
                        help="Accuracy a plan must gain over pass_through to be adopted")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tuning")
    args = parser.parse_args()
    if args.split == "val":
        raise SystemExit("Plans must be tuned on development data, not the val (evaluation) split.")

    classifier = FrozenClassifier(device=args.device)
    depth_cache = DepthCache()
    predictor = BatchPredictor(classifier, args.batch_size)

    records = list_images(args.split, per_class=args.per_class, seed=args.seed)
    for record in tqdm(records, desc=f"{args.split} images"):
        image = load_image(record.path)
        base = {"image_id": record.image_id, "label_index": record.label_index}
        for name in args.perturbations:
            perturbation = PERTURBATIONS[name]
            depth = depth_cache.get(record, image) if perturbation.needs_depth else None
            for level in LEVELS:
                x = perturbation(image, level, depth, rng_for(record.image_id, name, level, args.seed))
                for variant, plan in CANDIDATES[name].items():
                    predictor.add({**base, "perturbation": name, "level": level, "variant": variant},
                                  restore(x, name, {name: plan}))
    predictor.flush()

    df = pd.DataFrame(predictor.rows)
    summary = (df.groupby(["perturbation", "variant", "level"])["correct"].mean()
               .unstack("level").reset_index())
    summary.columns = [*summary.columns[:2], *[f"acc_L{c}" for c in summary.columns[2:]]]
    summary["acc_mean"] = summary[[c for c in summary.columns if c.startswith("acc_L")]].mean(axis=1)
    baseline = summary[summary["variant"] == PASS_THROUGH].set_index("perturbation")["acc_mean"]
    summary["gain_pp"] = 100.0 * (summary["acc_mean"] - summary["perturbation"].map(baseline))
    save(summary.sort_values(["perturbation", "acc_mean"], ascending=[True, False]),
         args.out / "plan_summary.csv")

    chosen, rows = {}, []
    for name, g in summary.groupby("perturbation"):
        best = g.sort_values("gain_pp", ascending=False).iloc[0]
        variant = best["variant"] if best["gain_pp"] >= args.min_gain_pp else PASS_THROUGH
        chosen[name] = variant
        rows.append({"perturbation": name, "chosen": variant,
                     "best_candidate": best["variant"], "best_gain_pp": best["gain_pp"],
                     "previous": "inpaint_always" if name != "gaussian_noise" else "nlm_0.90"})
    (args.out / "plans.json").write_text(json.dumps({
        "plans": chosen, "split": args.split, "images": len(records),
        "per_class": args.per_class, "seed": args.seed, "min_gain_pp": args.min_gain_pp,
    }, indent=2))

    print(f"\n{len(df):,} evaluations on {len(records)} {args.split} images")
    print(summary[["perturbation", "variant", "acc_mean", "gain_pp"]]
          .sort_values(["perturbation", "gain_pp"], ascending=[True, False]).round(4).to_string(index=False))
    print(f"\nSelected (must beat {PASS_THROUGH} by {args.min_gain_pp} pp):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
