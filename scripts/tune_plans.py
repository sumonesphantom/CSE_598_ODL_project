
"""Choose a classical restoration plan per perturbation on development data.

Every candidate is evaluated on perturbed train-split images. The best plan is
selected per perturbation, but it must beat pass-through by at least the
configured minimum gain before it is adopted.

The evaluation split is deliberately excluded from plan selection.

Outputs:
  plan_summary.csv   accuracy per perturbation, candidate, and severity
  plans.json         selected candidate per perturbation
"""

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Development split used for plan selection; do not use val.",
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        default=list(CANDIDATES),
        choices=list(CANDIDATES),
    )
    parser.add_argument(
        "--min-gain-pp",
        type=float,
        default=0.5,
        help="Minimum accuracy gain over pass-through required to adopt a plan.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tuning")
    args = parser.parse_args()

    if args.split.lower() == "val":
        raise SystemExit(
            "Plans must be tuned on development data, not the val evaluation split."
        )
    if args.per_class < 1:
        raise SystemExit("--per-class must be at least 1.")
    if args.min_gain_pp < 0:
        raise SystemExit("--min-gain-pp cannot be negative.")

    classifier = FrozenClassifier(device=args.device)
    depth_cache = DepthCache()
    predictor = BatchPredictor(classifier, args.batch_size)

    records = list_images(args.split, per_class=args.per_class, seed=args.seed)

    for record in tqdm(records, desc=f"{args.split} images"):
        image = load_image(record.path)
        base = {
            "image_id": record.image_id,
            "label_index": record.label_index,
        }

        for name in args.perturbations:
            perturbation = PERTURBATIONS[name]
            depth = (
                depth_cache.get(record, image)
                if perturbation.needs_depth
                else None
            )

            for level in LEVELS:
                x = perturbation(
                    image,
                    level,
                    depth,
                    rng_for(record.image_id, name, level, args.seed),
                )

                for variant, plan in CANDIDATES[name].items():
                    predictor.add(
                        {
                            **base,
                            "perturbation": name,
                            "level": level,
                            "variant": variant,
                        },
                        restore(x, name, {name: plan}),
                    )

    predictor.flush()

    df = pd.DataFrame(predictor.rows)
    if df.empty:
        raise RuntimeError("No candidate evaluations were produced.")

    summary = (
        df.groupby(["perturbation", "variant", "level"])["correct"]
        .mean()
        .unstack("level")
        .reset_index()
    )
    summary.columns = [
        *summary.columns[:2],
        *[f"acc_L{level}" for level in summary.columns[2:]],
    ]

    level_columns = [
        column for column in summary.columns if column.startswith("acc_L")
    ]
    summary["acc_mean"] = summary[level_columns].mean(axis=1)

    baseline = (
        summary[summary["variant"] == PASS_THROUGH]
        .set_index("perturbation")["acc_mean"]
    )
    summary["gain_pp"] = 100.0 * (
        summary["acc_mean"] - summary["perturbation"].map(baseline)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    save(
        summary.sort_values(
            ["perturbation", "acc_mean"],
            ascending=[True, False],
        ),
        args.out / "plan_summary.csv",
    )

    chosen = {}
    selection_rows = []

    for name, group in summary.groupby("perturbation"):
        best = group.sort_values("gain_pp", ascending=False).iloc[0]

        variant = (
            best["variant"]
            if best["gain_pp"] >= args.min_gain_pp
            else PASS_THROUGH
        )
        chosen[name] = variant

        selection_rows.append(
            {
                "perturbation": name,
                "chosen": variant,
                "best_candidate": best["variant"],
                "best_gain_pp": best["gain_pp"],
                "previous": (
                    "nlm_0.90"
                    if name == "gaussian_noise"
                    else "inpaint_always"
                ),
            }
        )

    plan_data = {
        "plans": chosen,
        "split": args.split,
        "images": len(records),
        "per_class": args.per_class,
        "seed": args.seed,
        "min_gain_pp": args.min_gain_pp,
    }

    (args.out / "plans.json").write_text(
        json.dumps(plan_data, indent=2),
        encoding="utf-8",
    )

    print(f"\n{len(df):,} candidate evaluations on {len(records)} {args.split} images")
    print(
        summary[
            ["perturbation", "variant", "acc_mean", "gain_pp"]
        ]
        .sort_values(
            ["perturbation", "gain_pp"],
            ascending=[True, False],
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nSelected plans (minimum gain over pass-through: "
        f"{args.min_gain_pp} pp):"
    )
    print(pd.DataFrame(selection_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
