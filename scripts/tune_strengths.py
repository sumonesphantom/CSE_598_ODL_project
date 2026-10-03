"""Choose cue and baseline strengths on development data (Imagenette train split).

The restoration study (run_restoration.py) evaluates on the val split, so its
strengths must be chosen elsewhere. This script sweeps each depth cue and each
depth-free baseline over a strength grid on perturbed and clean *train* images,
then picks one strength per method:

  the strength with the best mean accuracy over all perturbation x severity
  conditions, among strengths that cost at most --max-clean-drop pp of
  accuracy on clean images.

As in the main study, depth for the cues is re-estimated on the perturbed input.
One strength per method (not per perturbation) is chosen, because a deployed
system doesn't know which perturbation it is facing.

Outputs (in --out):
  records.csv.gz   one row per image x condition x method x strength
  summary.csv      per method x strength: dev accuracy on perturbed and clean inputs
  strengths.json   the selected strength per method (read by run_restoration.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.baselines import BASELINES, apply_baseline
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.cues import CUES, apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache, DepthEstimator
from bolero.detection import CLEAN
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import prefixed, save
from bolero.tuning import method_key, select_strengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", help="Development split (never the evaluation split)")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--grid", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--cues", nargs="*", default=list(CUES), choices=list(CUES))
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--perturbations", nargs="*", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    parser.add_argument("--max-clean-drop", type=float, default=1.0,
                        help="Max accuracy loss on clean dev images, in percentage points")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tuning")
    args = parser.parse_args()
    if args.split == "val":
        raise SystemExit("Strengths must be tuned on development data, not the val (evaluation) split.")

    classifier = FrozenClassifier(device=args.device)
    estimator = DepthEstimator(device=args.device)
    depth_cache = DepthCache(estimator)
    predictor = BatchPredictor(classifier, args.batch_size)
    conditions = [(CLEAN, 0)] + [(name, level) for name in args.perturbations for level in LEVELS]

    records = list_images(args.split, per_class=args.per_class, seed=args.seed)
    for record in tqdm(records, desc=f"{args.split} images"):
        clean = load_image(record.path)
        clean_depth = depth_cache.get(record, clean)
        base = {"image_id": record.image_id, "label_index": record.label_index}
        for name, level in conditions:
            if name == CLEAN:
                x, depth = clean, clean_depth
            else:
                x = PERTURBATIONS[name](clean, level, clean_depth, rng_for(record.image_id, name, level, args.seed))
                depth = estimator(x)
            cond = {**base, "perturbation": name, "level": level}
            predictor.add({**cond, "method": "none", "strength": 0.0}, x)
            for strength in args.grid:
                for cue in args.cues:
                    predictor.add({**cond, "method": method_key("cue", cue), "strength": strength},
                                  apply_cue(cue, x, depth, strength))
                for baseline in args.baselines:
                    predictor.add({**cond, "method": method_key("baseline", baseline), "strength": strength},
                                  apply_baseline(baseline, x, strength))
    predictor.flush()

    df = prefixed(predictor.rows, "out")
    save(df, args.out / "records.csv.gz")
    summary = summarize(df)
    save(summary, args.out / "summary.csv")

    chosen = select_strengths(summary, args.max_clean_drop)
    (args.out / "strengths.json").write_text(json.dumps({
        "strengths": {r.method: float(r.strength) for r in chosen.itertuples()},
        "constraint_met": {r.method: bool(r.constraint_met) for r in chosen.itertuples()},
        "split": args.split, "images": len(records), "per_class": args.per_class, "seed": args.seed,
        "grid": args.grid, "max_clean_drop_pp": args.max_clean_drop,
        "perturbations": args.perturbations,
    }, indent=2))
    print(chosen[["method", "strength", "acc_perturbed", "gain_pp", "clean_drop_pp", "constraint_met"]]
          .round(4).to_string(index=False))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per method x strength: mean accuracy over perturbed conditions, and accuracy on clean inputs."""
    perturbed = df[df["perturbation"] != CLEAN]
    clean = df[df["perturbation"] == CLEAN]
    none_pert = perturbed[perturbed["method"] == "none"]["out_correct"].mean()
    none_clean = clean[clean["method"] == "none"]["out_correct"].mean()
    rows = []
    for (method, strength), g in df[df["method"] != "none"].groupby(["method", "strength"], sort=False):
        pert = g[g["perturbation"] != CLEAN]
        acc_pert = pert.groupby(["perturbation", "level"])["out_correct"].mean().mean()
        acc_clean = g[g["perturbation"] == CLEAN]["out_correct"].mean()
        rows.append({
            "method": method, "strength": strength,
            "acc_perturbed": acc_pert, "acc_perturbed_none": none_pert,
            "gain_pp": 100.0 * (acc_pert - none_pert),
            "acc_clean": acc_clean, "acc_clean_none": none_clean,
            "clean_drop_pp": 100.0 * (none_clean - acc_clean),
            "n": len(g),
        })
    return pd.DataFrame(rows).sort_values(["method", "strength"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
