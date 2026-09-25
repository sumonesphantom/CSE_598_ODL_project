"""Stage 2 - train the signature-based perturbation detector.

Trains on perturbed copies of Imagenette *train* images and evaluates on *val*
images, so the detector never sees the images used in the restoration study.

Outputs (in --out):
  detector.joblib        the fitted detector
  eval_predictions.csv   one row per val sample: true / predicted / decided label
  eval_summary.csv       per perturbation: detection rate, routed-to-restoration rate
  headline.json          top-1 accuracy, clean false-alarm rate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.config import RESULTS_DIR, SEED
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.detection import CLEAN, PerturbationDetector, extract_features
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import save


def build(split: str, per_class: int, seed: int, depth_cache: DepthCache):
    features, labels, meta = [], [], []
    for record in tqdm(list_images(split, per_class=per_class, seed=seed), desc=f"{split} features"):
        image = load_image(record.path)
        depth = depth_cache.get(record, image)
        # Clean examples are repeated once per level so "clean" is not swamped.
        for level in LEVELS:
            features.append(extract_features(image))
            labels.append(CLEAN)
            meta.append({"image_id": record.image_id, "perturbation": CLEAN, "level": 0})
        for name, perturbation in PERTURBATIONS.items():
            for level in LEVELS:
                out = perturbation(image, level, depth, rng_for(record.image_id, name, level, seed))
                features.append(extract_features(out))
                labels.append(name)
                meta.append({"image_id": record.image_id, "perturbation": name, "level": level})
    return np.stack(features), np.array(labels), pd.DataFrame(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "detector")
    args = parser.parse_args()

    depth_cache = DepthCache()
    x_train, y_train, _ = build("train", args.train_per_class, args.seed, depth_cache)
    detector = PerturbationDetector(threshold=args.threshold).fit(x_train, y_train)
    detector.save(args.out / "detector.joblib")

    x_val, y_val, meta = build("val", args.val_per_class, args.seed + 1, depth_cache)
    proba = detector.predict_proba(x_val)
    best = proba.argmax(axis=1)
    predicted = detector.classes[best]
    confident = (predicted != CLEAN) & (proba[np.arange(len(best)), best] >= args.threshold)
    decided = np.where(confident, predicted, CLEAN)

    evals = meta.assign(true=y_val, predicted=predicted, probability=proba.max(axis=1), decided=decided)
    save(evals, args.out / "eval_predictions.csv")

    summary = (
        evals.groupby(["perturbation", "level"])
        .apply(lambda g: pd.Series({
            "n": len(g),
            "top1_correct": (g["predicted"] == g["true"]).mean(),
            "routed_correctly": (g["decided"] == g["true"]).mean(),
            "routed_to_restoration": (g["decided"] != CLEAN).mean(),
        }), include_groups=False)
        .reset_index()
    )
    save(summary, args.out / "eval_summary.csv")

    is_clean = evals["true"] == CLEAN
    headline = {
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "threshold": args.threshold,
        "top1_accuracy": float((evals["predicted"] == evals["true"]).mean()),
        "routed_correctly": float((evals["decided"] == evals["true"]).mean()),
        "clean_false_alarm_rate": float((evals.loc[is_clean, "decided"] != CLEAN).mean()),
        "perturbed_detected_rate": float((evals.loc[~is_clean, "decided"] != CLEAN).mean()),
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
