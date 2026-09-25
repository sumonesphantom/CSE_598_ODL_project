"""Stage 1 - diagnostic sweep: how does each perturbation change the frozen classifier?

For every validation image: classify the clean image, apply each of the 14
perturbations at 3 severities, classify each result, and record prediction,
confidence, correctness, and image fidelity (PSNR/SSIM vs. clean).

Outputs (in --out):
  records.csv.gz   one row per image x perturbation x severity
  summary.csv      per perturbation x severity: accuracy, change rate, transitions
  headline.json    overall change rate and transition counts
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.metrics import compare, psnr, ssim
from bolero.perturbations import LEVELS, PERTURBATIONS, rng_for
from bolero.records import prefixed, save

KEYS = ["image_id", "wnid", "class_name", "label_index"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=None, help="images per class (default: all 3,925)")
    parser.add_argument("--perturbations", nargs="*", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "diagnostic")
    args = parser.parse_args()

    records = list_images("val", per_class=args.per_class, seed=args.seed)
    classifier = FrozenClassifier(device=args.device)
    depth_cache = DepthCache()
    clean = BatchPredictor(classifier, args.batch_size)
    perturbed = BatchPredictor(classifier, args.batch_size)

    started = time.perf_counter()
    for record in tqdm(records, desc="images"):
        image = load_image(record.path)
        depth = depth_cache.get(record, image)
        meta = {"image_id": record.image_id, "wnid": record.wnid,
                "class_name": record.class_name, "label_index": record.label_index}
        clean.add(meta, image)
        for name in args.perturbations:
            perturbation = PERTURBATIONS[name]
            for level in LEVELS:
                t0 = time.perf_counter()
                out = perturbation(image, level, depth, rng_for(record.image_id, name, level, args.seed))
                perturb_ms = 1000.0 * (time.perf_counter() - t0)
                perturbed.add({
                    **meta, "perturbation": name, "family": perturbation.family, "level": level,
                    "severity": perturbation.severities[level - 1], "perturb_ms": perturb_ms,
                    "psnr": psnr(image, out), "ssim": ssim(image, out),
                }, out)
    clean.flush()
    perturbed.flush()
    elapsed = time.perf_counter() - started

    clean_df = prefixed(clean.rows, "clean")
    pert_df = prefixed(perturbed.rows, "pert")
    df = pert_df.merge(clean_df.drop(columns=["wnid", "class_name", "label_index"]), on="image_id")
    save(df, args.out / "records.csv.gz")

    summary = compare(df, "clean", "pert", ["perturbation", "family", "level", "severity"])
    fidelity = df.groupby(["perturbation", "level"])[["psnr", "ssim"]].mean().reset_index()
    save(summary.merge(fidelity, on=["perturbation", "level"]), args.out / "summary.csv")
    save(compare(df, "clean", "pert", ["perturbation", "level", "class_name"]), args.out / "per_class.csv")

    overall = compare(df.assign(all="all"), "clean", "pert", ["all"]).iloc[0]
    headline = {
        "images": len(records),
        "records": len(df),
        "perturbations": len(args.perturbations),
        "levels": len(LEVELS),
        "clean_accuracy": float(clean_df["clean_correct"].mean()),
        "perturbed_accuracy": float(overall["acc_pert"]),
        "prediction_changed_rate": float(overall["changed_rate"]),
        "correct_to_wrong": int(overall["correct_to_wrong"]),
        "wrong_to_correct": int(overall["wrong_to_correct"]),
        "wall_minutes": elapsed / 60.0,
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
