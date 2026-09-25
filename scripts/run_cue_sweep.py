"""Cue strength sweep on clean images (extends the thermal / border pilot to every cue).

Each depth-derived cue is applied to clean validation images at strengths
0.0 .. 1.0. A cue that is safe to use as a restoration step should leave the
prediction unchanged on clean inputs; the sweep measures where that stops
being true.

Outputs (in --out):
  records.csv.gz   one row per image x cue x strength
  summary.csv      per cue x strength: accuracy, prediction-change rate vs. clean
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.cues import CUES, apply_cue
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.metrics import compare
from bolero.records import prefixed, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--strengths", type=float, nargs="*", default=np.round(np.linspace(0, 1, 11), 2).tolist())
    parser.add_argument("--cues", nargs="*", default=list(CUES), choices=list(CUES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "cue_sweep")
    args = parser.parse_args()

    classifier = FrozenClassifier(device=args.device)
    depth_cache = DepthCache()
    clean = BatchPredictor(classifier, args.batch_size)
    cued = BatchPredictor(classifier, args.batch_size)

    for record in tqdm(list_images("val", per_class=args.per_class, seed=args.seed), desc="images"):
        image = load_image(record.path)
        depth = depth_cache.get(record, image)
        meta = {"image_id": record.image_id, "wnid": record.wnid,
                "class_name": record.class_name, "label_index": record.label_index}
        clean.add(meta, image)
        for cue in args.cues:
            for strength in args.strengths:
                cued.add({**meta, "cue": cue, "strength": strength}, apply_cue(cue, image, depth, strength))
    clean.flush()
    cued.flush()

    clean_df = prefixed(clean.rows, "clean").drop(columns=["wnid", "class_name", "label_index"])
    df = prefixed(cued.rows, "cue").merge(clean_df, on="image_id")
    save(df, args.out / "records.csv.gz")
    summary = compare(df, "clean", "cue", ["cue", "strength"])
    save(summary, args.out / "summary.csv")
    print(summary.pivot(index="strength", columns="cue", values="changed_rate").round(3).to_string())


if __name__ == "__main__":
    main()
