"""Smoke test: what do the distortions look like, and how much do they hurt the classifier?

For a few val images per class, applies every distortion at 3 severities,
classifies each result with the frozen ResNet-50, and renders contact sheets
(one row per distortion: clean, level 1, level 2, level 3).

Sets (--sets):
  geometric   Imagy-style warps from bolero.distortions (swirl, bulge, waves, ...)
  pipeline    the 14 study perturbations plus the held-out stress tests

Outputs (in --out):
  sheet_<set>_<image_id>.png   contact sheets for --sheets sample images
  summary.csv                  per set x distortion x level: accuracy and prediction change rate
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import BatchPredictor, FrozenClassifier
from bolero.config import RESULTS_DIR, SEED
from bolero.data import list_images, load_image
from bolero.depth import DepthCache
from bolero.distortions import DISTORTIONS
from bolero.perturbations import ALL_PERTURBATIONS, LEVELS, rng_for
from bolero.plotting import apply_style

SETS = {"geometric": DISTORTIONS, "pipeline": ALL_PERTURBATIONS}


def contact_sheet(rows: dict[str, list[np.ndarray]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 1 + len(LEVELS), figsize=(1.9 * (1 + len(LEVELS)), 1.45 * len(rows)),
                             squeeze=False, layout="constrained")
    for i, (name, images) in enumerate(rows.items()):
        for j, image in enumerate(images):
            ax = axes[i, j]
            ax.imshow(np.clip(image, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("clean" if j == 0 else f"level {j}", fontsize=9)
        axes[i, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    parser.add_argument("--sheets", type=int, default=3, help="number of images to render contact sheets for")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "distortions")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()
    records = list_images("val", per_class=args.per_class, seed=args.seed)
    # Spread the sample sheets across classes (records are grouped by class).
    sheet_ids = {r.image_id for r in records[:: max(1, len(records) // max(1, args.sheets))][: args.sheets]}
    predictor = BatchPredictor(FrozenClassifier(device=args.device), args.batch_size)
    depth_cache = DepthCache() if "pipeline" in args.sets else None

    for record in tqdm(records, desc="images"):
        image = load_image(record.path)
        depth = depth_cache.get(record, image) if depth_cache else None
        meta = {"image_id": record.image_id, "class_name": record.class_name, "label_index": record.label_index}
        predictor.add({**meta, "set": "clean", "distortion": "clean", "level": 0}, image)
        for set_name in args.sets:
            rows = {}
            for name, distortion in SETS[set_name].items():
                outs = [distortion(image, level, depth, rng_for(record.image_id, name, level, args.seed))
                        for level in LEVELS]
                for level, out in zip(LEVELS, outs):
                    predictor.add({**meta, "set": set_name, "distortion": name, "level": level}, out)
                rows[name] = [image, *outs]
            if record.image_id in sheet_ids:
                contact_sheet(rows, f"{set_name}: {record.class_name} ({record.image_id})",
                              args.out / f"sheet_{set_name}_{record.image_id}.png")
    predictor.flush()

    df = pd.DataFrame(predictor.rows)
    clean = df[df["set"] == "clean"].set_index("image_id")
    df = df[df["set"] != "clean"].assign(
        changed=lambda d: d["pred_index"].to_numpy() != clean.loc[d["image_id"], "pred_index"].to_numpy())
    summary = (df.groupby(["set", "distortion", "level"], sort=False)
               .agg(n=("correct", "size"), acc=("correct", "mean"), changed_rate=("changed", "mean"),
                    true_prob=("true_prob", "mean"))
               .reset_index())
    summary["delta_pp"] = 100.0 * (summary["acc"] - clean["correct"].mean())
    summary.to_csv(args.out / "summary.csv", index=False)

    print(f"Clean accuracy {clean['correct'].mean():.1%} on {len(clean)} images")
    print(summary.pivot_table(index=["set", "distortion"], columns="level", values="delta_pp", sort=False)
          .round(1).to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
