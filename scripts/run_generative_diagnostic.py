"""Classifier drop under generative weather, side by side with the physics weather effects.

Reads the Imagenette edits from make_generative_weather.py (--source imagenette), classifies the
clean and edited image with the frozen ResNet-50, and reports the accuracy change per condition
and level, for all edits and for the edits whose content survived.

"Content survived" means CLIP similarity to the clean image at or above a threshold. By default
the threshold is the 5th percentile of the physics weather effects' CLIP similarity on the same
images (they never move or remove the object), so the classifier under test plays no part in
the filter. Intervals are 95 % bootstrap over images.

Outputs (in --out): records.csv.gz (one row per edit), summary.csv, headline.json.

Usage:
  python scripts/run_generative_diagnostic.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.classifier import FrozenClassifier
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.data import list_images, load_image, to_pil
from bolero.depth import DepthCache
from bolero.generative import ClipEmbedder
from bolero.perturbations import LEVELS, WEATHER, rng_for


def bootstrap_delta(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean accuracy change (pp) with a 95 % CI, resampling images."""
    per = (df["edit_correct"].astype(float) - df["clean_correct"].astype(float)).groupby(df["image_id"]).mean().to_numpy()
    boots = np.random.default_rng(seed).choice(per, (n_boot, len(per))).mean(1) * 100
    return float(per.mean() * 100), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def classify(classifier, images: list[np.ndarray], labels: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        out.append(classifier.predict(images[i:i + batch], labels[i:i + batch]).index)
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generative", type=Path, default=DATA_DIR / "generative" / "imagenette")
    parser.add_argument("--clip-threshold", type=float, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "generative_imagenette")
    args = parser.parse_args()

    edits = pd.read_csv(args.generative / "manifest.csv")
    ids = edits["image_id"].unique()
    records = {r.image_id: r for r in list_images("val", per_class=None, seed=args.seed) if r.image_id in set(ids)}
    missing = set(ids) - set(records)
    if missing:
        raise SystemExit(f"{len(missing)} manifest images not found in Imagenette val")

    clip = ClipEmbedder(args.device)
    depth = DepthCache()
    clean_imgs, clean_emb, physics_sims = {}, {}, []
    for image_id in tqdm(ids, desc="clean + physics"):
        r = records[image_id]
        x = load_image(r.path)
        clean_imgs[image_id] = x
        d = depth.get(r, x)
        views = [to_pil(x)] + [to_pil(p(x, lv, d, rng_for(image_id, name, lv, args.seed)))
                               for name, p in WEATHER.items() for lv in LEVELS]
        emb = clip(views)
        clean_emb[image_id] = emb[0]
        physics_sims += list(emb[1:] @ emb[0])
    threshold = float(np.percentile(physics_sims, 5)) if args.clip_threshold is None else args.clip_threshold
    # Recompute similarity here rather than trusting the manifest, so both sides use one embedder.
    edited = [load_image(Path(p)) for p in edits["path"]]
    emb = clip([to_pil(x) for x in edited])
    edits["clip_sim"] = [float(e @ clean_emb[i]) for e, i in zip(emb, edits["image_id"])]
    edits["kept"] = edits["clip_sim"] >= threshold

    classifier = FrozenClassifier(device=args.device)
    labels = np.array([records[i].label_index for i in ids])
    clean_pred = dict(zip(ids, classify(classifier, [clean_imgs[i] for i in ids], labels)))
    edit_pred = classify(classifier, edited, edits["image_id"].map(lambda i: records[i].label_index).to_numpy())
    edits["clean_correct"] = [clean_pred[i] == records[i].label_index for i in edits["image_id"]]
    edits["edit_correct"] = edit_pred == edits["image_id"].map(lambda i: records[i].label_index).to_numpy()
    edits["prediction_changed"] = edit_pred != edits["image_id"].map(clean_pred).to_numpy()

    rows = []
    for (cond, level), g in edits.groupby(["condition", "level"]):
        for variant, sub in (("all", g), ("kept", g[g["kept"]])):
            d, lo, hi = bootstrap_delta(sub, args.n_boot, args.seed) if len(sub) else (np.nan,) * 3
            rows.append({"condition": cond, "level": level, "variant": variant, "images": len(sub),
                         "kept_rate": float(g["kept"].mean()), "clip_sim": float(sub["clip_sim"].mean()),
                         "acc_clean": float(sub["clean_correct"].mean()), "acc_edit": float(sub["edit_correct"].mean()),
                         "delta_pp": d, "ci_low_pp": lo, "ci_high_pp": hi,
                         "changed_rate": float(sub["prediction_changed"].mean())})
    summary = pd.DataFrame(rows)
    pooled = {f"{cond}/{variant}": bootstrap_delta(g if variant == "all" else g[g["kept"]], args.n_boot, args.seed)
              for cond, g in edits.groupby("condition") for variant in ("all", "kept")}

    args.out.mkdir(parents=True, exist_ok=True)
    edits.drop(columns=["clean_path"]).to_csv(args.out / "records.csv.gz", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "headline.json").write_text(json.dumps({
        "images": len(ids), "edits": len(edits), "clip_threshold": threshold,
        "clip_threshold_source": "5th percentile of physics similarity" if args.clip_threshold is None else "given",
        "clean_accuracy": float(np.mean([clean_pred[i] == records[i].label_index for i in ids])),
        "pooled_delta_pp": {k: {"delta": v[0], "ci": [v[1], v[2]]} for k, v in pooled.items()},
    }, indent=2))
    with pd.option_context("display.width", 200):
        print(f"CLIP threshold {threshold:.3f}")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
