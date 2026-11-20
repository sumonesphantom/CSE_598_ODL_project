"""Physics vs generative vs real degradation on the same frozen detector (TODO: "which synthetic
degradation predicts real-data performance best?").

Image sets, all scored by frozen COCO DETR, each with no preprocessing and each baseline:
  coco/clean                   300 clean COCO val2017 traffic scenes (weather_data.load_coco)
  coco/physics:<effect>/<L>    the same images through the five physics weather effects, levels 1-3
  coco/generative:<cond>/<L>   the same images edited by InstructPix2Pix (make_generative_weather.py)
  xwod/<cond>, dawn/<cond>     real heavy rain and dust (XWOD v1.1 test split; DAWN rain / sand)

Generative edits whose CLIP similarity to the clean image falls below a threshold are treated
as content changes ("label did not survive") and reported separately. The default threshold
is the 5th percentile of the physics effects' CLIP similarity on the same images: physics
effects never move or remove objects, so an edit that changes the image more than nearly all
of them is assumed to have changed the scene. The detector under test is never used to filter.

How well a synthetic set predicts each real set of its condition (rain / dust):
  map_gap        |mAP50(synthetic) - mAP50(real)|: does it reproduce the real difficulty?
  class_rank     Spearman over classes of AP50: do the same classes suffer?
  baseline_signs how many of the 3 baselines change mAP50 in the same direction as on real data
  same_best      does it pick the same best method (none or a baseline) as real data?
  kid_to_real    CLIP kernel distance to the real images (lower = looks more like real weather)
The other real dataset of the same condition is scored the same way, as the reference.

Outputs (in --out): sets/<key>.pkl.gz (detections + CLIP embeddings, reused on re-runs),
summary.csv, predict.csv, kid.csv, headline.json.

Usage:
  python scripts/run_degradation_compare.py                 # detect what is missing, then analyze
  python scripts/run_degradation_compare.py --stage analyze --clip-threshold 0.8
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.baselines import BASELINES, apply_baseline
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.data import to_pil
from bolero.degradation_compare import bootstrap_map50, class_ap50, image_matches, kid, spearman
from bolero.detection_eval import FrozenDetector, evaluate
from bolero.generative import ClipEmbedder
from bolero.perturbations import LEVELS, WEATHER, rng_for
from bolero.tuning import load_strengths, method_key
from bolero.weather_data import CLASSES, load_coco, load_dawn, load_xwod, open_rgb

CONDITIONS = ("rain", "dust")


def load_real(datasets) -> dict:
    """{"xwod__rain": [DetectionSample, ...], ...} for the real sets that are on disk."""
    real = {}
    for ds in datasets:
        try:
            samples = load_xwod(split="test") if ds == "xwod" else load_dawn()
        except FileNotFoundError as e:
            print(f"skipping {ds}: {e}")
            continue
        for cond in CONDITIONS:
            real[f"{ds}__{cond}"] = [s for s in samples if s.condition == cond]
    return real


def preprocess(method: str, x: np.ndarray, strengths: dict) -> Image.Image:
    if method == "none":
        return to_pil(x)
    name = method.split(":", 1)[1]
    return to_pil(apply_baseline(name, x, strengths[method_key("baseline", name)]))


def set_specs(coco, real, args) -> list[dict]:
    """Every image set: key, kind, condition, level, and how to load its images."""
    specs = [{"key": "coco__clean", "kind": "clean", "condition": "clean", "level": 0}]
    for name, p in WEATHER.items():
        for level in LEVELS:
            specs.append({"key": f"coco__physics_{name}__{level}", "kind": "physics", "effect": name,
                          "condition": p.family, "level": level})
    manifest = args.generative / "manifest.csv"
    if manifest.exists():
        m = pd.read_csv(manifest)
        for (cond, level), g in m.groupby(["condition", "level"]):
            paths = dict(zip(g["image_id"], g["path"]))
            if all(s.image_id in paths for s in coco):
                specs.append({"key": f"coco__generative_{cond}__{level}", "kind": "generative",
                              "effect": f"generative_{cond}", "condition": cond, "level": int(level), "paths": paths})
    else:
        print(f"no {manifest}; generative sets skipped (run make_generative_weather.py --source coco)")
    for key in real:
        specs.append({"key": key, "kind": "real", "condition": key.split("__")[1], "level": 0})
    return specs


def load_images(spec, coco, real, depths, depth_estimator):
    """Yield (image_id, float RGB array) for a set; EXIF orientation applied throughout."""
    if spec["kind"] == "real":
        for s in real[spec["key"]]:
            yield s.image_id, np.asarray(open_rgb(s.path), np.float32) / 255.0
        return
    for s in coco:
        path = spec["paths"][s.image_id] if spec["kind"] == "generative" else s.path
        x = np.asarray(open_rgb(path), np.float32) / 255.0
        if spec["kind"] == "physics":
            p = WEATHER[spec["effect"]]
            depth = None
            if p.needs_depth:
                if s.image_id not in depths:
                    depths[s.image_id] = depth_estimator()(x)
                depth = depths[s.image_id]
            x = p(x, spec["level"], depth, rng_for(s.image_id, spec["effect"], spec["level"], SEED))
        yield s.image_id, x


def detect(specs, coco, real, args, methods, strengths) -> None:
    sets_dir = args.out / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (sets_dir / f"{s['key']}.pkl.gz").exists()]
    print(f"{len(specs)} sets, {len(todo)} to detect")
    if not todo:
        return
    detector = FrozenDetector(args.model, device=args.device, min_score=args.min_score)
    clip = ClipEmbedder(args.device)
    depths, holder = {}, {}

    def depth_estimator():
        if "d" not in holder:
            from bolero.depth import DepthEstimator
            holder["d"] = DepthEstimator(device=args.device)
        return holder["d"]

    for spec in todo:
        ids, dets, views = [], {m: [] for m in methods}, []
        total = len(real[spec["key"]]) if spec["kind"] == "real" else len(coco)
        for image_id, x in tqdm(load_images(spec, coco, real, depths, depth_estimator), desc=spec["key"], total=total):
            ids.append(image_id)
            for m in methods:
                dets[m].append(detector(preprocess(m, x, strengths)))
            views.append(to_pil(x))
        emb = clip(views)
        with gzip.open(sets_dir / f"{spec['key']}.pkl.gz", "wb") as f:
            pickle.dump({"ids": ids, "dets": dets, "clip": emb}, f)


def load_set(args, key):
    with gzip.open(args.out / "sets" / f"{key}.pkl.gz", "rb") as f:
        return pickle.load(f)


def analyze(specs, coco, real, args, methods) -> None:
    data = {s["key"]: load_set(args, s["key"]) for s in specs}
    clean = data["coco__clean"]
    assert clean["ids"] == [s.image_id for s in coco], "clean set does not match the COCO sample"
    gt = {"coco": ([s.boxes for s in coco], [s.labels for s in coco])}
    gt.update({k: ([s.boxes for s in v], [s.labels for s in v]) for k, v in real.items()})
    for k in real:
        assert data[k]["ids"] == [s.image_id for s in real[k]], f"{k} does not match its loader"

    # Content-survival threshold, from how far the physics effects move CLIP embeddings.
    sims = {k: (d["clip"] * clean["clip"]).sum(1) for k, d in data.items()
            if k.startswith("coco__") and k != "coco__clean"}
    physics_sims = np.concatenate([v for k, v in sims.items() if "__physics_" in k])
    threshold = float(np.percentile(physics_sims, 5)) if args.clip_threshold is None else args.clip_threshold
    print(f"CLIP content threshold {threshold:.3f}"
          f" ({'5th percentile of physics similarity' if args.clip_threshold is None else 'given'})")

    def matches(key, method, idx=None):
        boxes, labels = gt["coco"] if key.startswith("coco__") else gt[key]
        dets = data[key]["dets"][method]
        if idx is not None:
            boxes, labels, dets = [boxes[i] for i in idx], [labels[i] for i in idx], [dets[i] for i in idx]
        return image_matches(boxes, labels, dets), (boxes, labels, dets)

    rows, profiles = [], {}
    for spec in specs:
        key = spec["key"]
        variants = [("all", None)]
        if spec["kind"] == "generative":
            variants.append(("kept", np.where(sims[key] >= threshold)[0]))
        for variant, idx in variants:
            none_m, _ = matches(key, "none", idx)
            ref = matches("coco__clean", "none", idx)[0] if spec["kind"] in ("physics", "generative") else None
            for m in methods:
                per, (boxes, labels, dets) = matches(key, m, idx)
                value, low, high = bootstrap_map50(per, args.n_boot, args.seed)
                row = {"set": key, "kind": spec["kind"], "effect": spec.get("effect", spec["kind"]),
                       "condition": spec["condition"], "level": spec["level"], "variant": variant, "method": m,
                       "images": len(boxes), "boxes": int(sum(len(b) for b in boxes)),
                       "map50": value, "map50_low": low, "map50_high": high,
                       "map50_95": evaluate(boxes, labels, dets)["map50_95"]}
                if m == "none" and ref is not None:
                    d, dl, dh = bootstrap_map50(per, args.n_boot, args.seed, reference=ref)
                    row.update(drop_vs_clean=d, drop_low=dl, drop_high=dh)
                if m != "none":
                    g, gl, gh = bootstrap_map50(per, args.n_boot, args.seed, reference=none_m)
                    row.update(gain_vs_none=g, gain_low=gl, gain_high=gh)
                if spec["kind"] in ("physics", "generative"):
                    row["clip_sim"] = float(np.mean(sims[key] if idx is None else sims[key][idx]))
                rows.append(row)
                if m == "none":
                    profiles[(key, variant)] = class_ap50(per)
    summary = pd.DataFrame(rows)

    # Appearance: CLIP kernel distance from each COCO set to each real set of its condition.
    kid_rows = []
    for real_key in real:
        cond = real_key.split("__")[1]
        for spec in specs:
            if spec["key"] != real_key and spec["condition"] in (cond, "clean"):
                kid_rows.append({"real": real_key, "set": spec["key"],
                                 "kid_to_real_x1000": 1000 * kid(data[spec["key"]]["clip"], data[real_key]["clip"])})
    kid_df = pd.DataFrame(kid_rows).set_index(["real", "set"])["kid_to_real_x1000"]

    # Prediction: how well each synthetic set reproduces each real set of its condition.
    s = summary.set_index(["set", "variant", "method"])
    mapv = lambda key, variant, m: s.loc[(key, variant, m), "map50"]
    pred_rows = []
    for real_key in real:
        cond = real_key.split("__")[1]
        real_gain = np.array([mapv(real_key, "all", m) - mapv(real_key, "all", "none") for m in methods[1:]])
        real_best = methods[int(np.argmax([mapv(real_key, "all", m) for m in methods]))]
        real_prof = profiles[(real_key, "all")]
        for spec in specs:
            # The other real dataset of the same condition is the reference: a synthetic set
            # cannot be expected to predict one real set better than another real set does.
            if spec["condition"] != cond or spec["kind"] == "clean" or spec["key"] == real_key:
                continue
            for variant in (["all", "kept"] if spec["kind"] == "generative" else ["all"]):
                none = mapv(spec["key"], variant, "none")
                gain = np.array([mapv(spec["key"], variant, m) - none for m in methods[1:]])
                best = methods[int(np.argmax([mapv(spec["key"], variant, m) for m in methods]))]
                prof = profiles[(spec["key"], variant)]
                cls = [c for c in CLASSES if not np.isnan(real_prof[c]) and not np.isnan(prof[c])]
                pred_rows.append({
                    "real": real_key, "kind": spec["kind"], "effect": spec.get("effect", spec["key"]),
                    "level": spec["level"], "variant": variant,
                    "images": int(s.loc[(spec["key"], variant, "none"), "images"]),
                    "map50": none, "real_map50": mapv(real_key, "all", "none"),
                    "map_gap": abs(none - mapv(real_key, "all", "none")),
                    "class_rank": spearman([prof[c] for c in cls], [real_prof[c] for c in cls]),
                    "baseline_signs": int((np.sign(gain) == np.sign(real_gain)).sum()),
                    "same_best": best == real_best, "best": best, "real_best": real_best,
                    "kid_to_real_x1000": float(kid_df.loc[(real_key, spec["key"])]),
                    "clean_kid_to_real_x1000": float(kid_df.loc[(real_key, "coco__clean")]),
                })
    predict = pd.DataFrame(pred_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    predict.to_csv(args.out / "predict.csv", index=False)
    kid_df.reset_index().to_csv(args.out / "kid.csv", index=False)
    (args.out / "headline.json").write_text(json.dumps({
        "coco_images": len(coco), "real_images": {k: len(v) for k, v in real.items()},
        "clip_threshold": threshold, "n_boot": args.n_boot, "model": args.model, "methods": methods,
        "clean_map50": float(mapv("coco__clean", "all", "none")),
    }, indent=2))
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        cols = ["set", "variant", "images", "map50", "map50_low", "map50_high", "drop_vs_clean", "clip_sim"]
        print(summary[summary.method == "none"][[c for c in cols if c in summary]].round(3).to_string(index=False))
        print(predict.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="COCO images (must match make_generative_weather.py)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--generative", type=Path, default=DATA_DIR / "generative" / "coco")
    parser.add_argument("--real", nargs="+", default=["xwod", "dawn"], choices=["xwod", "dawn"])
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--strengths", type=Path, default=RESULTS_DIR / "tuning" / "strengths.json")
    parser.add_argument("--model", default="detr")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--clip-threshold", type=float, default=None,
                        help="Content-survival threshold; default: 5th percentile of physics CLIP similarity")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--stage", choices=["all", "detect", "analyze"], default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "degradation_compare")
    args = parser.parse_args()

    coco = load_coco(n=args.n, seed=args.seed)
    real = load_real(args.real)
    methods = ["none"] + [f"baseline:{b}" for b in args.baselines]
    strengths, _ = load_strengths(args.strengths)
    specs = set_specs(coco, real, args)
    if args.stage in ("all", "detect"):
        detect(specs, coco, real, args, methods, strengths)
    if args.stage in ("all", "analyze"):
        analyze(specs, coco, real, args, methods)


if __name__ == "__main__":
    main()
