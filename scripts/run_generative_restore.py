"""Generative restoration on real rain and dust, with a hallucination check for every restorer.

Each real DAWN / XWOD image is restored by InstructPix2Pix ("remove the rain" / "remove the
sandstorm and dust haze", chosen from the image's known condition) at one or more edit
strengths. The frozen COCO DETR then runs on the raw image, on each restored version, and on
the preprocessing baselines, and every method is scored two ways:

  1. mAP50 / mAP50-95 against the ground truth, with the paired change vs. the raw image
     and a 95% image-bootstrap CI (does restoration help on average?)
  2. hallucination counts vs. the raw image at an operating threshold (bolero/hallucination.py):
     objects lost, gained, invented and relabeled, plus CLIP similarity to the raw image
     (does restoration make things up?)

Other restorers (e.g. Restormer / NAFNet / DRSformer outputs) can be checked the same way
without their code: pass --external NAME=DIR, where DIR holds
<dataset>/<condition>/<image_id>.(png|jpg) restored images. Images a restorer did not
produce are left out of that restorer's comparison (paired with the raw image on the same subset).

Restored images are cached (default data/generative_restore/, git-ignored) and an
interrupted run resumes where it stopped.

Outputs (in --out):
  summary.csv         dataset x condition x method: mAP, change vs raw + CI, hallucination rates,
                      CLIP similarity, restore and detect time
  per_class.csv       AP50 / recall50 per class
  per_image.csv       hallucination counts and CLIP similarity per image x method
  detections.pkl.gz   raw detections, for re-scoring at another threshold without the GPU
  headline.json       run config and the main numbers

Usage:
  python scripts/run_generative_restore.py --limit 100                 # pilot: 100 per dataset x condition
  python scripts/run_generative_restore.py                             # all of DAWN + XWOD test
  python scripts/run_generative_restore.py --external restormer=data/restored/restormer
  python scripts/run_generative_restore.py --rescore                   # recompute from detections.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.detection_eval import evaluate
from bolero.hallucination import compare_image, paired_bootstrap_map50, summarize
from bolero.weather_data import CLASSES, load_dawn, load_xwod, open_rgb

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def load_samples(args) -> tuple[list, dict]:
    samples, skipped = [], {}
    for ds in args.datasets:
        try:
            samples += load_dawn() if ds == "dawn" else load_xwod(split=args.xwod_split)
        except FileNotFoundError as e:
            skipped[ds] = str(e)
            print(f"skipping {ds}: {e}")
    if args.limit:
        df = pd.DataFrame({"i": range(len(samples)), "k": [(s.dataset, s.condition) for s in samples]})
        samples = [samples[i] for i in df.groupby("k")["i"].head(args.limit)]
    return samples, skipped


def ip2p_name(g: float) -> str:
    return f"ip2p@{g:g}"


def ip2p_path(cache: Path, s, g: float) -> Path:
    return cache / s.dataset / s.condition / f"ig{g:g}" / f"{s.image_id}.png"


def external_path(root: Path, s) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        p = root / s.dataset / s.condition / f"{s.image_id}{suf}"
        if p.exists():
            return p
    return None


def run_restorer(samples, args) -> dict[str, float]:
    """Make every missing InstructPix2Pix restoration; returns seconds per new edit, keyed by path."""
    from bolero.generative import WeatherRemover, edit_seed

    times_csv = args.cache / "times.csv"
    times = pd.read_csv(times_csv).set_index("path")["seconds"].to_dict() if times_csv.exists() else {}
    todo = [(s, g) for g in args.image_guidance for s in samples if not ip2p_path(args.cache, s, g).exists()]
    print(f"{len(samples)} images x {len(args.image_guidance)} strengths; {len(todo)} restorations to make")
    if not todo:
        return times
    remover = WeatherRemover(args.device, steps=args.steps)
    start = time.time()
    try:
        for s, g in tqdm(todo, desc="restore"):
            out = remover(open_rgb(s.path), s.condition, g, edit_seed(s.image_id, s.condition, g, args.seed),
                          text_guidance=args.text_guidance)
            p = ip2p_path(args.cache, s, g)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.save(p)
            times[str(p)] = remover.last_seconds
    finally:  # keep timings of an interrupted run
        args.cache.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": list(times), "seconds": list(times.values())}).to_csv(times_csv, index=False)
    print(f"restored {len(todo)} in {(time.time() - start) / 60:.1f} min")
    del remover
    return times


def method_image(method: str, s, raw: Image.Image, args, externals: dict, strengths) -> tuple[Image.Image | None, str | None]:
    """The image a method hands to the detector (None if the method has no output for `s`)."""
    if method == "none":
        return raw, None
    if method.startswith("baseline:"):
        from bolero.baselines import apply_baseline
        from bolero.tuning import method_key
        name = method.split(":", 1)[1]
        y = apply_baseline(name, np.asarray(raw, np.float32) / 255.0, strengths[method_key("baseline", name)])
        return Image.fromarray(np.round(np.clip(y, 0, 1) * 255).astype(np.uint8)), None
    if method.startswith("ip2p@"):
        p = ip2p_path(args.cache, s, float(method[5:]))
    else:
        p = external_path(externals[method], s)
        if p is None:
            return None, None
    with Image.open(p) as im:
        img = im.convert("RGB")
    if img.size != raw.size:  # boxes are in raw-image pixels
        img = img.resize(raw.size, Image.BICUBIC)
    return img, str(p)


def detect_all(samples, methods, args, externals) -> tuple[dict, dict, dict]:
    from bolero.detection_eval import FrozenDetector
    from bolero.tuning import load_strengths

    strengths, _ = load_strengths(args.strengths)
    detector = FrozenDetector(args.model, device=args.device, min_score=args.min_score)
    clip = None
    if not args.no_clip:
        from bolero.generative import ClipEmbedder
        clip = ClipEmbedder(args.device)
    dets = {m: [] for m in methods}
    sims = {m: [] for m in methods}
    paths = {m: [] for m in methods}
    for s in tqdm(samples, desc="detect"):
        raw = open_rgb(s.path)
        imgs = {m: method_image(m, s, raw, args, externals, strengths) for m in methods}
        emb = {}
        if clip is not None:
            have = [m for m in methods if imgs[m][0] is not None]
            emb = dict(zip(have, clip([imgs[m][0] for m in have])))
        for m in methods:
            img, p = imgs[m]
            dets[m].append(detector(img) if img is not None else None)
            sims[m].append(float(emb[m] @ emb["none"]) if m in emb else np.nan)
            paths[m].append(p)
    return dets, sims, paths


def score(samples, dets, sims, times, paths, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = list(dets)
    per_image, summary, per_class = [], [], []
    for i, s in enumerate(samples):
        for m in methods[1:]:
            if dets[m][i] is None:
                continue
            per_image.append({"dataset": s.dataset, "condition": s.condition, "image_id": s.image_id, "method": m,
                              "clip_sim": sims[m][i], **compare_image(dets["none"][i], dets[m][i], s.boxes, s.labels,
                                                                       args.threshold)})
    per_image = pd.DataFrame(per_image)

    groups = pd.Series([(s.dataset, s.condition) for s in samples])
    for (ds, cond), idx in groups.groupby(groups).groups.items():
        for m in methods:
            ix = [i for i in idx if dets[m][i] is not None]
            if not ix:
                continue
            gb, gl = [samples[i].boxes for i in ix], [samples[i].labels for i in ix]
            res = evaluate(gb, gl, [dets[m][i] for i in ix])
            row = {"dataset": ds, "condition": cond, "method": m, "images": len(ix),
                   "map50": res["map50"], "map50_95": res["map50_95"],
                   "person_recall50": res["per_class"]["person"]["recall50"],
                   "detect_ms": 1000 * np.mean([dets[m][i].seconds for i in ix])}
            if m != "none":
                d, lo, hi = paired_bootstrap_map50(gb, gl, [dets["none"][i] for i in ix], [dets[m][i] for i in ix],
                                                   CLASSES, reps=args.bootstrap, seed=args.seed)
                sub = per_image[(per_image.dataset == ds) & (per_image.condition == cond) & (per_image.method == m)]
                t = [times.get(paths[m][i]) for i in ix if paths[m][i]]
                t = [x for x in t if x is not None and not np.isnan(x)]
                row.update({"d_map50": d, "d_map50_lo": lo, "d_map50_hi": hi, **summarize(sub),
                            "clip_sim": sub["clip_sim"].mean(), "restore_s": np.mean(t) if t else np.nan})
            summary.append(row)
            per_class += [{"dataset": ds, "condition": cond, "method": m, "class": c, **v}
                          for c, v in res["per_class"].items()]
    return pd.DataFrame(summary), pd.DataFrame(per_class), per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["dawn", "xwod"], choices=["dawn", "xwod"])
    parser.add_argument("--xwod-split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="max images per dataset x condition (0 = all)")
    parser.add_argument("--image-guidance", nargs="*", type=float, default=None,
                        help="InstructPix2Pix strengths (lower = stronger edit); default 1.5 2.0; none = skip")
    parser.add_argument("--text-guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--baselines", nargs="*", default=["autocontrast"],
                        help="preprocessing baselines to include for reference")
    parser.add_argument("--external", nargs="*", default=[], metavar="NAME=DIR",
                        help="restored images from another method: DIR/<dataset>/<condition>/<image_id>.png")
    parser.add_argument("--threshold", type=float, default=0.5, help="operating score for hallucination counts")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap reps for the mAP50 CI (0 = off)")
    parser.add_argument("--no-clip", action="store_true", help="skip CLIP similarity")
    parser.add_argument("--model", default="detr")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--strengths", type=Path, default=RESULTS_DIR / "tuning" / "strengths.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "generative_restore")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "generative_restore")
    parser.add_argument("--rescore", action="store_true",
                        help="re-score from --out/detections.pkl.gz (new --threshold / --bootstrap), no models")
    args = parser.parse_args()
    from bolero.generative import RESTORE_IMAGE_GUIDANCE
    if args.image_guidance is None:
        args.image_guidance = list(RESTORE_IMAGE_GUIDANCE)

    externals = {}
    for spec in args.external:
        name, _, d = spec.partition("=")
        if not d or not Path(d).is_dir():
            parser.error(f"--external {spec}: expected NAME=DIR with an existing DIR")
        externals[f"ext:{name}"] = Path(d)

    samples, skipped = load_samples(args)
    if not samples:
        parser.error("no images loaded; see the skip messages above")
    pkl = args.out / "detections.pkl.gz"
    if args.rescore:
        with gzip.open(pkl, "rb") as f:
            saved = pickle.load(f)
        keys = {(s.dataset, s.condition, s.image_id): s for s in samples}
        samples = [keys[k] for k in saved["samples"]]
        dets, sims, paths, times = saved["dets"], saved["sims"], saved["paths"], saved["times"]
    else:
        methods = (["none"] + [f"baseline:{b}" for b in args.baselines]
                   + [ip2p_name(g) for g in args.image_guidance] + list(externals))
        times = run_restorer(samples, args) if args.image_guidance else {}
        dets, sims, paths = detect_all(samples, methods, args, externals)
        args.out.mkdir(parents=True, exist_ok=True)
        with gzip.open(pkl, "wb") as f:
            pickle.dump({"samples": [(s.dataset, s.condition, s.image_id) for s in samples],
                         "dets": dets, "sims": sims, "paths": paths, "times": times}, f)

    summary, per_class, per_image = score(samples, dets, sims, times, paths, args)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    per_class.to_csv(args.out / "per_class.csv", index=False)
    per_image.to_csv(args.out / "per_image.csv", index=False)
    (args.out / "headline.json").write_text(json.dumps({
        "images": len(samples), "skipped": skipped, "threshold": args.threshold, "bootstrap": args.bootstrap,
        "image_guidance": args.image_guidance, "text_guidance": args.text_guidance, "steps": args.steps,
        "externals": {k: str(v) for k, v in externals.items()}, "model": args.model,
        "rows": summary.round(4).replace({np.nan: None}).to_dict(orient="records"),
    }, indent=2))
    cols = ["dataset", "condition", "method", "images", "map50", "d_map50", "d_map50_lo", "d_map50_hi",
            "lost_rate", "gained_rate", "invented_per_100img", "relabel_rate", "clip_sim", "restore_s"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
