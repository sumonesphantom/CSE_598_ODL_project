"""Detection under real heavy rain and dust: frozen DETR on DAWN (and XWOD when present).

For each real adverse-weather image the frozen detector runs on the image as-is and
after each depth-free preprocessing baseline, and boxes are scored against the
ground truth with COCO-style mAP. This is the real-data counterpart of the
Imagenette classification study: does any preprocessing help a detector that was
never trained for the weather? (Li et al., CVPR 2019 found deraining usually hurts.)

Outputs (in --out):
  detections.pkl.gz   raw detections per image x method (for re-scoring)
  summary.csv         per dataset x condition x method: mAP50, mAP50-95, recall50, latency
  per_class.csv       per dataset x condition x method x class: AP50, AP50-95, recall50, n_gt
  headline.json       mAP50 per condition and method, plus run config

Usage:
  python scripts/run_detection.py                      # DAWN rain + sand(dust), all methods
  python scripts/run_detection.py --limit 20           # quick smoke run
"""

from __future__ import annotations

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
from bolero.config import RESULTS_DIR
from bolero.detection_eval import FrozenDetector, evaluate
from bolero.tuning import load_strengths, method_key
from bolero.weather_data import load_dawn, load_xwod


def preprocess(method: str, image: Image.Image, strengths: dict) -> Image.Image:
    if method == "none":
        return image
    name = method.split(":", 1)[1]
    x = np.asarray(image, np.float32) / 255.0
    y = apply_baseline(name, x, strengths[method_key("baseline", name)])
    return Image.fromarray(np.round(np.clip(y, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["dawn", "xwod"], choices=["dawn", "xwod"],
                        help="xwod is skipped with a note when data/xwod is missing")
    parser.add_argument("--xwod-split", default="test")
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--model", default="detr")
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--strengths", type=Path, default=RESULTS_DIR / "tuning" / "strengths.json")
    parser.add_argument("--limit", type=int, default=0, help="max images per dataset x condition (0 = all)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "detection")
    args = parser.parse_args()

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

    strengths, _ = load_strengths(args.strengths)
    methods = ["none"] + [f"baseline:{b}" for b in args.baselines]
    detector = FrozenDetector(args.model, device=args.device, min_score=args.min_score)
    dets = {m: [] for m in methods}
    for s in tqdm(samples, desc="images"):
        with Image.open(s.path) as im:
            image = im.convert("RGB")
        for m in methods:
            dets[m].append(detector(preprocess(m, image, strengths)))

    summary, per_class = [], []
    groups = pd.Series([(s.dataset, s.condition) for s in samples])
    for (ds, cond), idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        for m in methods:
            res = evaluate([samples[i].boxes for i in idx], [samples[i].labels for i in idx],
                           [dets[m][i] for i in idx])
            n_gt = sum(v["n_gt"] for v in res["per_class"].values())
            recall = sum(v["recall50"] * v["n_gt"] for v in res["per_class"].values() if v["n_gt"]) / max(n_gt, 1)
            summary.append({"dataset": ds, "condition": cond, "method": m, "images": len(idx), "boxes": n_gt,
                            "map50": res["map50"], "map50_95": res["map50_95"], "recall50": recall,
                            "person_recall50": res["per_class"]["person"]["recall50"],
                            "detect_ms": 1000 * np.mean([dets[m][i].seconds for i in idx])})
            per_class += [{"dataset": ds, "condition": cond, "method": m, "class": c, **v}
                          for c, v in res["per_class"].items()]

    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "detections.pkl.gz", "wb") as f:
        pickle.dump({"samples": [(s.dataset, s.condition, s.image_id) for s in samples], "dets": dets}, f)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(args.out / "per_class.csv", index=False)
    (args.out / "headline.json").write_text(json.dumps({
        "model": args.model, "min_score": args.min_score, "images": len(samples), "skipped": skipped,
        "map50": {f"{r.dataset}/{r.condition}/{r.method}": round(r.map50, 4) for r in summary.itertuples()},
    }, indent=2))
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
