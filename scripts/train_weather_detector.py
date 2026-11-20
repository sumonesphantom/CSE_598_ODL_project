"""Retrain the RF condition detector to recognize real rain and dust (clean / rain / dust).

Three detectors with the same features (detection.extract_features), the same random forest
and the same clean examples; only the source of the rain and dust examples differs:
  synthetic   the five physics weather effects (levels 1-3) applied to the clean pool
  real        real rain and dust photos (XWOD train + valid, VIRADA rain frames from training days)
  combined    both

Clean pool: VIRADA dry frames from training days + 300 COCO traffic scenes that are not in
the 300-image degradation-comparison sample.

Held-out test sets, never used for training:
  xwod/rain, xwod/dust      XWOD v1.1 test split
  dawn/rain, dawn/dust      DAWN rain / sand (a different dataset from all training data)
  virada/rain, virada/clean VIRADA frames from held-out days (split by day, so no clip or day
                            is in both training and test)
  coco/clean                the 300-image COCO comparison sample
  imagenette/clean          200 Imagenette val images, a clean source no detector trained on

Images are resized to a 320-pixel short side before features, so resolution does not identify
the dataset. A prediction counts as `decided` when the top label is not clean and has
probability >= 0.6, the detector's routing rule; otherwise the image is treated as clean.
Intervals are 95 % Wilson intervals over images.

SDSI (6,578 real dust-storm / 6,638 clear surveillance images) would be the natural extra dust
source; it has no public download, so it is not used. Add it with --sdsi data/sdsi once
obtained (folders storm/ and clear/, as weather_data.load_condition_folders reads).

Outputs (in --out): summary.csv, headline.json, detector_<name>.joblib.

Usage:
  python scripts/prepare_virada.py
  python scripts/train_weather_detector.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.data import list_images
from bolero.detection import CLEAN, PerturbationDetector, extract_features
from bolero.perturbations import LEVELS, WEATHER, rng_for
from bolero.weather_data import load_coco, load_condition_folders, load_dawn, load_xwod, open_rgb

SIDE = 320
THRESHOLD = 0.6


def standardize(path) -> np.ndarray:
    x = np.asarray(open_rgb(path), np.float32) / 255.0
    h, w = x.shape[:2]
    s = SIDE / min(h, w)
    return cv2.resize(x, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def virada_split(frames: pd.DataFrame) -> pd.DataFrame:
    """Hold out whole days: the last rain day and every other dry-only day."""
    rain_days = sorted(frames.loc[frames.label == "rain", "day"].unique())
    dry_only = sorted(set(frames.day) - set(rain_days))
    test_days = set(rain_days[-1:]) | set(dry_only[1::2])
    return frames.assign(split=np.where(frames.day.isin(test_days), "test", "train"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--virada", type=Path, default=DATA_DIR / "virada")
    parser.add_argument("--sdsi", type=Path, default=None, help="optional SDSI root (storm/ and clear/ folders)")
    parser.add_argument("--coco-clean", type=int, default=300, help="extra COCO clean training images")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "weather_detector")
    args = parser.parse_args()

    frames = virada_split(pd.read_csv(args.virada / "frames.csv"))
    print("VIRADA frames:\n", frames.groupby(["split", "label"]).agg(frames=("path", "size"), days=("day", "nunique")))

    test_coco = load_coco(n=300, seed=args.seed)
    held = {s.image_id for s in test_coco}
    pool = [s for s in load_coco(n=10**6, seed=args.seed) if s.image_id not in held]
    pick = np.random.default_rng(args.seed).choice(len(pool), args.coco_clean, replace=False)
    train_coco = [pool[i] for i in sorted(pick)]

    xwod_train = load_xwod(split="train") + load_xwod(split="valid")
    xwod_test, dawn = load_xwod(split="test"), load_dawn()
    clean_paths = [s.path for s in train_coco] + list(frames.query("split == 'train' and label == 'no_rain'").path)
    real_rain = [s.path for s in xwod_train if s.condition == "rain"] + list(frames.query("split == 'train' and label == 'rain'").path)
    real_dust = [s.path for s in xwod_train if s.condition == "dust"]
    if args.sdsi:
        sdsi = load_condition_folders(args.sdsi, "sdsi")
        real_dust += [c.path for c in sdsi if c.label in ("storm", "dust")]
        clean_paths += [c.path for c in sdsi if c.label in ("clear", "clean")]

    feats = lambda paths, desc: np.stack([extract_features(standardize(p)) for p in tqdm(paths, desc=desc)])
    f_clean = feats(clean_paths, "clean pool")
    f_rain, f_dust = feats(real_rain, "real rain"), feats(real_dust, "real dust")

    from bolero.depth import DepthEstimator
    depth = DepthEstimator(device=args.device)
    syn_x, syn_y = [], []
    for i, p in enumerate(tqdm(clean_paths, desc="physics weather")):
        x = standardize(p)
        d = depth(x)
        for name, pert in WEATHER.items():
            for level in LEVELS:
                syn_x.append(extract_features(pert(x, level, d, rng_for(f"wd{i}", name, level, args.seed))))
                syn_y.append(pert.family)
    syn_x, syn_y = np.stack(syn_x), np.array(syn_y)

    clean_y = np.array([CLEAN] * len(f_clean))
    training = {
        "synthetic": (np.vstack([f_clean, syn_x]), np.concatenate([clean_y, syn_y])),
        "real": (np.vstack([f_clean, f_rain, f_dust]),
                 np.concatenate([clean_y, ["rain"] * len(f_rain), ["dust"] * len(f_dust)])),
    }
    training["combined"] = (np.vstack([training["real"][0], syn_x]), np.concatenate([training["real"][1], syn_y]))

    tests = {
        "xwod/rain": ([s.path for s in xwod_test if s.condition == "rain"], "rain"),
        "xwod/dust": ([s.path for s in xwod_test if s.condition == "dust"], "dust"),
        "dawn/rain": ([s.path for s in dawn if s.condition == "rain"], "rain"),
        "dawn/dust": ([s.path for s in dawn if s.condition == "dust"], "dust"),
        "virada/rain": (list(frames.query("split == 'test' and label == 'rain'").path), "rain"),
        "virada/clean": (list(frames.query("split == 'test' and label == 'no_rain'").path), CLEAN),
        "coco/clean": ([s.path for s in test_coco], CLEAN),
        # Clean images from a source no detector trained on: guards against learning
        # "not COCO / VIRADA means weather" instead of rain and dust.
        "imagenette/clean": ([r.path for r in list_images("val", per_class=20, seed=args.seed)], CLEAN),
    }
    test_feats = {k: feats(paths, k) for k, (paths, _) in tests.items()}

    rows, args_out = [], args.out
    args_out.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in training.items():
        det = PerturbationDetector(threshold=THRESHOLD).fit(x, y)
        det.save(args_out / f"detector_{name}.joblib")
        for key, (paths, truth) in tests.items():
            proba = det.predict_proba(test_feats[key])
            top = det.classes[proba.argmax(1)]
            conf = proba.max(1)
            decided = np.where((top != CLEAN) & (conf >= THRESHOLD), top, CLEAN)
            n = len(paths)
            k_top, k_dec = int((top == truth).sum()), int((decided == truth).sum())
            rows.append({"detector": name, "test": key, "truth": truth, "images": n,
                         "top1": k_top / n, "top1_low": wilson(k_top, n)[0], "top1_high": wilson(k_top, n)[1],
                         "decided": k_dec / n, "decided_low": wilson(k_dec, n)[0], "decided_high": wilson(k_dec, n)[1],
                         **{f"pred_{c}": float((top == c).mean()) for c in (CLEAN, "rain", "dust")}})
    summary = pd.DataFrame(rows)
    summary.to_csv(args_out / "summary.csv", index=False)
    (args_out / "headline.json").write_text(json.dumps({
        "threshold": THRESHOLD, "side": SIDE,
        "training_counts": {k: dict(zip(*np.unique(v[1], return_counts=True))) for k, v in training.items()},
        "virada_test_days": sorted(frames.loc[frames.split == "test", "day"].unique()),
        "sdsi_used": bool(args.sdsi),
        "decided": {f"{r.detector}/{r.test}": round(r.decided, 3) for r in summary.itertuples()},
    }, indent=2, default=int))
    with pd.option_context("display.width", 200):
        print(summary.pivot(index="test", columns="detector", values="decided").round(3))
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
