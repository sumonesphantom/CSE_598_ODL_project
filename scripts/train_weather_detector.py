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
  carla/clean|rain|dust     CARLA frames from held-out towns (Town10HD, Town02), when --carla exists

With CARLA frames present, a fourth detector, `carla`, trains on the same clean pool plus CARLA
clean / rain / dust frames from the other towns (simulator weather instead of physics filters).

With the CARLA car-camera drives present (--car, record_ego_drive.py), a fifth detector,
`combined_car`, is the combined detector plus clean frames from the car's own camera (training
towns only, one frame per 1.5 s): a car has plenty of clean footage from its own camera, and
without it every detector sends 20-30 % of clean car frames to a repair (run_car_routing.py).
It adds three tests on the held-out towns (Town10HD, Town02), one frame per 2.5 s:
  car/clean                   clean car frames
  car/rain, car/dust          the same frames under physics rain / dust (effects and levels
                              rotate over frames; exact CARLA depth)

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
from bolero.car_perception import router_features
from bolero.ego import apply_weather
from bolero.weather_data import load_carla, load_coco, load_condition_folders, load_dawn, load_xwod, open_rgb

CARLA_TEST_TOWNS = ("Town10HD_Opt", "Town02_Opt")   # held out from the CARLA-trained detector

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
    parser.add_argument("--carla", type=Path, default=DATA_DIR / "carla",
                        help="CARLA frames (record_carla_weather.py); adds a CARLA-trained detector and CARLA tests")
    parser.add_argument("--car", type=Path, default=DATA_DIR / "carla_ego",
                        help="CARLA car-camera drives (record_ego_drive.py); adds combined_car and car tests")
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

    carla_tests = {}
    if (args.carla / "manifest.csv").exists():
        # Simulator weather: CARLA clean + rain + dust (levels 1-3) from the training towns. CARLA clean
        # frames join this detector's clean examples, or it could learn "rendered = weather".
        def carla_paths(cond, test):
            levels = (0,) if cond == "clean" else (1, 2, 3)
            return [s.path for lv in levels for s in load_carla(args.carla, cond, lv)
                    if (s.path.name.split("_")[0] in {t.split("_")[0].lower() for t in CARLA_TEST_TOWNS}) == test]
        f_cc = feats(carla_paths("clean", False), "carla clean (train towns)")
        f_cr = feats(carla_paths("rain", False), "carla rain (train towns)")
        f_cd = feats(carla_paths("dust", False), "carla dust (train towns)")
        training["carla"] = (np.vstack([f_clean, f_cc, f_cr, f_cd]),
                             np.concatenate([clean_y, [CLEAN] * len(f_cc), ["rain"] * len(f_cr), ["dust"] * len(f_cd)]))
        carla_tests = {"carla/clean": (carla_paths("clean", True), CLEAN), "carla/rain": (carla_paths("rain", True), "rain"),
                       "carla/dust": (carla_paths("dust", True), "dust")}

    car_test_feats = {}
    if (args.car / "manifest.csv").exists():
        from run_car_perception import load_frame

        drives = pd.read_csv(args.car / "manifest.csv")
        idx = drives.frame.str[-4:].astype(int)
        held_out = drives.town.isin(CARLA_TEST_TOWNS)
        car_clean = list(drives[~held_out & (idx % 3 == 0)].path)
        f_car = feats(car_clean, "car clean (train towns)")
        training["combined_car"] = (np.vstack([training["combined"][0], f_car]),
                                    np.concatenate([training["combined"][1], [CLEAN] * len(f_car)]))
        test_frames = list(drives[held_out & (idx % 5 == 0)].frame)
        rain_fx = [w for w in WEATHER if WEATHER[w].family == "rain"]
        dust_fx = [w for w in WEATHER if WEATHER[w].family == "dust"]
        car_test_feats = {"car/clean": [], "car/rain": [], "car/dust": []}
        for i, fid in enumerate(tqdm(test_frames, desc="car tests")):
            img, depth, _ = load_frame(args.car, fid)
            lv = LEVELS[i % 3]
            car_test_feats["car/clean"].append(router_features(img))
            car_test_feats["car/rain"].append(router_features(apply_weather(img, depth, rain_fx[i % len(rain_fx)], lv, fid, args.seed)))
            car_test_feats["car/dust"].append(router_features(apply_weather(img, depth, dust_fx[i % len(dust_fx)], lv, fid, args.seed)))
        car_test_feats = {k: np.stack(v) for k, v in car_test_feats.items()}

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
        **carla_tests,
    }
    test_feats = {k: feats(paths, k) for k, (paths, _) in tests.items()}
    for k, v in car_test_feats.items():
        tests[k] = ([None] * len(v), k.split("/")[1])
        test_feats[k] = v

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
