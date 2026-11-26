"""Phase 2.1 of the driving study: routing on the car's video, frame by frame and smoothed over time.

Every recorded drive (2 frames/s, 60 s) gets a weather schedule of 6-second segments:
clean, weather A level 1, 2, 3, clean, weather B level 1, 2, 3, clean, clean, where A and B are
rain and dust in an order that alternates between drives, and the rain effect (streaks / mist /
lens) and dust effect (haze / storm) rotate between drives. Each frame is weathered with the
project's physics effects (exact CARLA depth) and passed to each condition detector.

Routes compared:
  raw        the detector's per-frame rule (rain/dust when its probability >= 0.6, else clean)
  smoothed   car_perception.ConditionRouter: moving average of probabilities + hysteresis;
             its settings (alpha, enter, exit) are chosen on the training towns and reported
             on the held-out towns (Town10HD, Town02). combined_car was trained on clean frames
             of the training-town drives, so tuning on them is biased; it keeps its own rule

Measures per route, on the held-out towns:
  accuracy         share of frames routed to the true condition
  clean_false      share of clean frames routed to a repair
  wrong_weather    share of weather frames routed to the other weather's repair
  switches_min     route changes per minute (the schedule itself changes condition 4 times per minute)
  delay_s          seconds from the start of a weather segment to the first correct route
                   (segments never routed correctly count as the full 6 s)

Usage: python scripts/run_car_routing.py
Outputs (in --out): features.npz, routing_time.csv, settings.csv, headline.json
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.car_perception import FAMILIES, router_features, smooth_sequence
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.detection import CLEAN, PerturbationDetector
from bolero.ego import apply_weather

from run_car_perception import TEST_TOWNS, load_frame

SEGMENT_S, FPS = 6.0, 2.0
RAIN = ("rain_streaks", "rain_mist", "rain_lens")
DUST = ("dust_haze", "dust_storm")
ROUTERS = ("real", "synthetic", "combined", "carla", "combined_car")
CAR_TRAINED = ("combined_car",)   # detectors that saw clean frames from the training-town drives
GRID = {"alpha": (1.0, 0.6, 0.4, 0.25), "enter": (0.5, 0.6, 0.7), "exit": (0.3, 0.4, 0.5, 0.6, 0.7)}


def schedule(k: int, n_frames: int) -> list[tuple[str, int]]:
    """(effect, level) per frame for drive number k."""
    rain, dust = RAIN[k % 3], DUST[k % 2]
    a, b = (rain, dust) if k % 2 == 0 else (dust, rain)
    segs = [("clean", 0), (a, 1), (a, 2), (a, 3), ("clean", 0), (b, 1), (b, 2), (b, 3), ("clean", 0), ("clean", 0)]
    per = int(SEGMENT_S * FPS)
    return [segs[min(i // per, len(segs) - 1)] for i in range(n_frames)]


def family(effect: str) -> str:
    return CLEAN if effect == "clean" else ("rain" if effect.startswith("rain") else "dust")


def measures(routes: list[str], truth: list[str]) -> dict:
    routes, truth = np.asarray(routes), np.asarray(truth)
    clean, weather = truth == CLEAN, truth != CLEAN
    per = int(SEGMENT_S * FPS)
    delays = []
    for start in range(0, len(truth), per):
        seg = slice(start, start + per)
        if truth[start] != CLEAN and (start == 0 or truth[start - 1] != truth[start]):
            ok = np.flatnonzero(routes[seg] == truth[seg])
            delays.append(ok[0] / FPS if len(ok) else SEGMENT_S)
    return {"frames": len(truth), "correct": int((routes == truth).sum()),
            "clean_n": int(clean.sum()), "clean_false": int((routes[clean] != CLEAN).sum()),
            "weather_n": int(weather.sum()),
            "wrong_weather": int(((routes != truth) & (routes != CLEAN) & weather).sum()),
            "switches": int((routes[1:] != routes[:-1]).sum()), "delays": delays}


def aggregate(ms: list[dict]) -> dict:
    tot = lambda k: sum(m[k] for m in ms)
    minutes = tot("frames") / FPS / 60.0
    delays = [d for m in ms for d in m["delays"]]
    return {"accuracy": tot("correct") / tot("frames"), "clean_false": tot("clean_false") / tot("clean_n"),
            "wrong_weather": tot("wrong_weather") / tot("weather_n"), "switches_min": tot("switches") / minutes,
            "delay_s": float(np.mean(delays)) if delays else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR / "carla_ego")
    p.add_argument("--detectors", type=Path, default=RESULTS_DIR / "weather_detector")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_routing")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.data / "manifest.csv").sort_values(["drive", "frame"])
    drives = sorted(m.drive.unique())
    routers = {n: PerturbationDetector.load(args.detectors / f"detector_{n}.joblib") for n in ROUTERS
               if (args.detectors / f"detector_{n}.joblib").exists()}

    cache = args.out / "features.npz"   # router features per drive: any detector can be scored later
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        features, truth = z["features"].item(), z["truth"].item()
    else:
        features, truth = {}, {}
        for k, drive in enumerate(tqdm(drives, desc="drives")):
            fids = list(m[m.drive == drive].frame)
            sched = schedule(k, len(fids))
            feats = []
            for fid, (eff, lv) in zip(fids, sched):
                img, depth, _ = load_frame(args.data, fid)
                feats.append(router_features(apply_weather(img, depth, eff, lv, fid, args.seed)))
            features[drive] = np.stack(feats)
            truth[drive] = [family(e) for e, _ in sched]
        np.savez(cache, features=np.array(features, dtype=object), truth=np.array(truth, dtype=object))
    probs = {}
    for n, det in routers.items():
        cls = list(det.classes)
        probs[n] = {}
        for drive, f in features.items():
            pr = det.predict_proba(f)
            probs[n][drive] = np.stack([pr[:, cls.index(c)] if c in cls else np.zeros(len(pr)) for c in FAMILIES], 1)

    town = dict(zip(m.drive, m.town))
    test = [d for d in drives if town[d] in TEST_TOWNS]
    train = [d for d in drives if town[d] not in TEST_TOWNS]
    rows, settings = [], []
    for n in routers:
        # Smoothing settings: best training-town accuracy, ties broken by fewer switches.
        best = None
        for alpha, enter, exit_ in itertools.product(*GRID.values()):
            if exit_ > enter:   # exit == enter, alpha 1 is the plain per-frame rule
                continue
            agg = aggregate([measures(smooth_sequence(probs[n][d], alpha, enter, exit_), truth[d]) for d in train])
            key = (round(agg["accuracy"], 3), -agg["switches_min"])
            settings.append({"router": n, "alpha": alpha, "enter": enter, "exit": exit_, **agg})
            if best is None or key > best[0]:
                best = (key, (alpha, enter, exit_))
        alpha, enter, exit_ = best[1]
        if n in CAR_TRAINED:
            # Trained on clean frames of the training-town drives, so tuning on those drives is
            # biased (no false alarms there by construction): use the detector's own rule.
            alpha, enter, exit_ = 1.0, 0.6, 0.6
        for split, ds in (("train", train), ("test", test)):
            raw = aggregate([measures(smooth_sequence(probs[n][d], 1.0, 0.6, 0.6), truth[d]) for d in ds])
            smo = aggregate([measures(smooth_sequence(probs[n][d], alpha, enter, exit_), truth[d]) for d in ds])
            rows.append({"router": n, "split": split, "route": "raw", **raw})
            rows.append({"router": n, "split": split, "route": "smoothed", "alpha": alpha, "enter": enter,
                         "exit": exit_, **smo})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "routing_time.csv", index=False)
    pd.DataFrame(settings).to_csv(args.out / "settings.csv", index=False)
    head = table[table.split == "test"].set_index(["router", "route"]).drop(columns="split").round(3)
    (args.out / "headline.json").write_text(json.dumps(
        {"test_drives": test, "schedule_switches_per_min": 4.0,
         "test": {f"{r}/{k}": v for (r, k), v in head.to_dict("index").items()}}, indent=2, default=str))
    with pd.option_context("display.width", 200):
        print(head.to_string())


if __name__ == "__main__":
    main()
