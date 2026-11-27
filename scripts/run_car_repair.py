"""The repair step, redone: which repair should the car use for rain and for dust?

Phase 2 chose the repairers on simulated weather (v1), where dark-channel dehazing does very well
because v1 dust is exactly the model dehazing inverts; on real dust photos it does not help. Here
the choice is made on REAL photos and checked on the car frames under both weather models.

Repairers: none, autocontrast, dcp_dehaze, restormer_derain (Phase 2's candidates that are fast
enough to matter; InstructIR was never chosen) and task_repair (trained through the frozen detector
on XWOD train photos: train_task_repair.py). Detector: the car's Mask R-CNN.

Real photos (stage "real"):
  select   XWOD validation split, rain and dust      -> the repairer per condition is chosen here
  test     XWOD test split; DAWN (a different dataset); 300 clean COCO photos (harm check)
  routed   the car's condition router (combined_car, threshold 0.6) picks the repairer per photo
Car frames (stage "sim"): the held-out-town frames of run_car_perception.py (every 5th frame of
Town10HD and Town02), clean and every weather effect at levels 1-3, under weather v1 and v2
(bolero.ego.weather_v2). Measures as in Phase 2: mAP50 and in-path recall at the operating score.

Usage: python scripts/run_car_repair.py [--stage real|sim|all]
Outputs (in --out): real.csv, sim.csv, choice.json (read by run_car_scenarios.py)
"""

import argparse
import json
import pickle
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from run_car_perception import (
    TEST_TOWNS,
    block_ci,
    blocks_of,
    frame_stats,
    gt_of,
    load_frame,
    summarize,
)
from tqdm import tqdm

from bolero.car_perception import CameraModel, CarDetector, router_features
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.degradation_compare import bootstrap_map50, image_matches
from bolero.detection import CLEAN, PerturbationDetector
from bolero.detection_eval import Detections
from bolero.ego import apply_weather
from bolero.learned_restoration import build
from bolero.perturbations import LEVELS, WEATHER
from bolero.weather_data import load_coco, load_dawn, load_xwod, open_rgb

CANDIDATES = ("none", "autocontrast", "dcp_dehaze", "restormer_derain", "task_repair")
SIM_VARIANTS = [("clean", 0)] + [(e, lv) for e in WEATHER for lv in LEVELS]


def repairers(max_side):
    return {r: build(r, max_side=max_side if r.startswith("restormer") else None) for r in CANDIDATES}


def real(args, head, reps, detector, router):
    xv, xt, dawn = load_xwod(split="valid"), load_xwod(split="test"), load_dawn()
    sets = {"select/xwod_valid/rain": [s for s in xv if s.condition == "rain"],
            "select/xwod_valid/dust": [s for s in xv if s.condition == "dust"],
            "test/xwod/rain": [s for s in xt if s.condition == "rain"], "test/xwod/dust": [s for s in xt if s.condition == "dust"],
            "test/dawn/rain": [s for s in dawn if s.condition == "rain"], "test/dawn/dust": [s for s in dawn if s.condition == "dust"],
            "test/coco/clean": load_coco(n=300, seed=args.seed)}
    cache = args.out / "real_dets.pkl"
    store = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    for name, samples in sets.items():
        if name in store:
            continue
        dets, routes = {r: [] for r in CANDIDATES}, []
        for s in tqdm(samples, desc=name):
            x = np.asarray(open_rgb(s.path), np.float32) / 255.0
            pr = router.predict_proba(router_features(x)[None])[0]
            top = router.classes[pr.argmax()]
            routes.append(top if (top != CLEAN and pr.max() >= 0.6) else CLEAN)
            for r, fn in reps.items():
                d = detector(fn(x))
                dets[r].append(Detections(d.boxes, d.scores, d.labels))
        store[name] = {"dets": dets, "routes": routes}
        cache.write_bytes(pickle.dumps(store))

    matches = {name: {r: image_matches([s.boxes for s in sm], [s.labels for s in sm], store[name]["dets"][r]) for r in CANDIDATES}
               for name, sm in sets.items()}
    # Choose per condition on the validation split: the best mAP50; "none" unless something beats it.
    choice = {CLEAN: "none"}
    for fam in ("rain", "dust"):
        m = matches[f"select/xwod_valid/{fam}"]
        scores = {r: bootstrap_map50(m[r], 1, args.seed)[0] for r in CANDIDATES}
        choice[fam] = max(scores, key=lambda r: (scores[r], r == "none"))
    rows = []
    for name, sm in sets.items():
        m = matches[name]
        routed = image_matches([s.boxes for s in sm], [s.labels for s in sm],
                               [store[name]["dets"][choice[rt]][i] for i, rt in enumerate(store[name]["routes"])])
        for pipe, mt in [(r, m[r]) for r in CANDIDATES] + [("routed", routed)]:
            v = bootstrap_map50(mt, args.n_boot, args.seed)
            dv = bootstrap_map50(mt, args.n_boot, args.seed, reference=m["none"]) if pipe != "none" else (0.0, 0.0, 0.0)
            rows.append({"set": name, "pipeline": pipe, "images": len(sm), "map50": v[0], "low": v[1], "high": v[2],
                         "change": dv[0], "change_low": dv[1], "change_high": dv[2]})
        r = pd.Series(store[name]["routes"])
        rows[-1].update({f"routed_{c}": float((r == c).mean()) for c in (CLEAN, "rain", "dust")})
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "real.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.pivot(index="pipeline", columns="set", values="change").round(3))
    return choice


def sim(args, head, reps, detector, choice):
    m = pd.read_csv(args.data / "manifest.csv")
    frames = m[(m.frame.str[-4:].astype(int) % 5 == 0) & m.town.isin(TEST_TOWNS)].reset_index(drop=True)
    score = head["score_thresholds"]["maskrcnn"]
    cache = args.out / "sim"
    cache.mkdir(exist_ok=True)
    for fid in tqdm(frames.frame, desc="car frames"):
        path = cache / f"{fid}.pkl"
        if path.exists():
            continue
        img, depth, _ = load_frame(args.data, fid)
        rec = {}
        for model in ("v1", "v2"):
            for eff, lv in SIM_VARIANTS:
                w = apply_weather(img, depth, eff, lv, fid, args.seed, model=model)
                for r, fn in reps.items():
                    if model == "v1" and r not in ("none", "task_repair", "dcp_dehaze"):
                        continue          # v1 with the other repairers is Phase 2 (run_car_perception.py)
                    d = detector(fn(w))
                    rec[(model, eff, lv, r)] = (d.boxes, d.scores, d.labels)
        path.write_bytes(pickle.dumps(rec))
    labels = {f: json.loads((args.data / "labels" / f"{f}.json").read_text()) for f in frames.frame}
    first = labels[frames.frame.iloc[0]]
    camera = CameraModel(first["width"], first["height"], first["fov"], first["camera_height"], -2.0)
    recs = {f: pickle.loads((cache / f"{f}.pkl").read_bytes()) for f in frames.frame}
    blocks = blocks_of(frames)
    rows = []
    for key in sorted({k for r in recs.values() for k in r}, key=str):
        model, eff, lv, r = key
        stats = [frame_stats(*[gt_of(labels[f])[i] for i in (0, 2)], Detections(*recs[f][key]), score, camera) for f in frames.frame]
        s = summarize(stats)
        rows.append({"weather_model": model, "effect": eff, "level": lv, "family": "clean" if eff == "clean" else WEATHER[eff].family,
                     "repairer": r, **s})
        if r != "none":
            base = [frame_stats(*[gt_of(labels[f])[i] for i in (0, 2)], Detections(*recs[f][(model, eff, lv, "none")]), score, camera)
                    for f in frames.frame]
            def diff(idx, stats=stats, base=base):
                return summarize([stats[i] for i in idx])["path_recall"] - summarize([base[i] for i in idx])["path_recall"]

            lo, hi = block_ci(diff, blocks, 200, args.seed)
            rows[-1].update({"path_recall_change": diff(np.arange(len(frames))), "path_change_low": lo, "path_change_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "sim.csv", index=False)
    fam = out[out.effect != "clean"].groupby(["weather_model", "family", "repairer"])[["map50", "path_recall"]].mean().round(3)
    print(fam.to_string())
    print(out[out.effect == "clean"][["weather_model", "repairer", "map50", "path_recall"]].round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "real", "sim"), default="all")
    p.add_argument("--data", type=Path, default=DATA_DIR / "carla_ego")
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_repair")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    reps = repairers(args.max_side)
    detector = CarDetector("maskrcnn")
    router = PerturbationDetector.load(RESULTS_DIR / "weather_detector" / f"detector_{head['car_router']}.joblib")
    choice_path = args.out / "choice.json"
    if args.stage in ("all", "real"):
        choice = real(args, head, reps, detector, router)
        choice_path.write_text(json.dumps({"repairers": choice, "chosen_on": "XWOD validation split (real photos), mAP50",
                                           "candidates": list(CANDIDATES)}, indent=2))
        print("choice:", choice)
    choice = json.loads(choice_path.read_text())["repairers"]
    if args.stage in ("all", "sim"):
        sim(args, head, reps, detector, choice)


if __name__ == "__main__":
    main()
