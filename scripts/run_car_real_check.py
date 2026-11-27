"""Phase 2 real-photo check: do the car pipeline's repairs help (or hurt) on real photos?

The simulator study (run_car_perception.py) uses physics weather on CARLA frames. Two things it
cannot show: whether a repair harms real clean photos (CARLA renders may simply gain from extra
contrast), and whether the gains carry over to real rain and dust. This runs the car's detector
(Mask R-CNN, as chosen) on photos already on disk:

  coco/clean               300 clean COCO street scenes (the comparison sample)
  xwod/rain, xwod/dust     XWOD v1.1 test split (traffic and street cameras)
  dawn/rain, dawn/dust     DAWN rain and sand (driving and street photos)

Pipelines: none; each repairer applied to every image; routed = the car's condition detector
(combined_car, threshold 0.6) picks the repairer chosen in run_car_perception.py.
Measure: mAP50 (six classes) and its change vs. no repair, 95 % image-bootstrap CI.
This is not Phase 5 (real dashcam weather, e.g. ACDC); it is a quick check with data on hand.

Usage: python scripts/run_car_real_check.py
Outputs (in --out): real_check.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.car_perception import CarDetector, router_features
from bolero.config import RESULTS_DIR, SEED
from bolero.degradation_compare import bootstrap_map50, image_matches
from bolero.detection import CLEAN, PerturbationDetector
from bolero.detection_eval import Detections
from bolero.learned_restoration import build
from bolero.weather_data import load_coco, load_dawn, load_xwod, open_rgb

REPAIRERS = ("none", "autocontrast", "dcp_dehaze", "restormer_derain")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_real_check")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    det_name, router_name = head["car_detector"], head["car_router"]
    choice = head["repairer_choice"][det_name]

    xwod, dawn = load_xwod(split="test"), load_dawn()
    sets = {"coco/clean": load_coco(n=300, seed=args.seed),
            "xwod/rain": [s for s in xwod if s.condition == "rain"], "xwod/dust": [s for s in xwod if s.condition == "dust"],
            "dawn/rain": [s for s in dawn if s.condition == "rain"], "dawn/dust": [s for s in dawn if s.condition == "dust"]}
    detector = CarDetector(det_name)
    repairers = {r: build(r, max_side=args.max_side if r.startswith(("restormer", "instructir")) else None) for r in REPAIRERS}
    router = PerturbationDetector.load(RESULTS_DIR / "weather_detector" / f"detector_{router_name}.joblib")

    rows = []
    for name, samples in sets.items():
        dets = {r: [] for r in REPAIRERS}
        routes = []
        for s in tqdm(samples, desc=name):
            x = np.asarray(open_rgb(s.path), np.float32) / 255.0
            pr = router.predict_proba(router_features(x)[None])[0]
            top = router.classes[pr.argmax()]
            routes.append(top if (top != CLEAN and pr.max() >= 0.6) else CLEAN)
            for r, fn in repairers.items():
                d = detector(fn(x))
                dets[r].append(Detections(d.boxes, d.scores, d.labels))
        gt_b, gt_l = [s.boxes for s in samples], [s.labels for s in samples]
        matches = {r: image_matches(gt_b, gt_l, dets[r]) for r in REPAIRERS}
        routed = image_matches(gt_b, gt_l, [dets[choice[rt]][i] for i, rt in enumerate(routes)])
        base, lo, hi = bootstrap_map50(matches["none"], args.n_boot, args.seed)
        rows.append({"set": name, "pipeline": "none", "images": len(samples), "map50": base, "low": lo, "high": hi,
                     "change": 0.0, "change_low": 0.0, "change_high": 0.0})
        for pipe, mt in [(f"always_{r}", matches[r]) for r in REPAIRERS[1:]] + [(f"routed_{router_name}", routed)]:
            v = bootstrap_map50(mt, args.n_boot, args.seed)
            dv = bootstrap_map50(mt, args.n_boot, args.seed, reference=matches["none"])
            rows.append({"set": name, "pipeline": pipe, "images": len(samples), "map50": v[0], "low": v[1], "high": v[2],
                         "change": dv[0], "change_low": dv[1], "change_high": dv[2],
                         **({"routed_share": {c: float(np.mean(np.array(routes) == c)) for c in (CLEAN, "rain", "dust")}}
                            if pipe.startswith("routed") else {})})
        print(pd.DataFrame(rows[-len(REPAIRERS):]).round(3).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "real_check.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.pivot(index="pipeline", columns="set", values="change").round(3))


if __name__ == "__main__":
    main()
