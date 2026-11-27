"""The algorithm's depth step on the car: does depth (or SAM) help the car find what is in its path?

The car pipeline of Phases 2-4 runs the algorithm's decision logic (condition router -> repair
for that condition -> frozen detector) but not its depth step. This adds the pieces from `main`
that the car did not use, each as an option on top of that pipeline, and measures them on the
car frames of the held-out towns (every 5th frame of Town10HD and Town02, as in Phase 2), clean
and under every weather v2 effect at levels 1-3:

  base         the Phase 4 pipeline: router (combined_car, 0.6 rule) -> repair -> Mask R-CNN
  cue:<name>   + one of the six depth cues (bolero.cues) at its tuned strength
               (results/tuning/strengths.json), depth from Depth-Anything-V2 on the repaired frame
  sam:<name>   + the boundary or separation cue with a SAM structure map instead of depth
               (bolero.car_depth.object_structure: every object-sized SAM mask)
  conformal    the router's 0.6 rule replaced by the conformal gate (bolero.conformal), calibrated
               on clean car frames of the training towns the router never saw, alpha = --alpha
  conformal_local  the same gate calibrated on clean frames of the held-out towns that are not in
               the test set (as a car would be calibrated in the area it drives in)
  obstacle     the depth obstacle warning (bolero.car_depth.DepthObstacle): frames where the
               nearest object in the car's path is found by the detector, by the warning, or both

Measures: mAP50, in-path recall (share of objects in the car's path found), and for each frame
with an object in the path whether the nearest one is found; changes vs. base with 95 % block
bootstrap intervals over 10-second stretches (as in Phase 2); time per stage.

Usage: python scripts/run_car_depth.py [--stage all|infer|analyze|tune-obstacle]
  tune-obstacle  choose the obstacle-warning settings on clean frames of the training towns
                 (writes obstacle_tuning.csv; the chosen values are fixed in OBSTACLE below)
  tune-warning   cut the warning's false alarms: choose its filters on WEATHERED training-town
                 frames (half resolution, as the car runs it), then check them once on the
                 held-out frames (warning_tuning.csv, warning_validation.csv; fixed in WARNING)
Outputs (in --out): summary.csv, by_level.csv, obstacle.csv, routing.csv, timing.csv, headline.json
"""

import argparse
import json
import pickle
import time
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

from bolero.car_depth import DepthObstacle, object_structure
from bolero.car_perception import CameraModel, CarDetector, router_features
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.conformal import ConformalGate
from bolero.cues import CUES, apply_cue
from bolero.depth import DepthEstimator
from bolero.detection import CLEAN, PerturbationDetector
from bolero.detection_eval import Detections
from bolero.ego import apply_weather
from bolero.learned_restoration import build
from bolero.perturbations import LEVELS, WEATHER
from bolero.segmentation import Segmenter
from bolero.tuning import load_strengths, method_key

VARIANTS = [("clean", 0)] + [(e, lv) for e in WEATHER for lv in LEVELS]
SAM_CUES = ("depth_boundary_emphasis", "entity_background_separation")   # the team's two default cues
# Obstacle-warning settings, chosen on clean frames of the training towns (every 4th frame of
# Town01, 03, 05; best found-minus-false-warning over a grid): 62 % of path hazards found within
# 30 % of their distance, 14 % false warnings on frames without one.
OBSTACLE = {"ratio": 0.8, "min_area": 100, "opening": 1, "scale": 0.65}
DIST_TOL = 0.3          # the warning finds a hazard if its distance is within 30 % of the true one


def methods():
    return ["base"] + [f"cue:{c}" for c in CUES] + [f"sam:{c}" for c in SAM_CUES] + ["conformal", "conformal_local"]


def route_rule(router, feats):
    p = router.predict_proba(feats[None])[0]
    top = router.classes[p.argmax()]
    return str(top) if (top != CLEAN and p.max() >= 0.6) else CLEAN


def calibrate_gate(args, router, local: bool = False):
    m = pd.read_csv(args.data / "manifest.csv")
    idx = m.frame.str[-4:].astype(int)
    if local:   # held-out towns, frames not in the test set (the test uses every 5th frame)
        calib = m[m.town.isin(TEST_TOWNS) & (idx % 5 != 0) & (idx % 4 == 1)].frame.tolist()
    else:       # training-town clean frames the router was not trained on (it used every 3rd frame)
        calib = m[~m.town.isin(TEST_TOWNS) & (idx % 3 != 0) & (idx % 4 == 1)].frame.tolist()
    feats = []
    for fid in tqdm(calib, desc="gate calibration"):
        img, depth, _ = load_frame(args.data, fid)
        feats.append(router_features(apply_weather(img, depth, "clean", 0, fid, args.seed, model="v2")))
    gate = ConformalGate.calibrate(router, np.vstack(feats), args.alpha)
    return gate, len(calib)


def tune_obstacle(args):
    """Grid search of the obstacle-warning settings on clean training-town frames (every 4th)."""
    import itertools

    m = pd.read_csv(args.data / "manifest.csv")
    idx = m.frame.str[-4:].astype(int)
    fr = m[~m.town.isin(TEST_TOWNS) & (idx % 4 == 1)].frame.tolist()
    est, cam = DepthEstimator(), CameraModel(1280, 720, 90, 1.4, -2.0)
    recs = []
    for f in tqdm(fr, desc="depth"):
        img, _, lab = load_frame(args.data, f)
        gt = min((o["distance"] for o in lab["objects"] if o["in_path"]), default=None)
        recs.append((est(img), gt))
    rows = []
    for ratio, area, opening, scale in itertools.product((0.6, 0.7, 0.8), (50, 100, 200, 400), (0, 1),
                                                         (0.65, 0.7, 0.75, 0.8)):
        ob = DepthObstacle(cam, ratio=ratio, min_area=area, opening=opening, scale=scale)
        tp = fa = pos = neg = 0
        for D, gt in recs:
            o = ob(D)["distance"]
            if gt is not None:
                pos += 1
                tp += o is not None and abs(o - gt) / gt < DIST_TOL
            else:
                neg += 1
                fa += o is not None
        rows.append({"ratio": ratio, "min_area": area, "opening": opening, "scale": scale,
                     "found": tp / pos, "false_warning": fa / neg, "frames": len(recs), "hazard_frames": pos})
    out = pd.DataFrame(rows).assign(score=lambda r: r.found - r.false_warning).sort_values("score", ascending=False)
    out.to_csv(args.out / "obstacle_tuning.csv", index=False)
    print(out.head(10).round(3).to_string(index=False))


# The obstacle warning as the car runs it: on the depth map at half resolution (640 x 360), with the
# false-warning filters chosen by --stage tune-warning on weathered training-town frames.
WARNING_CAMERA = CameraModel(640, 360, 90, 1.4, -2.0)
WARNING = {"ratio": 0.8, "half_width": 1.5, "min_area": 25, "opening": 1, "scale": 0.65,
           "min_distance": 6.0, "min_height": 0.8, "reject_bottom": False}


def warning_maps(args, frames, router, repair, est):
    """Half-resolution depth of the repaired frame for every frame x weather variant, plus the
    nearest in-path object's true distance (None if nothing is in the path)."""
    import cv2

    out = []
    for fid in tqdm(frames, desc="depth"):
        img, depth, lab = load_frame(args.data, fid)
        gt = min((o["distance"] for o in lab["objects"] if o["in_path"]), default=None)
        for eff, lv in VARIANTS:
            w = apply_weather(img, depth, eff, lv, fid, args.seed, model="v2")
            route = route_rule(router, router_features(w))
            x = repair[route](w) if route in repair else w
            D = cv2.resize(est(x), (640, 360), interpolation=cv2.INTER_AREA).astype(np.float16)
            fam = "clean" if eff == "clean" else WEATHER[eff].family
            out.append({"frame": fid, "effect": eff, "level": lv, "family": fam, "gt": gt, "depth": D})
    return out


def warning_scores(recs, blobs_of, flt) -> dict:
    """found: share of frames whose nearest path object is 8-40 m away that the warning finds
    (within 30 %); false: share of frames with nothing in the path that get a warning."""
    found = hz = false = near_false = nohz = 0
    for r, blobs in zip(recs, blobs_of):
        ok = [b for b in blobs if b["pixels"] >= flt["min_area"] and abs(b["lateral"]) < flt["half_width"]
              and b["distance"] >= flt["min_distance"] and b["height"] >= flt["min_height"]
              and not (flt["reject_bottom"] and b["at_bottom"]) and b["distance"] < 40.0]
        warn = min((b["distance"] for b in ok), default=None)
        if r["gt"] is None:
            nohz += 1
            false += warn is not None
            near_false += warn is not None and warn < 15.0
        elif 8.0 <= r["gt"] <= 40.0:
            hz += 1
            found += warn is not None and abs(warn - r["gt"]) / r["gt"] < DIST_TOL
    return {"found": found / max(hz, 1), "false_warning": false / max(nohz, 1),
            "near_false_warning": near_false / max(nohz, 1), "hazard_frames": hz, "frames_without_hazard": nohz}


def warning_grid(args, frames, router, repair, est):
    """Found / false-warning rates of every filter setting on weathered frames."""
    import itertools

    train = warning_maps(args, frames, router, repair, est)
    rows, cands = [], {}
    for ratio, half_width in itertools.product((0.7, 0.8), (1.0, 1.5, 2.0)):
        ob = DepthObstacle(WARNING_CAMERA, ratio=ratio, half_width=half_width, min_area=25, opening=1, scale=0.65)
        cands[(ratio, half_width)] = [ob.candidates(r["depth"].astype(np.float32))[0] for r in train]
        for min_area, min_distance, min_height, reject_bottom in itertools.product(
                (25, 50, 100), (0.0, 4.0, 6.0), (0.0, 0.4, 0.8, 1.2), (False, True)):
            flt = {"ratio": ratio, "half_width": half_width, "min_area": min_area, "opening": 1, "scale": 0.65,
                   "min_distance": min_distance, "min_height": min_height, "reject_bottom": reject_bottom}
            rows.append({**flt, **warning_scores(train, cands[(ratio, half_width)], flt)})
    return pd.DataFrame(rows)


def tune_warning(args):
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    router = PerturbationDetector.load(RESULTS_DIR / "weather_detector" / f"detector_{head['car_router']}.joblib")
    choice = json.loads((RESULTS_DIR / "car_repair" / "choice.json").read_text())["repairers"]
    repair = {fam: build(name) for fam, name in choice.items() if name != "none"}
    est = DepthEstimator()
    m = pd.read_csv(args.data / "manifest.csv")
    idx = m.frame.str[-4:].astype(int)
    grid_path = args.out / "warning_tuning.csv"
    if args.reuse_grid and grid_path.exists():
        grid = pd.read_csv(grid_path)
    else:
        grid = warning_grid(args, m[~m.town.isin(TEST_TOWNS) & (idx % 12 == 3)].frame.tolist(), router, repair, est)
        grid.to_csv(grid_path, index=False)
    # Choice: the most hazards found minus twice the false warnings (a false warning makes the car
    # brake for nothing, so it counts double). A hard cap on false warnings left no useful setting.
    best = grid.assign(score=grid.found - 2.0 * grid.false_warning).sort_values(
        ["score", "found"], ascending=[False, False]).iloc[0]
    flt = {k: (bool(best[k]) if k == "reject_bottom" else float(best[k]) if k not in ("min_area", "opening") else int(best[k]))
           for k in WARNING}
    print("chosen on the training towns:", flt, best[["found", "false_warning", "near_false_warning"]].round(3).to_dict())
    # One check on the held-out frames (the step-B test frames), per weather family.
    test = warning_maps(args, m[m.town.isin(TEST_TOWNS) & (idx % 5 == 0)].frame.tolist(), router, repair, est)
    ob = DepthObstacle(WARNING_CAMERA, **{k: flt[k] for k in ("ratio", "half_width", "min_area", "opening", "scale")})
    blobs = [ob.candidates(r["depth"].astype(np.float32))[0] for r in test]
    val = []
    for name, sel in [("all", lambda r: True)] + [(f, (lambda r, f=f: r["family"] == f)) for f in ("clean", "rain", "dust")] + \
                     [(e, (lambda r, e=e: r["effect"] == e)) for e in WEATHER]:
        idx_ = [i for i, r in enumerate(test) if sel(r)]
        for label, f in (("without the new filters", {**flt, "min_distance": 0.0, "min_height": 0.0, "reject_bottom": False,
                                                       "half_width": flt["half_width"]}), ("after", flt)):
            val.append({"set": name, "warning": label, **warning_scores([test[i] for i in idx_], [blobs[i] for i in idx_], f)})
    val = pd.DataFrame(val)
    val.to_csv(args.out / "warning_validation.csv", index=False)
    (args.out / "warning_choice.json").write_text(json.dumps(flt, indent=2))
    with pd.option_context("display.width", 200):
        print(val.round(3).to_string(index=False))


def infer(args, frames):
    cache = args.out / "infer"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames.frame if not (cache / f"{f}.pkl").exists()]
    gate_path = args.out / "gate.json"
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    router = PerturbationDetector.load(RESULTS_DIR / "weather_detector" / f"detector_{head['car_router']}.joblib")
    if not gate_path.exists():
        g1, n1 = calibrate_gate(args, router)
        g2, n2 = calibrate_gate(args, router, local=True)
        gate_path.write_text(json.dumps({"training_towns": {"threshold": g1.threshold, "alpha": g1.alpha, "calibration_frames": n1},
                                         "held_out_towns": {"threshold": g2.threshold, "alpha": g2.alpha, "calibration_frames": n2}},
                                        indent=2))
    g = json.loads(gate_path.read_text())
    gate = ConformalGate(router, g["training_towns"]["threshold"], g["training_towns"]["alpha"])
    gate_local = ConformalGate(router, g["held_out_towns"]["threshold"], g["held_out_towns"]["alpha"])
    if not todo:
        return
    choice = json.loads((RESULTS_DIR / "car_repair" / "choice.json").read_text())["repairers"]
    repair = {fam: build(name) for fam, name in choice.items() if name != "none"}
    strengths, _ = load_strengths(RESULTS_DIR / "tuning" / "strengths.json")
    detector = CarDetector("maskrcnn", fp16=True)
    depth_model = DepthEstimator()
    sam = Segmenter("sam", sam_grid=args.sam_grid)
    camera = CameraModel(1280, 720, 90, 1.4, -2.0)
    obstacle = DepthObstacle(camera, **OBSTACLE)

    def det(x):
        d = detector(x)
        return d.boxes, d.scores, d.labels

    for fid in tqdm(todo, desc="frames"):
        img, depth_true, _ = load_frame(args.data, fid)
        rec = {}
        for eff, lv in VARIANTS:
            w = apply_weather(img, depth_true, eff, lv, fid, args.seed, model="v2")
            feats = router_features(w)
            route, croute = route_rule(router, feats), str(gate.route(feats[None])[0])
            lroute = str(gate_local.route(feats[None])[0])
            ms = {}
            t = time.perf_counter()
            x = repair[route](w) if route in repair else w
            ms["repair"] = 1e3 * (time.perf_counter() - t)
            t = time.perf_counter()
            D = depth_model(x)
            ms["depth"] = 1e3 * (time.perf_counter() - t)
            t = time.perf_counter()
            S = object_structure(sam(x), x.shape[:2])
            ms["sam"] = 1e3 * (time.perf_counter() - t)
            t = time.perf_counter()
            out = {"base": det(x)}
            ms["detect"] = 1e3 * (time.perf_counter() - t)
            for c in CUES:
                t = time.perf_counter()
                y = apply_cue(c, x, D, strengths[method_key("cue", c)])
                ms[f"cue:{c}"] = 1e3 * (time.perf_counter() - t)
                out[f"cue:{c}"] = det(y)
            for c in SAM_CUES:
                out[f"sam:{c}"] = det(apply_cue(c, x, S, strengths[method_key("cue", c)]))
            for key, r in (("conformal", croute), ("conformal_local", lroute)):
                out[key] = out["base"] if r == route else det(repair[r](w) if r in repair else w)
            t = time.perf_counter()
            ob = obstacle(D)
            ms["obstacle"] = 1e3 * (time.perf_counter() - t)
            rec[(eff, lv)] = {"route": route, "croute": croute, "lroute": lroute, "dets": out, "obstacle": ob["distance"],
                              "obstacle_lateral": ob["lateral"], "ms": ms}
        with open(cache / f"{fid}.pkl", "wb") as fh:
            pickle.dump(rec, fh)


def analyze(args, frames):
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    score = head["score_thresholds"]["maskrcnn"]
    labels = {f: json.loads((args.data / "labels" / f"{f}.json").read_text()) for f in frames.frame}
    first = labels[frames.frame.iloc[0]]
    camera = CameraModel(first["width"], first["height"], first["fov"], first["camera_height"], -2.0)
    recs = {}
    for f in frames.frame:
        with open(args.out / "infer" / f"{f}.pkl", "rb") as fh:
            recs[f] = pickle.load(fh)
    blocks = blocks_of(frames)
    fam_of = {v: ("clean" if v[0] == "clean" else WEATHER[v[0]].family) for v in VARIANTS}

    stats = {}       # (method, variant) -> per-frame stats
    for v in VARIANTS:
        for mth in methods():
            stats[(mth, v)] = [frame_stats(gt_of(labels[f])[0], gt_of(labels[f])[2], Detections(*recs[f][v]["dets"][mth]),
                                           score, camera) for f in frames.frame]

    def pooled(mth, vs):
        """Per-frame stats pooled over several weather variants (frame index i of each)."""
        return [stats[(mth, v)][i] for v in vs for i in range(len(frames))], np.concatenate([blocks] * len(vs))

    rows, level_rows = [], []
    groups = {"clean": [v for v in VARIANTS if fam_of[v] == "clean"], "rain": [v for v in VARIANTS if fam_of[v] == "rain"],
              "dust": [v for v in VARIANTS if fam_of[v] == "dust"]}
    for fam, vs in groups.items():
        base, bl = pooled("base", vs)
        for mth in methods():
            st, _ = pooled(mth, vs)
            s = summarize(st)
            row = {"family": fam, "method": mth, **{k: s[k] for k in ("map50", "path_recall", "critical_hit")}}
            if mth != "base":
                def diff(idx, st=st, base=base):
                    return summarize([st[i] for i in idx])["path_recall"] - summarize([base[i] for i in idx])["path_recall"]

                lo, hi = block_ci(diff, bl, args.n_boot, args.seed)
                row.update({"path_recall_change": diff(np.arange(len(st))), "change_low": lo, "change_high": hi,
                            "map50_change": s["map50"] - summarize(base)["map50"]})
            rows.append(row)
    for v in VARIANTS:
        b = summarize(stats[("base", v)])
        for mth in methods():
            s = summarize(stats[(mth, v)])
            level_rows.append({"effect": v[0], "level": v[1], "method": mth, "map50": s["map50"],
                               "path_recall": s["path_recall"], "path_recall_change": s["path_recall"] - b["path_recall"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(level_rows).to_csv(args.out / "by_level.csv", index=False)

    # Obstacle warning: per frame with an object in the path, is the nearest one found?
    ob_rows = []
    for fam, vs in groups.items():
        hz = det_found = warn_found = either = nohz = false_warn = 0
        for v in vs:
            for i, f in enumerate(frames.frame):
                objs = gt_of(labels[f])[2]
                path = [o for o in objs if o["in_path"]]
                warn = recs[f][v]["obstacle"]
                if path:
                    hz += 1
                    near = min(o["distance"] for o in path)
                    d_ok = bool(stats[("base", v)][i]["crit_hit"])
                    w_ok = warn is not None and abs(warn - near) / near < DIST_TOL
                    det_found += d_ok
                    warn_found += w_ok
                    either += d_ok or w_ok
                else:
                    nohz += 1
                    false_warn += warn is not None
        ob_rows.append({"family": fam, "hazard_frames": hz, "found_by_detector": det_found / max(hz, 1),
                        "found_by_warning": warn_found / max(hz, 1), "found_by_either": either / max(hz, 1),
                        "frames_without_hazard": nohz, "false_warning_rate": false_warn / max(nohz, 1)})
    obstacle = pd.DataFrame(ob_rows)
    obstacle.to_csv(args.out / "obstacle.csv", index=False)

    # Routing: how often each gate sends frames to a repair.
    r_rows = []
    for fam, vs in groups.items():
        for name, key in (("rule 0.6", "route"), ("conformal (training towns)", "croute"),
                          ("conformal (held-out towns)", "lroute")):
            routes = pd.Series([recs[f][v][key] for v in vs for f in frames.frame])
            r_rows.append({"family": fam, "gate": name, "to_repair": float((routes != CLEAN).mean()),
                           "to_correct_repair": float((routes == fam).mean()) if fam != "clean" else float("nan")})
    routing = pd.DataFrame(r_rows)
    routing.to_csv(args.out / "routing.csv", index=False)

    ms = pd.DataFrame([recs[f][v]["ms"] for f in frames.frame for v in VARIANTS])
    timing = ms.median().round(1).rename("median_ms").reset_index().rename(columns={"index": "stage"})
    timing.to_csv(args.out / "timing.csv", index=False)

    gate = json.loads((args.out / "gate.json").read_text())
    head_out = {"frames": len(frames), "weather_variants": len(VARIANTS), "score": score, "gate": gate,
                "obstacle_settings": OBSTACLE, "summary": summary.round(4).to_dict("records"),
                "obstacle": obstacle.round(4).to_dict("records"), "routing": routing.round(4).to_dict("records"),
                "timing_ms": dict(zip(timing.stage, timing.median_ms))}
    (args.out / "headline.json").write_text(json.dumps(head_out, indent=2, default=float))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
        print(obstacle.round(3).to_string(index=False))
        print(routing.round(3).to_string(index=False))
        print(timing.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "infer", "analyze", "tune-obstacle", "tune-warning"), default="all")
    p.add_argument("--data", type=Path, default=DATA_DIR / "carla_ego")
    p.add_argument("--alpha", type=float, default=0.05, help="conformal gate: share of clean frames allowed to be sent to a repair")
    p.add_argument("--sam-grid", type=int, default=16)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_depth")
    p.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    p.add_argument("--reuse-grid", action="store_true", help="tune-warning: reuse warning_tuning.csv instead of recomputing it")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "tune-obstacle":
        tune_obstacle(args)
        return
    if args.stage == "tune-warning":
        tune_warning(args)
        return
    m = pd.read_csv(args.data / "manifest.csv")
    frames = m[(m.frame.str[-4:].astype(int) % 5 == 0) & m.town.isin(TEST_TOWNS)].reset_index(drop=True)
    if args.limit:
        frames = frames.head(args.limit)
    if args.stage in ("all", "infer"):
        infer(args, frames)
    if args.stage in ("all", "analyze"):
        analyze(args, frames)


if __name__ == "__main__":
    main()
