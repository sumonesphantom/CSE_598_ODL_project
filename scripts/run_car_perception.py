"""Phase 1.3 + 2.2 + 2.3 of the driving study: on the car's camera, does weather hurt detection,
which repair wins it back, does routing on the condition detector pick the right repair, and
how well are the objects in the car's path isolated?

Frames: every --every-th frame of the CARLA car drives (record_ego_drive.py; 2 Hz, so --every 5
= one frame per 2.5 s). Each frame is evaluated clean and under every physics weather effect at
levels 1-3 (16 variants), each variant through every repairer (learned_restoration.REPAIRERS),
each repaired frame through each detector (DETR, Mask R-CNN).

Towns: choices (operating score threshold, which repairer to use for rain and for dust) are made
on the training towns (Town03, 05, 01) and reported on the held-out towns (Town10HD, Town02), the
same split as the CARLA condition-detector tests.

Measures (all class-agnostic matching at IoU 0.5 except mAP50, which is per class):
  map50           frozen-detector mAP50 over the six classes
  path_recall     share of ground-truth objects in the car's path (ahead, < 2 m sideways, < 40 m)
                  that are detected at the operating score threshold
  critical_hit    share of frames whose nearest in-path object is detected
  iso_*           the isolation stage (car_perception.isolate, ground-plane distances):
                  precision / recall of the predicted in-path set, median relative distance error
                  of matched objects within 40 m, and whether the predicted critical object is the
                  true one
Intervals: 95 % block bootstrap over 10-second blocks of each drive (neighbouring frames are
correlated, so single frames are not resampled independently).

Pipelines compared (per frame and weather variant):
  none               no repair
  oracle             the repairer chosen for the true condition (upper bound for routing)
  always_<repairer>  one repairer applied to everything (no routing)
  routed_<detector>  the condition detector's per-frame decision (threshold 0.6) picks the repairer

Usage:
  python scripts/run_car_perception.py            # inference (resumable), then analysis
  python scripts/run_car_perception.py --stage analyze
Outputs (in --out): infer/<frame>.pkl, summary.csv, pipelines.csv, routing_car.csv, headline.json
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from bolero.car_perception import FAMILIES, CameraModel, CarDetector, isolate, router_features
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.degradation_compare import ap_from, match_image
from bolero.detection import CLEAN, PerturbationDetector
from bolero.detection_eval import Detections, iou_matrix
from bolero.ego import PATH_RANGE, apply_weather
from bolero.weather_data import CLASSES
from bolero.learned_restoration import REPAIRERS, build
from bolero.perturbations import LEVELS, WEATHER

TEST_TOWNS = ("Town10HD_Opt", "Town02_Opt")
VARIANTS = [("clean", 0)] + [(e, lv) for e in WEATHER for lv in LEVELS]
FAMILY = {"clean": CLEAN, **{e: p.family for e, p in WEATHER.items()}}
DETECTORS = ("detr", "maskrcnn")
ROUTERS = ("real", "synthetic", "combined", "carla", "combined_car")
CAR_ROUTER = "combined_car"   # chosen in run_car_routing.py (fewest false alarms on clean car frames)
BLOCK_SECONDS = 10.0


def load_frame(root: Path, fid: str):
    img = cv2.imread(str(root / "images" / f"{fid}.png"))[..., ::-1].astype(np.float32) / 255.0
    depth = cv2.imread(str(root / "depth" / f"{fid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
    return img, depth, json.loads((root / "labels" / f"{fid}.json").read_text())


def infer(args, frames: pd.DataFrame):
    out = args.out / "infer"
    out.mkdir(parents=True, exist_ok=True)
    todo = [f for f in frames.frame if not (out / f"{f}.pkl").exists()]
    if not todo:
        return
    # Learned repairers run at --max-side (their deployable setting); the classical dehazer at full resolution.
    repairers = {r: build(r, max_side=args.max_side if r.startswith(("restormer", "instructir")) else None)
                 for r in REPAIRERS}
    detectors = {d: CarDetector(d) for d in DETECTORS}
    for fid in tqdm(todo, desc="frames"):
        img, depth, _ = load_frame(args.data, fid)
        rec = {"features": {}, "dets": {}, "repair_ms": {}}
        for eff, lv in VARIANTS:
            w = apply_weather(img, depth, eff, lv, fid, args.seed)
            rec["features"][(eff, lv)] = router_features(w)
            for r, fn in repairers.items():
                t = time.perf_counter()
                x = fn(w)
                rec["repair_ms"][(eff, lv, r)] = 1e3 * (time.perf_counter() - t)
                for d, det in detectors.items():
                    o = det(x)
                    rec["dets"][(eff, lv, r, d)] = (o.boxes, o.scores, o.labels)
        with open(out / f"{fid}.pkl", "wb") as fh:
            pickle.dump(rec, fh)


# --- measures --------------------------------------------------------------------------------
def gt_of(labels: dict):
    objs = labels["objects"]
    boxes = np.array([o["box"] for o in objs], np.float32).reshape(-1, 4)
    return boxes, [o["label"] for o in objs], objs


def agnostic_hits(gt_boxes: np.ndarray, det: Detections, score: float) -> np.ndarray:
    """Per GT box: matched by a confident detection of any class (greedy, IoU >= 0.5)."""
    keep = det.scores >= score
    boxes, scores = det.boxes[keep], det.scores[keep]
    hit = np.zeros(len(gt_boxes), bool)
    if len(gt_boxes) and len(boxes):
        ious = iou_matrix(boxes[np.argsort(-scores)], gt_boxes)
        for row in ious:
            cand = np.where(hit, -1.0, row)
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                hit[j] = True
    return hit


def frame_stats(gt_boxes, objs, det: Detections, score: float, camera: CameraModel) -> dict:
    """Per-frame counts for the safety and isolation measures (summed later, so blocks resample them)."""
    hit = agnostic_hits(gt_boxes, det, score)
    path = np.array([o["in_path"] for o in objs], bool)
    labels = [o["label"] for o in objs]
    s = {"matches": {c: match_image(gt_boxes, labels, det, c) for c in CLASSES},
         "path_n": int(path.sum()), "path_hit": int((hit & path).sum()),
         "obj_n": len(objs), "obj_hit": int(hit.sum())}
    crit = [i for i in np.argsort([o["distance"] for o in objs]) if objs[i]["in_path"]] if objs else []
    s["crit_n"] = int(bool(crit))
    s["crit_hit"] = int(bool(crit) and hit[crit[0]])
    pred = isolate(det, camera, score)
    pb = np.array([p["box"] for p in pred], np.float32).reshape(-1, 4)
    ious = iou_matrix(pb, gt_boxes) if len(pb) and len(gt_boxes) else np.zeros((len(pb), len(gt_boxes)))
    match = [int(r.argmax()) if r.size and r.max() >= 0.5 else -1 for r in ious]
    pred_path = [k for k, p in enumerate(pred) if p["in_path"]]
    s["iso_pred_path"] = len(pred_path)
    s["iso_pred_path_true"] = sum(1 for k in pred_path if match[k] >= 0 and objs[match[k]]["in_path"])
    s["iso_gt_path_found"] = len({match[k] for k in pred_path if match[k] >= 0 and objs[match[k]]["in_path"]})
    errs = [abs(pred[k]["distance"] - objs[m]["distance"]) / max(objs[m]["distance"], 1.0)
            for k, m in enumerate(match) if m >= 0 and objs[m]["distance"] < PATH_RANGE]
    s["iso_dist_err"] = errs
    pc = next((k for k, p in enumerate(pred) if p["critical"]), None)
    s["iso_crit_ok"] = int(bool(crit) and pc is not None and match[pc] == crit[0])
    return s


def map50_rows(rows: list[dict], idx) -> float:
    aps = [ap_from([rows[i]["matches"][c] for i in idx]) for c in CLASSES]
    aps = [a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else float("nan")


def summarize(rows: list[dict], idx=None) -> dict:
    """Aggregate per-frame stats (and mAP50) over the frames in `idx` (all if None)."""
    idx = np.arange(len(rows)) if idx is None else idx
    tot = lambda k: sum(rows[i][k] for i in idx)
    errs = [e for i in idx for e in rows[i]["iso_dist_err"]]
    return {"map50": map50_rows(rows, idx),
            "path_recall": tot("path_hit") / max(tot("path_n"), 1),
            "object_recall": tot("obj_hit") / max(tot("obj_n"), 1),
            "critical_hit": tot("crit_hit") / max(tot("crit_n"), 1),
            "iso_path_precision": tot("iso_pred_path_true") / max(tot("iso_pred_path"), 1),
            "iso_path_recall": tot("iso_gt_path_found") / max(tot("path_n"), 1),
            "iso_critical_ok": tot("iso_crit_ok") / max(tot("crit_n"), 1),
            "iso_distance_err": float(np.median(errs)) if errs else float("nan")}


def path_recall(rows: list[dict], idx) -> float:
    return sum(rows[i]["path_hit"] for i in idx) / max(sum(rows[i]["path_n"] for i in idx), 1)


def blocks_of(frames: pd.DataFrame) -> np.ndarray:
    return (frames.drive + "_" + (frames.time // BLOCK_SECONDS).astype(int).astype(str)).to_numpy()


def block_ci(fn, blocks: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95 % interval of fn(idx) under resampling of whole blocks."""
    ids = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in ids}
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([members[b] for b in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(args, frames: pd.DataFrame):
    recs = {f: pickle.load(open(args.out / "infer" / f"{f}.pkl", "rb")) for f in frames.frame}
    labels = {f: json.loads((args.data / "labels" / f"{f}.json").read_text()) for f in frames.frame}
    gts = {f: gt_of(labels[f]) for f in frames.frame}
    first = labels[frames.frame.iloc[0]]
    camera = CameraModel(first["width"], first["height"], first["fov"], first["camera_height"], -2.0)
    test = frames.town.isin(TEST_TOWNS).to_numpy()
    det_of = lambda f, key: Detections(*recs[f]["dets"][key][:3])

    # Operating score threshold per detector: best class-agnostic F1 on clean training-town frames.
    thresholds = {}
    for d in DETECTORS:
        best = (0.0, 0.5)
        for thr in np.arange(0.3, 0.95, 0.05):
            tp = fp = fn = 0
            for f in frames.frame[~test]:
                det = det_of(f, ("clean", 0, "none", d))
                hit = agnostic_hits(gts[f][0], det, thr)
                tp, fn = tp + hit.sum(), fn + (~hit).sum()
                fp += max(int((det.scores >= thr).sum()) - int(hit.sum()), 0)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            best = max(best, (f1, round(float(thr), 2)))
        thresholds[d] = best[1]

    def stats_for(fids, key_of, d):
        """key_of(fid) -> (effect, level, repairer)."""
        return [frame_stats(gts[f][0], gts[f][2], det_of(f, (*key_of(f), d)), thresholds[d], camera) for f in fids]

    # 1.3 + 2.2: every variant x repairer x detector, on train and test towns.
    summary = []
    for d in DETECTORS:
        for split, mask in (("train", ~test), ("test", test)):
            fids = list(frames.frame[mask])
            for eff, lv in VARIANTS:
                for r in REPAIRERS:
                    rows = stats_for(fids, lambda f: (eff, lv, r), d)
                    summary.append({"detector": d, "split": split, "effect": eff, "family": FAMILY[eff], "level": lv,
                                    "repairer": r, "frames": len(fids), **summarize(rows),
                                    "repair_ms": float(np.mean([recs[f]["repair_ms"][(eff, lv, r)] for f in fids]))})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "summary.csv", index=False)

    # Repairer per weather family, chosen on the training towns by mean mAP50 over its effects and levels.
    choice = {}
    for d in DETECTORS:
        tr = summary[(summary.detector == d) & (summary.split == "train") & (summary.level > 0)]
        means = tr.groupby(["family", "repairer"]).map50.mean().unstack()
        choice[d] = {CLEAN: "none", **{fam: str(means.loc[fam].idxmax()) for fam in ("rain", "dust")}}

    # Condition detectors on car frames (per frame decision, threshold 0.6).
    routers = {n: PerturbationDetector.load(args.detectors / f"detector_{n}.joblib")
               for n in ROUTERS if (args.detectors / f"detector_{n}.joblib").exists()}
    decided, routing_rows = {}, []
    for n, det in routers.items():
        for eff, lv in VARIANTS:
            x = np.stack([recs[f]["features"][(eff, lv)] for f in frames.frame])
            p = det.predict_proba(x)
            top, conf = det.classes[p.argmax(1)], p.max(1)
            dec = np.where((top != CLEAN) & (conf >= 0.6), top, CLEAN)
            decided[(n, eff, lv)] = dict(zip(frames.frame, dec))
            for split, mask in (("train", ~test), ("test", test)):
                routing_rows.append({"router": n, "split": split, "effect": eff, "family": FAMILY[eff], "level": lv,
                                     "correct": float((dec[mask] == FAMILY[eff]).mean()),
                                     **{f"as_{c}": float((dec[mask] == c).mean()) for c in FAMILIES}})
    routing = pd.DataFrame(routing_rows)
    routing.to_csv(args.out / "routing_car.csv", index=False)

    # Pipelines on the held-out towns.
    fids = list(frames.frame[test])
    blocks = blocks_of(frames[test])
    pipes, rows_out = {}, []
    for d in DETECTORS:
        pipes = {"none": lambda f, e, l: "none",
                 "oracle": lambda f, e, l: choice[d][FAMILY[e]],
                 f"always_{choice[d]['rain']}": lambda f, e, l: choice[d]["rain"],
                 f"always_{choice[d]['dust']}": lambda f, e, l: choice[d]["dust"],
                 **{f"routed_{n}": (lambda n: lambda f, e, l: choice[d][decided[(n, e, l)][f]])(n) for n in routers}}
        for name, pick in pipes.items():
            for eff, lv in VARIANTS:
                rows = stats_for(fids, lambda f: (eff, lv, pick(f, eff, lv)), d)
                s = summarize(rows)
                lo, hi = block_ci(lambda idx: path_recall(rows, idx), blocks, args.n_boot, args.seed)
                mlo, mhi = block_ci(lambda idx: map50_rows(rows, idx), blocks, args.n_boot, args.seed)
                rows_out.append({"detector": d, "pipeline": name, "effect": eff, "family": FAMILY[eff], "level": lv,
                                 **s, "path_recall_low": lo, "path_recall_high": hi, "map50_low": mlo, "map50_high": mhi})
    pipelines = pd.DataFrame(rows_out)
    pipelines.to_csv(args.out / "pipelines.csv", index=False)

    # Paired change vs. no repair (same frames, same blocks) for the routed pipeline with each router.
    deltas = []
    for d in DETECTORS:
        for n in ["oracle"] + [f"routed_{n}" for n in routers]:
            for fam in ("rain", "dust"):
                for lv in LEVELS:
                    effs = [e for e in WEATHER if FAMILY[e] == fam]
                    def stacked(pipe):
                        rows = []
                        for e in effs:
                            pick = (lambda f: choice[d][fam]) if pipe == "oracle" else \
                                   (lambda f: "none") if pipe == "none" else \
                                   (lambda f: choice[d][decided[(pipe[7:], e, lv)][f]])
                            rows += stats_for(fids, lambda f: (e, lv, pick(f)), d)
                        return rows
                    a, b = stacked(n), stacked("none")
                    bl = np.concatenate([blocks + f"|{e}" for e in effs])
                    for m, fn in (("path_recall", path_recall), ("map50", map50_rows)):
                        diff = lambda idx: fn(a, idx) - fn(b, idx)
                        lo, hi = block_ci(diff, bl, args.n_boot, args.seed)
                        deltas.append({"detector": d, "pipeline": n, "family": fam, "level": lv, "measure": m,
                                       "change": diff(np.arange(len(a))), "low": lo, "high": hi})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(args.out / "deltas.csv", index=False)

    # The car's detector: best mean in-path recall on the training towns, with the chosen repairs.
    tr = summary[summary.split == "train"]
    with_repair = tr[tr.apply(lambda r: r.repairer == choice[r.detector][r.family], axis=1)]
    car_detector = str(with_repair.groupby("detector").path_recall.mean().idxmax())

    tst = summary[summary.split == "test"]
    headline = {
        "car_detector": car_detector,
        "car_router": CAR_ROUTER,
        "train_path_recall_with_repair": with_repair.groupby("detector").path_recall.mean().round(3).to_dict(),
        "frames": {"train": int((~test).sum()), "test": int(test.sum())},
        "score_thresholds": thresholds,
        "repairer_choice": choice,
        "weather_hurts": {d: {f"{fam}_L{lv}": round(float(tst[(tst.detector == d) & (tst.repairer == "none") & (tst.family == fam)
                                                               & (tst.level == lv)].map50.mean()
                                                       - tst[(tst.detector == d) & (tst.repairer == "none") & (tst.level == 0)].map50.mean()), 3)
                              for fam in ("rain", "dust") for lv in LEVELS} for d in DETECTORS},
        "clean_map50_test": {d: round(float(tst[(tst.detector == d) & (tst.repairer == "none") & (tst.level == 0)].map50.iloc[0]), 3)
                             for d in DETECTORS},
        "router_correct_test": routing[routing.split == "test"].groupby(["router", "family"]).correct.mean().round(3)
                               .unstack().to_dict("index"),
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(json.dumps(headline, indent=2, default=str))
        print(deltas.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR / "carla_ego")
    p.add_argument("--detectors", type=Path, default=RESULTS_DIR / "weather_detector")
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--max-side", type=int, default=640, help="resolution the learned repairers run at")
    p.add_argument("--stage", choices=("all", "infer", "analyze"), default="all")
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_perception")
    args = p.parse_args()
    m = pd.read_csv(args.data / "manifest.csv")
    frames = m[m.frame.str[-4:].astype(int) % args.every == 0].reset_index(drop=True)
    print(f"{len(frames)} frames ({frames.town.isin(TEST_TOWNS).sum()} in held-out towns)")
    if args.stage in ("all", "infer"):
        infer(args, frames)
    if args.stage in ("all", "analyze"):
        analyze(args, frames)


if __name__ == "__main__":
    main()
