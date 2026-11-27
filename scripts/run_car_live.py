"""Phase 2.4 of the driving study: is the pipeline fast enough to drive on? Two modes.

--benchmark (no CARLA): time every stage on recorded held-out car frames under weather:
  routing, each repairer at several working resolutions (--sizes; 0 = full 1280x720),
  each detector, isolation. Repairers and detectors are timed on the GPU, warmed up first.

live (default; needs a CARLA server): the car drives Town10HD (held out from every choice) on
autopilot while the weather changes on a schedule; each camera frame is weathered (physics
effects, exact depth: the simulator's side, not timed as pipeline) and passed through the full
pipeline: router -> repair -> detector -> isolation. The simulator waits for the pipeline
(synchronous mode), so timings are the pipeline's own. Writes a side-by-side video: the
weathered camera view with the route, and the repaired view with the isolated objects (red =
in the car's path, thick = most critical).

Pipeline settings come from results/car_perception/headline.json (repairer per condition,
detector threshold) and results/car_routing/routing_time.csv (router and smoothing), unless given.

Usage:
  python scripts/run_car_live.py --benchmark
  python scripts/run_car_live.py --seconds 90
Outputs (in --out): benchmark.csv; live_frames.csv, live_summary.json, live.mp4
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from bolero.car_perception import CameraModel, CarDetector, CarPerception, ConditionRouter, isolate
from bolero.config import DATA_DIR, RESULTS_DIR, SEED
from bolero.detection import CLEAN, PerturbationDetector
from bolero.ego import apply_weather, decode_depth
from bolero.learned_restoration import build

from run_car_perception import TEST_TOWNS, load_frame

LIVE_SCHEDULE = [("clean", 0, 10), ("rain_streaks", 2, 10), ("rain_mist", 3, 10), ("clean", 0, 8),
                 ("dust_haze", 2, 10), ("dust_storm", 3, 10), ("clean", 0, 8), ("rain_lens", 2, 10),
                 ("dust_haze", 3, 10), ("clean", 0, 4)]


def chosen_settings(args) -> dict:
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    det = args.detector or head.get("car_detector", "detr")
    s = {"detector": det, "score": head["score_thresholds"][det], "repairers": head["repairer_choice"][det]}
    rt = pd.read_csv(RESULTS_DIR / "car_routing" / "routing_time.csv")
    router = args.router or head.get("car_router", "combined")
    row = rt[(rt.router == router) & (rt.route == "smoothed") & (rt.split == "test")].iloc[0]
    s.update(router=router, alpha=float(row.alpha), enter=float(row.enter), exit=float(row.exit))
    return s


def sync(device_type="cuda"):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *a, repeat=1):
    sync()
    t = time.perf_counter()
    for _ in range(repeat):
        out = fn(*a)
    sync()
    return out, 1e3 * (time.perf_counter() - t) / repeat


def benchmark(args):
    m = pd.read_csv(args.data / "manifest.csv")
    fids = list(m[m.town.isin(TEST_TOWNS) & (m.frame.str[-4:].astype(int) % 10 == 0)].frame)[: args.frames]
    weathers = [("rain_mist", 2), ("dust_haze", 2), ("rain_streaks", 3), ("dust_storm", 2)]
    frames = []
    for i, f in enumerate(fids):
        img, depth, _ = load_frame(args.data, f)
        eff, lv = weathers[i % len(weathers)]
        frames.append(apply_weather(img, depth, eff, lv, f, args.seed))
    router = ConditionRouter(PerturbationDetector.load(args.detectors / "detector_combined.joblib"))
    rows = []
    for x in frames[:2]:
        router(x)
    rows.append({"stage": "route", "variant": "rf_320", "ms": np.median([timed(router, x)[1] for x in frames])})
    for kind in ("detr", "maskrcnn"):
        det = CarDetector(kind)
        det(frames[0])
        rows.append({"stage": "detect", "variant": kind, "ms": np.median([timed(det, x)[1] for x in frames])})
        d0 = det(frames[0])
        rows.append({"stage": "isolate", "variant": kind,
                     "ms": np.median([timed(isolate, d0, CameraModel())[1] for _ in frames])})
    for name in ("autocontrast", "dcp_dehaze", "restormer_derain", "instructir_derain", "instructir_dehaze"):
        for size in args.sizes if name.startswith(("restormer", "instructir")) else [0]:
            rep = build(name, max_side=size or None)
            rep(frames[0])
            rows.append({"stage": "repair", "variant": f"{name}@{size or 'full'}",
                         "ms": np.median([timed(rep, x)[1] for x in frames])})
            print(rows[-1], flush=True)
            del rep
    del router
    for kind in ("maskrcnn",):
        det = CarDetector(kind, fp16=True)
        det(frames[0])
        rows.append({"stage": "detect", "variant": f"{kind}_fp16", "ms": np.median([timed(det, x)[1] for x in frames])})
    fast = build("dcp_dehaze", max_side=640)
    rows.append({"stage": "repair", "variant": "dcp_dehaze@640", "ms": np.median([timed(fast, x)[1] for x in frames])})
    fast_router = ConditionRouter(PerturbationDetector.load(args.detectors / "detector_combined_car.joblib"))
    fast_router(frames[0])
    rows.append({"stage": "route", "variant": "rf_320_single_job", "ms": np.median([timed(fast_router, x)[1] for x in frames])})
    out = pd.DataFrame(rows)
    out["fps_alone"] = 1000.0 / out.ms
    out.to_csv(args.out / "benchmark.csv", index=False)
    print(out.round(1).to_string(index=False))
    fidelity(args)


def fidelity(args):
    """Do the speed settings (dehazing at 640 px, Mask R-CNN in fp16) change detection? Held-out
    evaluation frames, every weather variant: in-path recall and mAP50 with the evaluated settings
    (full-resolution dehazing, fp32; from run_car_perception.py) and with the fast ones."""
    import pickle

    from run_car_perception import (TEST_TOWNS as TT, VARIANTS, frame_stats, gt_of, map50_rows,
                                    path_recall)
    from bolero.detection_eval import Detections

    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    det_name = head["car_detector"]
    choice, score = head["repairer_choice"][det_name], head["score_thresholds"][det_name]
    m = pd.read_csv(args.data / "manifest.csv")
    fids = list(m[m.town.isin(TT) & (m.frame.str[-4:].astype(int) % 5 == 0)].frame)
    fast_det = CarDetector(det_name, fp16=True)
    fast_rep = {"rain": build(choice["rain"], max_side=640), "dust": build(choice["dust"], max_side=640)}
    camera = CameraModel()
    rows = []
    for eff, lv in VARIANTS[1:]:
        fam = "rain" if eff.startswith("rain") else "dust"
        slow, quick = [], []
        for f in fids:
            img, depth, lab = load_frame(args.data, f)
            gtb, _, objs = gt_of(lab)
            rec = pickle.load(open(RESULTS_DIR / "car_perception" / "infer" / f"{f}.pkl", "rb"))
            slow.append(frame_stats(gtb, objs, Detections(*rec["dets"][(eff, lv, choice[fam], det_name)]), score, camera))
            d = fast_det(fast_rep[fam](apply_weather(img, depth, eff, lv, f, args.seed)))
            quick.append(frame_stats(gtb, objs, d, score, camera))
        idx = np.arange(len(fids))
        rows.append({"effect": eff, "level": lv, "path_recall_eval": path_recall(slow, idx),
                     "path_recall_fast": path_recall(quick, idx), "map50_eval": map50_rows(slow, idx),
                     "map50_fast": map50_rows(quick, idx)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "fidelity.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("mean change (fast - evaluated): path_recall %.4f, map50 %.4f" % (
        (out.path_recall_fast - out.path_recall_eval).mean(), (out.map50_fast - out.map50_eval).mean()))


def overlay(weathered, result, info, size=(640, 360)):
    a = cv2.resize((weathered * 255).astype(np.uint8), size)
    b = (result["repaired"] * 255).astype(np.uint8).copy()
    for o in result["objects"]:
        x1, y1, x2, y2 = map(int, o["box"])
        color = (255, 40, 40) if o["in_path"] else (60, 220, 60)
        cv2.rectangle(b, (x1, y1), (x2, y2), color, 6 if o["critical"] else 2)
        cv2.putText(b, f'{o["label"]} {o["distance"]:.0f}m', (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    b = cv2.resize(b, size)
    frame = np.hstack([a, b])
    bar = np.full((40, frame.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, info, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    return cv2.cvtColor(np.vstack([frame, bar]), cv2.COLOR_RGB2BGR)


def live(args):
    import carla

    from record_ego_drive import FOV, HEIGHT, WIDTH, Drive

    s = chosen_settings(args)
    print("settings:", s, flush=True)
    router = ConditionRouter(PerturbationDetector.load(args.detectors / f"detector_{s['router']}.joblib"),
                             alpha=s["alpha"], enter=s["enter"], exit=s["exit"])
    repairers = {fam: build(name, max_side=args.max_side) for fam, name in s["repairers"].items() if name != "none"}
    detector = CarDetector(s["detector"], fp16=True)
    pipeline = CarPerception(router, repairers, detector, CameraModel(WIDTH, HEIGHT, FOV, 1.4, -2.0), s["score"],
                             async_route=True)

    client = carla.Client("localhost", args.port)
    client.set_timeout(120.0)
    rec = Drive(client, argparse.Namespace(seed=args.seed, cross=0.3, vehicles=50, walkers=120, camera_height=1.4))
    rec.load(args.town)
    rec.populate(45.0, 150.0)
    rec.mount()
    rec.queues.pop("instance")
    for sensor in rec.sensors:
        if sensor.type_id == "sensor.camera.instance_segmentation":
            sensor.stop()
    for _ in range(100):
        rec.tick()
    writer = cv2.VideoWriter(str(args.out / "live.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (1280, 400))
    timeline = [seg for seg in LIVE_SCHEDULE for _ in range(int(seg[2] * 20))]
    n = min(len(timeline), int(args.seconds * 20))
    rows = []
    try:
        for k in range(0, n, args.every):
            for _ in range(args.every - 1):
                rec.world.tick()
                for q in rec.queues.values():
                    while not q.empty():
                        q.get()
            data = rec.tick()
            eff, lv, _ = timeline[k]
            rgb = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)[..., 2::-1].astype(np.float32) / 255.0
            depth = decode_depth(np.frombuffer(data["depth"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4))
            w = apply_weather(rgb, depth, eff, lv, f"live{k}", args.seed)
            sync()
            res = pipeline(w)
            sync()
            truth = CLEAN if eff == "clean" else ("rain" if eff.startswith("rain") else "dust")
            crit = next((o for o in res["objects"] if o["critical"]), None)
            rows.append({"t": k / 20.0, "effect": eff, "level": lv, "truth": truth, "route": res["route"],
                         "objects": len(res["objects"]), "in_path": sum(o["in_path"] for o in res["objects"]),
                         "critical_m": crit["distance"] if crit else np.nan, **{f"ms_{a}": b for a, b in res["ms"].items()}})
            info = (f"t={k / 20:5.1f}s  weather: {eff} L{lv}  route: {res['route']}  "
                    f"pipeline {res['ms']['total']:.0f} ms ({1000 / res['ms']['total']:.0f} fps)"
                    + (f"  critical: {crit['label']} {crit['distance']:.0f} m" if crit else ""))
            writer.write(overlay(w, res, info))
    finally:
        writer.release()
        rec.cleanup()
        st = rec.world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = False, None
        rec.world.apply_settings(st)
        rec.tm.set_synchronous_mode(False)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "live_frames.csv", index=False)
    summ = {"settings": s, "max_side": args.max_side, "frames": len(df),
            "ms_median": {c[3:]: float(df[c].median()) for c in df.columns if c.startswith("ms_")},
            "ms_p95": {c[3:]: float(df[c].quantile(0.95)) for c in df.columns if c.startswith("ms_")},
            "fps_median": float(1000.0 / df.ms_total.median()),
            "share_frames_over_100ms": float((df.ms_total > 100).mean()),
            "route_accuracy": float((df.route == df.truth).mean()),
            "ms_total_by_route": df.groupby("route").ms_total.median().round(1).to_dict()}
    (args.out / "live_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--data", type=Path, default=DATA_DIR / "carla_ego")
    p.add_argument("--detectors", type=Path, default=RESULTS_DIR / "weather_detector")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--sizes", type=int, nargs="+", default=[0, 960, 640, 480])
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--detector", default=None)
    p.add_argument("--router", default=None)
    p.add_argument("--town", default="Town10HD_Opt")
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--every", type=int, default=2, help="process one camera frame every N ticks (2 = 10 Hz)")
    p.add_argument("--video-fps", type=float, default=10.0)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_live")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark(args) if args.benchmark else live(args)


if __name__ == "__main__":
    main()
