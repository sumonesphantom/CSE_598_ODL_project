"""Phases 3 and 4 of the driving study: does the weather pipeline make the car drive more safely?

Phase 3 (the agent and the scenarios). The car drives a straight stretch of road at 36 km/h with
the driving agent of bolero.driving (lane following from the map; braking only from perception).
Four scenarios, each staged at the same --locations per town in the held-out towns (Town10HD,
Town02), on straight road (junctions crossed straight over), with nothing else moving:
  pedestrian_cross  a pedestrian 2 m beyond the right edge of the car's lane (kerb, or the next lane
                    on wider roads) walks into the lane as the car approaches and stops in its middle
                    (1.6 m/s, starting when the car is 35 m away)
  stopped_car       a car stands still in the lane 70 m ahead
  lead_brake        a car drives ahead at the same speed, 30 m in front, and brakes at 6 m/s^2
                    to a stop 3 s into the run
  adjacent_car      a car stands still in the neighbouring lane 70 m ahead (the car should NOT
                    brake: this measures false braking)

Phase 4 (the experiment). Every scenario and location is driven under each weather variant (clean,
and every v2 weather effect at levels 2 and 3; bolero.ego.weather_v2) with each perception pipeline:
  gt        ground-truth perception (CARLA's instance labels and depth): the upper bound, blind to weather
  none      the car's detector (Mask R-CNN) on the weathered frame, no repair
  routed    the full algorithm: condition router -> repairer chosen for that condition -> detector
Perception runs at 10 Hz and its decision takes effect 0.1 s later in every pipeline.
Measures per run (bolero.driving.run_measures, from simulator ground truth): collision, impact
speed, minimum gap, minimum time-to-collision, where braking began, seconds spent braking.

Requires a running CARLA 0.9.16 server, e.g.  ./CarlaUE4.sh -RenderOffScreen -nosound
Usage:
  python scripts/run_car_scenarios.py --stage pilot      # Phase 3 check: each scenario once, gt + none, clean; videos
  python scripts/run_car_scenarios.py --stage check      # ground truth at every location: the routes are sound
  python scripts/run_car_scenarios.py                    # Phase 4 (resumable), then analysis
  python scripts/run_car_scenarios.py --stage analyze
  python scripts/run_car_scenarios.py --stage demo --demo-runs RUN_ID ...   # re-drive runs with video; demo/demo.mp4
Outputs (in --out): runs.csv, traces/<run>.csv.gz, locations.json, pilot/*.mp4, and from the
analysis: outcomes.csv, paired.csv, headline.json
"""

import argparse
import json
import queue
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import carla
import cv2
import numpy as np
import pandas as pd

from bolero.car_perception import (
    CameraModel,
    CarDetector,
    CarPerception,
    ConditionRouter,
    isolate,
)
from bolero.config import RESULTS_DIR, SEED
from bolero.detection import PerturbationDetector
from bolero.driving import BrakePolicy, gap_and_offset, pure_pursuit, run_measures
from bolero.ego import apply_weather, decode_depth, objects_from_instances
from bolero.learned_restoration import build

WIDTH, HEIGHT, FOV = 1280, 720, 90
EGO = "vehicle.lincoln.mkz_2020"
TOWNS = ("Town10HD_Opt", "Town02_Opt")
SCENARIOS = ("pedestrian_cross", "stopped_car", "lead_brake", "adjacent_car")
EFFECTS = ("rain_streaks", "rain_mist", "rain_lens", "dust_haze", "dust_storm")
VARIANTS = [("clean", 0)] + [(e, lv) for e in EFFECTS for lv in (2, 3)]
PIPELINES = ("gt", "none", "routed")
DT, PERCEIVE_EVERY, LATENCY_TICKS = 0.05, 2, 2
SPEED = 10.0
HAZARD_AT, LEAD_AT, TRIGGER = 70.0, 30.0, 35.0
MAX_SECONDS = 16.0
ROUTE_TOLERANCE = 1.5      # m: a run whose car strays further from the planned route is invalid


# --- locations ---------------------------------------------------------------------------------
STEP = 2.0                 # m between route points


def _offset(origin, loc) -> float:
    """Sideways distance of `loc` from the straight line through `origin` along its heading."""
    f = origin.transform.get_forward_vector()
    o = origin.transform.location
    return abs(f.x * (loc.y - o.y) - f.y * (loc.x - o.x))


def straight_path(start_wp, length: float, step: float = STEP, look: float = 12.0):
    """Waypoints every `step` m along the straightest way ahead, crossing junctions straight over.
    At a branch, each option is followed `look` m further and the one that stays closest to the
    start's heading line is taken (a turning lane and the straight lane look alike for the first
    metres inside a junction). Returns None if the road ends."""
    path, cur = [start_wp], start_wp
    while len(path) * step <= length:
        nx = cur.next(step)
        if not nx:
            return None
        if len(nx) > 1:
            def score(n):
                far = n
                for _ in range(int(look / step)):
                    more = far.next(step)
                    if not more:
                        break
                    far = min(more, key=lambda m: _offset(start_wp, m.transform.location))
                return _offset(start_wp, far.transform.location)
            nx = [min(nx, key=score)]
        cur = nx[0]
        path.append(cur)
    return path


def straight_starts(world, n: int, seed: int, length: float = 110.0, min_sep: float = 60.0, tolerance: float = 0.5):
    """Start waypoints with >= `length` m of straight road ahead (junctions may be crossed straight
    over; every route point within `tolerance` m of the start's heading line), whose hazard point
    (HAZARD_AT ahead) is outside a junction and has a driving lane on its left."""
    m = world.get_map()
    wps = [w for w in m.generate_waypoints(4.0) if w.lane_type == carla.LaneType.Driving and not w.is_junction]
    wps.sort(key=lambda w: (w.road_id, w.section_id, w.lane_id, round(w.s, 1)))
    good = []
    for w in wps:
        path = straight_path(w, length)
        if path is None or max(_offset(w, p.transform.location) for p in path) > tolerance:
            continue
        hz = path[round(HAZARD_AT / STEP)]
        left = hz.get_left_lane()
        if hz.is_junction or left is None or left.lane_type != carla.LaneType.Driving:
            continue
        good.append(w)
    rng = np.random.default_rng(seed)
    chosen = []
    for i in rng.permutation(len(good)):
        w = good[i]
        if all(w.transform.location.distance(c.transform.location) > min_sep for c in chosen):
            chosen.append(w)
        if len(chosen) == n:
            break
    return chosen


def route_target(route: np.ndarray, xy, ahead: float = 8.0, step: float = STEP):
    """(look-ahead point, distance from the route) for a position."""
    d = np.hypot(*(route - np.asarray(xy, float)).T)
    i = int(d.argmin())
    a = min(i, len(route) - 2)                    # the route segment next to the closest point
    seg, rel = route[a + 1] - route[a], np.asarray(xy, float) - route[a]
    dev = abs(seg[0] * rel[1] - seg[1] * rel[0]) / max(np.hypot(*seg), 1e-6)
    return route[min(i + round(ahead / step), len(route) - 1)], float(dev)


# --- one run -----------------------------------------------------------------------------------
class Stage:
    """Spawns the car, its sensors and the scenario's actors, and steps the scenario script."""

    def __init__(self, world, start_wp, scenario: str, want_instances: bool, seed: int):
        self.world, self.scenario = world, scenario
        self.map = world.get_map()
        bl = world.get_blueprint_library()
        self.actors, self.sensors = [], []
        tf = start_wp.transform
        self.ego = world.spawn_actor(bl.find(EGO), carla.Transform(tf.location + carla.Location(z=0.3), tf.rotation))
        self.actors.append(self.ego)
        path = straight_path(start_wp, HAZARD_AT + 60.0)
        self.route = np.array([[p.transform.location.x, p.transform.location.y] for p in path])
        self.hazard_wp = path[round(HAZARD_AT / STEP)]
        self.hazard, self.walker_target, self.started, self.stopped = None, None, False, False
        car_bp = bl.find("vehicle.tesla.model3")
        car_bp.set_attribute("color", "40,40,160")
        if scenario in ("stopped_car", "adjacent_car"):
            wp = self.hazard_wp if scenario == "stopped_car" else self.hazard_wp.get_left_lane()
            t = wp.transform
            self.hazard = world.spawn_actor(car_bp, carla.Transform(t.location + carla.Location(z=0.3), t.rotation))
        elif scenario == "lead_brake":
            t = path[round(LEAD_AT / STEP)].transform
            self.hazard = world.spawn_actor(car_bp, carla.Transform(t.location + carla.Location(z=0.3), t.rotation))
        elif scenario == "pedestrian_cross":
            t = self.hazard_wp.transform
            right = t.get_right_vector()
            off = self.hazard_wp.lane_width / 2.0 + 2.0
            walker_bp = bl.filter("walker.pedestrian.0001")[0]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            for extra in (0.0, -0.5, 0.5, -1.0):
                loc = t.location + right * (off + extra) + carla.Location(z=1.0)
                self.hazard = world.try_spawn_actor(walker_bp, carla.Transform(loc, carla.Rotation(yaw=t.rotation.yaw - 90.0)))
                if self.hazard is not None:
                    break
            if self.hazard is None:
                raise RuntimeError("could not place the pedestrian")
            self.walker_dir = carla.Vector3D(-right.x, -right.y, 0.0)
        self.actors.append(self.hazard)
        ext = self.ego.bounding_box.extent
        pose = carla.Transform(carla.Location(x=ext.x + 0.05, z=1.4), carla.Rotation(pitch=-2.0))
        self.queues = {}
        kinds = [("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth")]
        if want_instances:
            kinds.append(("instance", "sensor.camera.instance_segmentation"))
        for kind, name in kinds:
            bp = bl.find(name)
            for k, v in (("image_size_x", WIDTH), ("image_size_y", HEIGHT), ("fov", FOV)):
                bp.set_attribute(k, str(v))
            s = world.spawn_actor(bp, pose, attach_to=self.ego)
            q = queue.Queue()
            s.listen(q.put)
            self.sensors.append(s)
            self.queues[kind] = q
        self.collisions = []
        cs = world.spawn_actor(bl.find("sensor.other.collision"), carla.Transform(), attach_to=self.ego)
        cs.listen(lambda e: self.collisions.append(e.other_actor.type_id))
        self.sensors.append(cs)

    def launch(self):
        """Give the car (and a lead car) their starting speed."""
        f = self.ego.get_transform().get_forward_vector()
        self.ego.set_target_velocity(carla.Vector3D(f.x * SPEED, f.y * SPEED, 0.0))
        if self.scenario == "lead_brake":
            self.hazard.set_target_velocity(carla.Vector3D(f.x * SPEED, f.y * SPEED, 0.0))
        if self.scenario in ("stopped_car", "adjacent_car"):
            self.hazard.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))

    def script(self, t: float, gap: float):
        if self.scenario == "lead_brake":
            v = SPEED if t < 3.0 else max(0.0, SPEED - 6.0 * (t - 3.0))
            tf = self.hazard.get_transform()
            nxt, _ = route_target(self.route, [tf.location.x, tf.location.y], 6.0)
            steer = pure_pursuit([tf.location.x, tf.location.y], tf.rotation.yaw, nxt)
            f = tf.get_forward_vector()
            self.hazard.set_target_velocity(carla.Vector3D(f.x * v, f.y * v, 0.0))
            self.hazard.apply_control(carla.VehicleControl(steer=steer, brake=1.0 if v == 0 else 0.0))
        elif self.scenario == "pedestrian_cross":
            if not self.started and gap <= TRIGGER:
                self.started = True
            if self.started and not self.stopped:
                c = self.hazard_wp.transform
                rel = self.hazard.get_location() - c.location
                r = c.get_right_vector()
                if rel.x * r.x + rel.y * r.y <= 0.0:          # reached the lane centre
                    self.stopped = True
            speed = 1.6 if self.started and not self.stopped else 0.0
            self.hazard.apply_control(carla.WalkerControl(direction=self.walker_dir, speed=speed))

    def tick(self):
        frame = self.world.tick()
        out = {}
        for kind, q in self.queues.items():
            while True:
                data = q.get(timeout=10.0)
                if data.frame == frame:
                    out[kind] = data
                    break
        return out

    def destroy(self, client):
        for s in self.sensors:
            try:
                s.stop()
            except RuntimeError:
                pass
        client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in self.sensors + self.actors if a is not None])
        self.world.tick()


def hazard_state(stage: Stage):
    e, h = stage.ego.get_transform(), stage.hazard.get_transform()
    eh = stage.ego.bounding_box.extent.x
    hh = stage.hazard.bounding_box.extent.x if stage.scenario != "pedestrian_cross" else 0.3
    gap, lat = gap_and_offset([e.location.x, e.location.y], e.rotation.yaw, eh, [h.location.x, h.location.y], hh)
    f = e.get_forward_vector()
    hv = stage.hazard.get_velocity()
    return gap, lat, hv.x * f.x + hv.y * f.y


def drive(client, world, start_wp, scenario, variant, pipeline, perceive, run_id, args, video=None):
    stage = Stage(world, start_wp, scenario, pipeline == "gt", args.seed)
    policy = BrakePolicy(target_speed=SPEED)
    pending, react = [], None
    trace, routes = [], []
    eff, lv = variant
    in_path_width = stage.ego.bounding_box.extent.y + (stage.hazard.bounding_box.extent.y if scenario != "pedestrian_cross" else 0.3) + 0.2
    try:
        stage.tick()
        stage.launch()
        stopped_for, coll_tick, max_dev = 0, None, 0.0
        for k in range(int(MAX_SECONDS / DT)):
            t = k * DT
            gap, lat, hz = hazard_state(stage)
            stage.script(t, gap)
            ego_tf = stage.ego.get_transform()
            v = stage.ego.get_velocity()
            speed = float(np.hypot(v.x, v.y))
            ahead, dev = route_target(stage.route, [ego_tf.location.x, ego_tf.location.y])
            steer = pure_pursuit([ego_tf.location.x, ego_tf.location.y], ego_tf.rotation.yaw, ahead)
            if coll_tick is None:
                max_dev = max(max_dev, dev)
            while pending and pending[0][0] <= k:
                react = pending.pop(0)[1]
            throttle, brake = policy.control(speed, react)
            stage.ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
            if getattr(args, "show", False):         # CARLA window: chase camera behind the car
                back = ego_tf.location - ego_tf.get_forward_vector() * 9.0 + carla.Location(z=4.0)
                world.get_spectator().set_transform(carla.Transform(back, carla.Rotation(pitch=-14.0, yaw=ego_tf.rotation.yaw)))
            data = stage.tick()
            if k % PERCEIVE_EVERY == 0:
                rgb = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)[..., 2::-1].astype(np.float32) / 255.0
                depth = decode_depth(np.frombuffer(data["depth"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4))
                if pipeline == "gt":
                    inst = np.frombuffer(data["instance"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)
                    objects, route, shown = objects_from_instances(inst, depth, FOV, exclude_ids=[stage.ego.id]), "gt", rgb
                else:
                    w = apply_weather(rgb, depth, eff, lv, f"{run_id}_{k}", args.seed, model=args.weather_model, t=t)
                    objects, route, shown = perceive(w)
                routes.append(route)
                d = policy.observe(objects, PERCEIVE_EVERY * DT, speed)
                pending.append((k + LATENCY_TICKS, d))
                if video is not None:
                    video.write(frame_overlay(shown, objects, f"{run_id}  t={t:4.1f}s  {speed * 3.6:4.1f} km/h  "
                                              f"brake {brake:.2f}  route {route}  gap {gap:5.1f} m"))
            coll = bool(stage.collisions)
            trace.append({"t": t, "speed": speed, "throttle": throttle, "brake": brake, "gap": gap, "lateral": lat,
                          "hazard_speed": hz, "collision": coll, "react": react if react is not None else np.nan})
            if coll and coll_tick is None:
                coll_tick = k
            stopped_for = stopped_for + 1 if speed < 0.2 else 0
            if (coll_tick is not None and k - coll_tick >= 10) or stopped_for >= 20 or gap < -12.0:
                break
    finally:
        stage.destroy(client)
    m = run_measures(trace, in_path_width)
    m["collided_with"] = stage.collisions[0] if stage.collisions else ""
    m["seconds"] = trace[-1]["t"]
    m["route_deviation"] = max_dev        # largest distance from the planned route before any collision
    if routes and pipeline != "gt":
        r = pd.Series(routes)
        m.update({f"route_{c}": float((r == c).mean()) for c in ("clean", "rain", "dust")})
    return m, pd.DataFrame(trace)


def frame_overlay(image, objects, info):
    b = (np.clip(image, 0, 1) * 255).astype(np.uint8).copy()
    for o in objects:
        x1, y1, x2, y2 = map(int, o["box"])
        color = (255, 40, 40) if o["in_path"] else (60, 220, 60)
        cv2.rectangle(b, (x1, y1), (x2, y2), color, 3)
        cv2.putText(b, f'{o["label"]} {o["distance"]:.0f}m', (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    b = cv2.resize(b, (960, 540))
    bar = np.full((36, 960, 3), 20, np.uint8)
    cv2.putText(bar, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return cv2.cvtColor(np.vstack([b, bar]), cv2.COLOR_RGB2BGR)


# --- perception pipelines ----------------------------------------------------------------------
def settings():
    head = json.loads((RESULTS_DIR / "car_perception" / "headline.json").read_text())
    det = head["car_detector"]
    choice_path = RESULTS_DIR / "car_repair" / "choice.json"
    repairers = json.loads(choice_path.read_text())["repairers"] if choice_path.exists() else head["repairer_choice"][det]
    return {"detector": det, "score": head["score_thresholds"][det], "router": head["car_router"],
            "repairers": repairers, "repairers_from": str(choice_path if choice_path.exists() else "car_perception/headline.json")}


def make_pipelines(s, args):
    detector = CarDetector(s["detector"], fp16=True)
    camera = CameraModel(WIDTH, HEIGHT, FOV, 1.4, -2.0)
    router = ConditionRouter(PerturbationDetector.load(args.detectors / f"detector_{s['router']}.joblib"),
                             alpha=1.0, enter=0.6, exit=0.6)
    repairers = {fam: build(name, max_side=640 if name.startswith(("restormer", "instructir")) else None)
                 for fam, name in s["repairers"].items() if name != "none"}
    full = CarPerception(router, repairers, detector, camera, s["score"])

    def none(w):
        return isolate(detector(w), camera, s["score"]), "clean", w

    def routed(w):
        r = full(w)
        return r["objects"], r["route"], r["repaired"]

    def reset():
        router.reset()

    return {"none": none, "routed": routed, "gt": None}, reset


# --- analysis ----------------------------------------------------------------------------------
def analyze(args):
    runs = pd.read_csv(args.out / "runs.csv")
    # A location is dropped if the car left its planned route in any run there (the driving agent,
    # not perception, failed): such runs would count crashes into scenery as perception failures.
    off = runs[runs.route_deviation > ROUTE_TOLERANCE][["town", "location"]].drop_duplicates()
    dropped = [f"{t}_L{loc}" for t, loc in off.itertuples(index=False)]
    if dropped:
        print("dropped locations (car left the route):", dropped)
    runs = runs[~(runs.town + "_L" + runs.location.astype(str)).isin(dropped)].copy()
    runs["weather"] = np.where(runs.effect == "clean", "clean", runs.effect.str.split("_").str[0])
    hazard = runs[runs.scenario != "adjacent_car"]
    gt = hazard[hazard.pipeline == "gt"].set_index(["scenario", "town", "location"])
    rows = []
    for (weather, pipe), g in hazard[hazard.pipeline != "gt"].groupby(["weather", "pipeline"]):
        rows.append({"weather": weather, "pipeline": pipe, "runs": len(g), "collision_rate": g.collision.mean(),
                     "mean_impact_kmh": 3.6 * g.impact_speed.mean(), "median_min_gap": g.min_gap.median(),
                     "near_miss_rate": ((~g.collision) & (g.min_gap < 2.0)).mean(),
                     "median_brake_onset_gap": g.brake_onset_gap.median()})
    for weather in ("clean", "rain", "dust"):
        rows.append({"weather": weather, "pipeline": "gt", "runs": len(gt), "collision_rate": gt.collision.mean(),
                     "mean_impact_kmh": 3.6 * gt.impact_speed.mean(), "median_min_gap": gt.min_gap.median(),
                     "near_miss_rate": ((~gt.collision) & (gt.min_gap < 2.0)).mean(),
                     "median_brake_onset_gap": gt.brake_onset_gap.median()})
    outcomes = pd.DataFrame(rows).sort_values(["weather", "pipeline"])
    outcomes.to_csv(args.out / "outcomes.csv", index=False)
    by_level = (hazard[hazard.pipeline != "gt"].groupby(["effect", "level", "pipeline"])
                .agg(runs=("collision", "size"), collision_rate=("collision", "mean"), median_min_gap=("min_gap", "median"))
                .reset_index())
    by_level.to_csv(args.out / "by_effect.csv", index=False)
    by_scen = (runs.groupby(["scenario", "weather", "pipeline"])
               .agg(runs=("collision", "size"), collision_rate=("collision", "mean"), median_min_gap=("min_gap", "median"),
                    brake_seconds=("brake_seconds", "mean"), min_speed=("min_speed", "mean")).reset_index())
    by_scen.to_csv(args.out / "by_scenario.csv", index=False)

    # Paired comparison: the same scenario, location and weather, driven with and without repair.
    key = ["scenario", "town", "location", "effect", "level"]
    a = hazard[hazard.pipeline == "none"].set_index(key)
    b = hazard[hazard.pipeline == "routed"].set_index(key)
    pair = a[["collision", "min_gap", "impact_speed"]].join(b[["collision", "min_gap", "impact_speed"]], lsuffix="_none",
                                                            rsuffix="_routed", how="inner").reset_index()
    pair["weather"] = np.where(pair.effect == "clean", "clean", pair.effect.str.split("_").str[0])
    pair.to_csv(args.out / "paired.csv", index=False)
    rng = np.random.default_rng(args.seed)
    head = {"settings": settings(), "weather_model": args.weather_model, "dropped_locations": dropped, "paired": {}}
    for weather, g in pair.groupby("weather"):
        diff = g.collision_routed.astype(float).to_numpy() - g.collision_none.astype(float).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        head["paired"][weather] = {
            "pairs": len(g), "collision_none": float(g.collision_none.mean()), "collision_routed": float(g.collision_routed.mean()),
            "change": float(diff.mean()), "change_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "saved": int(((g.collision_none) & (~g.collision_routed)).sum()),
            "caused": int(((~g.collision_none) & (g.collision_routed)).sum())}
    adj = runs[runs.scenario == "adjacent_car"]
    head["false_braking"] = (adj.groupby(["weather", "pipeline"]).agg(runs=("brake_seconds", "size"), brake_seconds=("brake_seconds", "mean"),
                                                                     braked_share=("brake_seconds", lambda s: (s > 0).mean()))
                             .round(2).reset_index().to_dict("records"))
    (args.out / "headline.json").write_text(json.dumps(head, indent=2, default=float))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(outcomes.round(3).to_string(index=False))
        print(json.dumps(head["paired"], indent=2))


class ShowWhileWriting:
    """A video writer that also shows each frame in a window (--show)."""

    def __init__(self, writer, title: str):
        self.writer, self.title = writer, title
        cv2.namedWindow("car camera + perception", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("car camera + perception", 960, 576)

    def write(self, frame):
        self.writer.write(frame)
        cv2.imshow("car camera + perception", frame)
        cv2.waitKey(1)

    def release(self):
        self.writer.release()


RUN_ID = re.compile(r"^(Town\w+?_Opt)_L(\d+)_(" + "|".join(SCENARIOS) + r")_([a-z_]+?)(\d)_(" + "|".join(PIPELINES) + r")$")


def parse_run_id(run_id: str):
    m = RUN_ID.match(run_id)
    if m is None:
        raise ValueError(f"not a run id: {run_id}")
    town, li, scen, eff, lv, pipe = m.groups()
    return town, int(li), scen, eff, int(lv), pipe


def stitch_demo(args):
    """demo/demo.mp4: for each demo case, the unrepaired car (left) beside the full algorithm
    (right), same weather and scenario, with each run's outcome in a title strip."""
    folder = args.out / "demo"
    runs = pd.read_csv(folder / "runs.csv").set_index("run")
    cases = sorted({r.rsplit("_", 1)[0] for r in runs.index})
    out = None
    for case in cases:
        pair = [f"{case}_none", f"{case}_routed"]
        if not all((folder / f"{r}.mp4").exists() and r in runs.index for r in pair):
            continue
        clips = []
        for r in pair:
            cap, frames = cv2.VideoCapture(str(folder / f"{r}.mp4")), []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            m = runs.loc[r]
            verdict = f"COLLISION at {3.6 * m.impact_speed:.0f} km/h" if m.collision else f"stopped safely, {m.min_gap:.1f} m short"
            name = "no repair" if r.endswith("_none") else "our algorithm (route + repair)"
            clips.append((frames, f"{name}: {verdict}", (40, 40, 220) if m.collision else (60, 180, 60)))
        n = max(len(c[0]) for c in clips)
        for k in range(n + 15):                         # hold the last frame for 1.5 s
            tiles = []
            for frames, title, color in clips:
                f = frames[min(k, len(frames) - 1)]
                strip = np.full((40, f.shape[1], 3), 30, np.uint8)
                cv2.putText(strip, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                tiles.append(np.vstack([strip, f]))
            frame = np.hstack(tiles)
            head = np.full((44, frame.shape[1], 3), 0, np.uint8)
            cv2.putText(head, case.replace("_", " "), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            frame = np.vstack([head, frame])
            if out is None:
                out = cv2.VideoWriter(str(folder / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, frame.shape[1::-1])
            out.write(frame)
    if out is not None:
        out.release()
        print("wrote", folder / "demo.mp4")


# --- main --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", "pilot", "check", "demo", "run", "analyze"), default="all")
    p.add_argument("--locations", type=int, default=4, help="locations per town")
    p.add_argument("--towns", nargs="+", default=list(TOWNS))
    p.add_argument("--show", action="store_true", help="pilot / demo: show the drive live (chase camera in the "
                   "CARLA window, perception view in a second window) while recording")
    p.add_argument("--demo-runs", nargs="+", default=[], help="run ids to re-drive with video (--stage demo)")
    p.add_argument("--weather-model", default="v2")
    p.add_argument("--detectors", type=Path, default=RESULTS_DIR / "weather_detector")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "car_scenarios")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "analyze":
        analyze(args)
        return

    demo = [parse_run_id(r) for r in args.demo_runs]
    if args.stage == "demo":
        args.towns = [t for t in args.towns if any(d[0] == t for d in demo)]
    s = settings()
    print("settings:", s, flush=True)
    perceive, reset = make_pipelines(s, args)
    client = carla.Client("localhost", args.port)
    client.set_timeout(120.0)
    runs_path = args.out / "runs.csv"
    side = args.stage in ("pilot", "check", "demo")
    rows = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() and not side else []
    done = {r["run"] for r in rows}
    (args.out / "traces").mkdir(exist_ok=True)
    loc_info = {}
    for ti, town in enumerate(args.towns):
        world = client.load_world(town)
        st = world.get_settings()
        st.synchronous_mode, st.fixed_delta_seconds = True, DT
        world.apply_settings(st)
        world.set_weather(carla.WeatherParameters(cloudiness=10.0, precipitation=0.0, wetness=0.0, sun_altitude_angle=45.0,
                                                  sun_azimuth_angle=150.0, fog_density=0.0))
        world.tick()
        starts = straight_starts(world, args.locations if args.stage != "pilot" else 1, args.seed)
        loc_info[town] = [[w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw] for w in starts]
        print(town, "locations:", len(starts), flush=True)
        plan = []
        for li, wp in enumerate(starts):
            for scen in SCENARIOS:
                if args.stage == "demo":
                    plan += [(li, wp, scen, (e, lv), pipe) for t, dl, ds, e, lv, pipe in demo if (t, dl, ds) == (town, li, scen)]
                    continue
                if args.stage == "check":         # ground truth only: are the locations and routes sound?
                    plan.append((li, wp, scen, ("clean", 0), "gt"))
                    continue
                if args.stage == "pilot":
                    plan += [(li, wp, scen, ("clean", 0), pipe) for pipe in ("gt", "none")]
                    plan += [(li, wp, scen, v, "none") for v in (("dust_haze", 3), ("rain_mist", 3))]
                    plan += [(li, wp, scen, v, "routed") for v in (("dust_haze", 3), ("rain_mist", 3))]
                    continue
                plan.append((li, wp, scen, ("clean", 0), "gt"))
                plan += [(li, wp, scen, v, pipe) for v in VARIANTS for pipe in ("none", "routed")]
        start = time.time()
        for n, (li, wp, scen, (eff, lv), pipe) in enumerate(plan):
            run_id = f"{town}_L{li}_{scen}_{eff}{lv}_{pipe}"
            if run_id in done:
                continue
            reset()
            video = None
            if args.stage in ("pilot", "demo"):
                (args.out / args.stage).mkdir(exist_ok=True)
                video = cv2.VideoWriter(str(args.out / args.stage / f"{run_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (960, 576))
                if args.show:
                    video = ShowWhileWriting(video, run_id)
            try:
                m, trace = drive(client, world, wp, scen, (eff, lv), pipe, perceive[pipe], run_id, args, video)
            except RuntimeError as e:
                print("skipped", run_id, e, flush=True)
                continue
            finally:
                if video is not None:
                    video.release()
            trace.to_csv(args.out / "traces" / f"{run_id}.csv.gz", index=False)
            row = {"run": run_id, "town": town, "location": li, "scenario": scen, "effect": eff, "level": lv, "pipeline": pipe, **m}
            rows.append(row)
            if side:
                (args.out / args.stage).mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out / args.stage / "runs.csv" if side else runs_path, index=False)
            print(f"[{n + 1}/{len(plan)} {time.time() - start:.0f}s] {run_id}: collision={m['collision']} dev={m['route_deviation']:.2f} "
                  f"min_gap={m['min_gap']:.1f} onset_gap={m['brake_onset_gap']:.1f} brake_s={m['brake_seconds']:.1f}", flush=True)
        st.synchronous_mode, st.fixed_delta_seconds = False, None
        world.apply_settings(st)
    (args.out / "locations.json").write_text(json.dumps(loc_info, indent=2))
    if args.stage == "all":
        analyze(args)
    if args.stage == "demo":
        stitch_demo(args)


if __name__ == "__main__":
    main()
