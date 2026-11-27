"""Record paired clean / rain / dust traffic-camera frames in CARLA, with exact boxes and depth.

Each scene is one instant at a fixed, elevated traffic camera. Every actor is frozen, the scene
is rendered once per weather setting, then traffic moves on to the next instant. So the seven
frames of a scene (clean, rain levels 1-3, dust levels 1-3) differ only in the weather.

Boxes come from CARLA's instance-segmentation camera, rendered with the clean frame: the visible
extent of each actor, labelled with the six weather-track classes (CARLA's "Rider" counts as
person, as in COCO). Tiny or mostly hidden objects are dropped (--min-box, --min-pixels).
Depth is CARLA's ground-truth depth in metres, saved for the clean frame.

Requires a running CARLA 0.9.16 server, e.g.
  ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000

Outputs (in --out, default data/carla/, git-ignored):
  images/<scene>__<condition>__<level>.jpg   level 0 = clean
  depth/<scene>.png                           16-bit, depth in centimetres (clipped at 655 m)
  labels/<scene>.json                         boxes (x1 y1 x2 y2 px), labels, width, height, map, camera
  manifest.csv                                one row per image

Usage:
  python scripts/record_carla_weather.py --pilot           # 1 scene, for checking the setup
  python scripts/record_carla_weather.py --scenes 300
"""

import argparse
import json
import queue
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from bolero.config import DATA_DIR, SEED

import carla  # noqa: E402  (installed from the CARLA release's PythonAPI/carla/dist wheel)

TAG_CLASS = {12: "person", 13: "person", 14: "car", 15: "truck", 16: "bus", 18: "motorcycle", 19: "bicycle"}
WIDTH, HEIGHT, FOV = 1280, 720, 90
LEVELS = (1, 2, 3)


# Rain levels are CARLA's own SoftRainNoon / MidRainyNoon / HardRainNoon presets. CARLA cannot
# render yellow-brown dust: its dust_storm parameter mostly greys the sky and its fog has no
# colour control, so "dust" here is the dust_storm setting plus grey fog thickening by level.
RAIN = {1: dict(cloudiness=20, precipitation=30, precipitation_deposits=50, fog_density=3, wind_intensity=30),
        2: dict(cloudiness=60, precipitation=60, precipitation_deposits=60, fog_density=3, wind_intensity=60),
        3: dict(cloudiness=100, precipitation=100, precipitation_deposits=90, fog_density=7, wind_intensity=100)}
DUST = {1: dict(cloudiness=50, dust_storm=100, fog_density=20, fog_distance=0, mie_scattering_scale=1.0, wind_intensity=60),
        2: dict(cloudiness=70, dust_storm=100, fog_density=40, fog_distance=0, mie_scattering_scale=2.0, wind_intensity=80),
        3: dict(cloudiness=90, dust_storm=100, fog_density=65, fog_distance=0, mie_scattering_scale=3.0, wind_intensity=100)}


def weather(condition: str, level: int, sun: float, azimuth: float) -> carla.WeatherParameters:
    """Clean, rain or dust at a level; the sun stays where the scene put it."""
    w = carla.WeatherParameters(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0,
                                sun_altitude_angle=sun, sun_azimuth_angle=azimuth, fog_density=0.0,
                                fog_distance=0.75, fog_falloff=0.1, wind_intensity=10.0, dust_storm=0.0)
    for k, v in {"rain": RAIN, "dust": DUST}.get(condition, {}).get(level, {}).items():
        setattr(w, k, float(v))
    return w


def conditions():
    yield "clean", 0
    for c in ("rain", "dust"):
        for lv in LEVELS:
            yield c, lv


def boxes_from_instances(inst: np.ndarray, min_box: int, min_pixels: int):
    """inst: (H, W, 4) BGRA from the instance camera (R = semantic tag, G/B = instance id)."""
    tag = inst[..., 2].astype(np.int32)
    ident = inst[..., 1].astype(np.int32) | (inst[..., 0].astype(np.int32) << 8)
    boxes, labels = [], []
    for t, name in TAG_CLASS.items():
        mask = tag == t
        if not mask.any():
            continue
        ids = ident[mask]
        ys, xs = np.nonzero(mask)
        for i in np.unique(ids):
            sel = ids == i
            if sel.sum() < min_pixels:
                continue
            x1, x2, y1, y2 = xs[sel].min(), xs[sel].max() + 1, ys[sel].min(), ys[sel].max() + 1
            if (x2 - x1) < min_box or (y2 - y1) < min_box:
                continue
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            labels.append(name)
    return boxes, labels


class Recorder:
    def __init__(self, client: carla.Client, args):
        self.client, self.args = client, args
        self.rng = random.Random(args.seed)
        self.world = None
        self.actors, self.controllers, self.sensors = [], [], []

    # --- world setup -----------------------------------------------------------------------
    def load(self, town: str):
        self.cleanup()
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.args.seed)
        self.world.set_pedestrians_seed(self.args.seed)
        self.world.tick()

    def cleanup(self):
        for s in self.sensors:
            try:
                s.stop()
            except RuntimeError:
                pass
        for cid in self.controllers:
            c = self.world.get_actor(cid) if self.world is not None else None
            if c is not None:
                try:
                    c.stop()
                except RuntimeError:
                    pass
        if self.world is not None and (self.sensors or self.actors or self.controllers):
            ids = [s.id for s in self.sensors] + self.controllers + self.actors
            ids = [i for i in ids if self.world.get_actor(i) is not None]
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids])
            self.world.tick()
        self.actors, self.controllers, self.sensors = [], [], []

    def camera_poses(self, n: int):
        """Elevated cameras beside the road, looking along it (like a pole-mounted traffic camera)."""
        points = self.world.get_map().get_spawn_points()
        self.rng.shuffle(points)
        poses = []
        for sp in points:
            fwd, right = sp.get_forward_vector(), sp.get_right_vector()
            loc = sp.location - fwd * 12.0 + right * 4.0
            loc.z += self.rng.uniform(5.0, 9.0)
            rot = carla.Rotation(pitch=self.rng.uniform(-22.0, -12.0), yaw=sp.rotation.yaw + self.rng.uniform(-15, 15))
            if all(loc.distance(p.location) > 40.0 for p in poses):
                poses.append(carla.Transform(loc, rot))
            if len(poses) == n:
                break
        return poses

    def populate(self, cam: carla.Transform):
        """Vehicles at the spawn points nearest the camera, pedestrians on nearby walkable ground."""
        bl = self.world.get_blueprint_library()
        vehicles = [b for b in bl.filter("vehicle.*") if b.has_attribute("number_of_wheels")]
        points = sorted(self.world.get_map().get_spawn_points(), key=lambda p: p.location.distance(cam.location))
        batch = []
        for sp in points[: self.args.vehicles]:
            b = self.rng.choice(vehicles)
            if b.has_attribute("color"):
                b.set_attribute("color", self.rng.choice(b.get_attribute("color").recommended_values))
            batch.append(carla.command.SpawnActor(b, sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, 8000)))
        self.actors += [r.actor_id for r in self.client.apply_batch_sync(batch, True) if not r.error]

        walkers = bl.filter("walker.pedestrian.*")
        spots = []
        for _ in range(4000):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None and loc.distance(cam.location) < 45.0:
                spots.append(loc)
            if len(spots) >= self.args.walkers:
                break
        res = self.client.apply_batch_sync([carla.command.SpawnActor(self.rng.choice(walkers), carla.Transform(l + carla.Location(z=1.0)))
                                            for l in spots], True)
        ids = [r.actor_id for r in res if not r.error]
        ctrl_bp = bl.find("controller.ai.walker")
        res = self.client.apply_batch_sync([carla.command.SpawnActor(ctrl_bp, carla.Transform(), i) for i in ids], True)
        self.controllers = [r.actor_id for r in res if not r.error]
        self.actors += ids
        self.world.tick()
        for cid in self.controllers:
            c = self.world.get_actor(cid)
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + self.rng.random())

    def mount(self, cam: carla.Transform):
        bl = self.world.get_blueprint_library()
        self.queues = {}
        for kind in ("rgb", "depth", "instance"):
            bp = bl.find({"rgb": "sensor.camera.rgb", "depth": "sensor.camera.depth",
                          "instance": "sensor.camera.instance_segmentation"}[kind])
            bp.set_attribute("image_size_x", str(WIDTH))
            bp.set_attribute("image_size_y", str(HEIGHT))
            bp.set_attribute("fov", str(FOV))
            sensor = self.world.spawn_actor(bp, cam)
            q = queue.Queue()
            sensor.listen(q.put)
            self.sensors.append(sensor)
            self.queues[kind] = q

    def tick(self) -> dict:
        frame = self.world.tick()
        out = {}
        for kind, q in self.queues.items():
            while True:
                data = q.get(timeout=10.0)
                if data.frame == frame:
                    out[kind] = data
                    break
        return out

    def freeze(self, frozen: bool):
        """Stop (or restart) every actor. CARLA can remove an actor at any time (e.g. a vehicle that
        leaves the map), so actors that are gone are dropped instead of failing the scene."""
        alive = []
        for aid in self.actors:
            a = self.world.get_actor(aid)
            if a is None:
                continue
            try:
                if a.type_id.startswith("vehicle."):
                    if frozen:
                        a.set_autopilot(False, 8000)
                    a.set_simulate_physics(not frozen)
                    if not frozen:
                        a.set_autopilot(True, 8000)
                else:
                    a.set_simulate_physics(not frozen)
                alive.append(aid)
            except RuntimeError:
                continue
        self.actors = alive
        controllers = []
        for cid in self.controllers:
            c = self.world.get_actor(cid)
            if c is None:
                continue
            try:
                if frozen:
                    c.stop()
                else:
                    c.start()
                    c.go_to_location(self.world.get_random_location_from_navigation())
                controllers.append(cid)
            except RuntimeError:
                continue
        self.controllers = controllers

    # --- recording -------------------------------------------------------------------------
    def record_scene(self, scene: str, town: str, cam: carla.Transform, out: Path):
        sun, azimuth = self.rng.uniform(15.0, 70.0), self.rng.uniform(0.0, 360.0)
        self.freeze(True)
        rows, frames = [], {}
        for cond, lv in conditions():
            self.world.set_weather(weather(cond, lv, sun, azimuth))
            for _ in range(self.args.settle):
                self.tick()
            frames[(cond, lv)] = self.tick()
        self.freeze(False)

        clean = frames[("clean", 0)]
        inst = np.frombuffer(clean["instance"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)
        boxes, labels = boxes_from_instances(inst, self.args.min_box, self.args.min_pixels)
        if len(boxes) < self.args.min_objects:
            return []
        for (cond, lv), f in frames.items():
            rgb = np.frombuffer(f["rgb"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3]
            path = out / "images" / f"{scene}__{cond}__{lv}.jpg"
            cv2.imwrite(str(path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"scene": scene, "town": town, "condition": cond, "level": lv, "path": str(path),
                         "objects": len(boxes)})
        d = np.frombuffer(clean["depth"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4).astype(np.float64)
        metres = (d[..., 2] + d[..., 1] * 256 + d[..., 0] * 65536) / (256 ** 3 - 1) * 1000.0
        cv2.imwrite(str(out / "depth" / f"{scene}.png"), np.clip(metres * 100, 0, 65535).astype(np.uint16))
        (out / "labels" / f"{scene}.json").write_text(json.dumps({
            "boxes": boxes, "labels": labels, "width": WIDTH, "height": HEIGHT, "town": town,
            "camera": {"x": cam.location.x, "y": cam.location.y, "z": cam.location.z,
                       "pitch": cam.rotation.pitch, "yaw": cam.rotation.yaw, "fov": FOV},
            "sun_altitude": sun, "sun_azimuth": azimuth}))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--towns", nargs="+", default=["Town10HD_Opt", "Town03_Opt", "Town05_Opt", "Town01_Opt", "Town02_Opt"])
    parser.add_argument("--scenes", type=int, default=300)
    parser.add_argument("--cameras-per-town", type=int, default=12)
    parser.add_argument("--instants", type=int, default=5, help="scenes recorded per camera position")
    parser.add_argument("--gap", type=int, default=60, help="ticks (0.05 s) of traffic between instants")
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--settle", type=int, default=3, help="ticks after a weather change before capture")
    parser.add_argument("--vehicles", type=int, default=30)
    parser.add_argument("--walkers", type=int, default=25)
    parser.add_argument("--min-box", type=int, default=12)
    parser.add_argument("--min-pixels", type=int, default=150)
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--pilot", action="store_true", help="one scene in the first town")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "carla")
    args = parser.parse_args()
    if args.pilot:
        args.scenes, args.towns, args.cameras_per_town, args.instants = 1, args.towns[:1], 3, 1

    for sub in ("images", "depth", "labels"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    rows = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    done = {r["scene"] for r in rows}

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    rec = Recorder(client, args)
    per_town = int(np.ceil(args.scenes / len(args.towns)))
    start = time.time()
    try:
        for town in args.towns:
            todo = [f"{town.split('_')[0].lower()}_c{c:02d}_t{t}" for c in range(args.cameras_per_town)
                    for t in range(args.instants)][:per_town]
            if all(s in done for s in todo):
                continue
            rec.load(town)
            recorded = sum(1 for r in rows if r["town"] == town and r["condition"] == "clean")
            for c, cam in enumerate(rec.camera_poses(args.cameras_per_town)):
                if recorded >= per_town:
                    break
                try:
                    rec.cleanup()
                    rec.populate(cam)
                    rec.mount(cam)
                    for _ in range(args.warmup):
                        rec.tick()
                    for t in range(args.instants):
                        scene = f"{town.split('_')[0].lower()}_c{c:02d}_t{t}"
                        if scene not in done:
                            new = rec.record_scene(scene, town, cam, args.out)
                            rows += new
                            recorded += bool(new)
                            pd.DataFrame(rows).to_csv(manifest, index=False)
                            print(f"{scene}: {new[0]['objects'] if new else 0} objects, "
                                  f"{sum(1 for r in rows if r['condition'] == 'clean')} scenes, "
                                  f"{(time.time() - start) / 60:.1f} min", flush=True)
                        for _ in range(args.gap):
                            rec.tick()
                except RuntimeError as e:  # CARLA occasionally fails a tick on an actor it removed itself
                    print(f"camera {c} in {town} abandoned: {e}", flush=True)
                    try:
                        rec.cleanup()
                    except RuntimeError:
                        rec.actors, rec.controllers, rec.sensors = [], [], []
            rec.cleanup()
    finally:
        rec.cleanup()
        if rec.world is not None:
            settings = rec.world.get_settings()
            settings.synchronous_mode, settings.fixed_delta_seconds = False, None
            rec.world.apply_settings(settings)
            rec.tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
