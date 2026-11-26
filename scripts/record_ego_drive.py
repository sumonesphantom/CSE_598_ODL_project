"""Record what a car's front camera sees while it drives through CARLA traffic (clean weather),
with exact boxes, distances and depth for every frame.

The car (a Lincoln MKZ) drives on Traffic Manager autopilot among other vehicles and
pedestrians; a front camera at bumper height (1280x720, 90 deg FOV), a depth camera and an
instance-segmentation camera share one pose. Frames are kept every --every ticks (0.05 s).

Weather is NOT rendered by CARLA: CARLA's rain and dust barely affect a detector (CARLA.md), so
the driving study adds the project's physics weather to these clean frames afterwards, from the
exact depth saved here (bolero.ego.apply_weather).

Ground truth per object (bolero.ego.objects_from_instances): visible box, label (six weather-track
classes, riders = person), distance and lateral offset from the camera, and whether it is in
the car's path (ahead, within 2 m sideways and 40 m). The car's own body is excluded.

Requires a running CARLA 0.9.16 server, e.g.
  ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000

Outputs (in --out, default data/carla_ego/, git-ignored):
  images/<drive>_<frame>.png     clean RGB (PNG, so no compression artefacts the weather detector could use)
  depth/<drive>_<frame>.png      16-bit centimetres, half resolution (for weather synthesis)
  labels/<drive>_<frame>.json    objects, ego speed, camera
  manifest.csv                   one row per frame: drive, town, frame, time, path, objects, in_path, speed

Usage:
  python scripts/record_ego_drive.py --pilot
  python scripts/record_ego_drive.py
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
from bolero.ego import decode_depth, objects_from_instances

import carla  # noqa: E402

WIDTH, HEIGHT, FOV = 1280, 720, 90
EGO = "vehicle.lincoln.mkz_2020"


class Drive:
    def __init__(self, client, args):
        self.client, self.args = client, args
        self.rng = random.Random(args.seed)
        self.world, self.ego = None, None
        self.spawn_skip = 0
        self.actors, self.controllers, self.sensors = [], [], []

    def load(self, town, tm_port=8000):
        """Load a town with a fresh Traffic Manager. A Traffic Manager keeps the road map of the
        town it was created in, so reusing one after load_world steers cars by the old town's
        roads (into buildings); each town therefore gets its own port."""
        self.cleanup()
        self.world = self.client.load_world(town)
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
        self.world.apply_settings(s)
        self.tm_port = tm_port
        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.args.seed)
        self.world.set_pedestrians_seed(self.args.seed)
        self.world.set_pedestrians_cross_factor(self.args.cross)
        self.world.tick()

    def cleanup(self):
        for s in self.sensors:
            try:
                s.stop()
            except RuntimeError:
                pass
        if self.world is not None:
            for cid in self.controllers:
                c = self.world.get_actor(cid)
                if c is not None:
                    try:
                        c.stop()
                    except RuntimeError:
                        pass
            ids = [s.id for s in self.sensors] + self.controllers + self.actors
            ids = [i for i in ids if self.world.get_actor(i) is not None]
            if ids:
                self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids])
                self.world.tick()
        self.actors, self.controllers, self.sensors, self.ego = [], [], [], None

    def populate(self, sun, azimuth):
        bl = self.world.get_blueprint_library()
        self.world.set_weather(carla.WeatherParameters(cloudiness=10.0, precipitation=0.0, wetness=0.0,
                                                       sun_altitude_angle=sun, sun_azimuth_angle=azimuth,
                                                       fog_density=0.0, wind_intensity=10.0))
        points = self.world.get_map().get_spawn_points()
        self.rng.shuffle(points)
        ego_bp = bl.find(EGO)
        ego_bp.set_attribute("role_name", "hero")
        skip = self.spawn_skip
        for sp in points[skip:] + points[:skip]:
            self.ego = self.world.try_spawn_actor(ego_bp, sp)
            if self.ego:
                break
        self.actors.append(self.ego.id)
        self.ego.set_autopilot(True, self.tm_port)
        self.tm.vehicle_percentage_speed_difference(self.ego, -10.0)
        here = self.ego.get_transform().location
        rest = sorted([p for p in points if p.location.distance(here) > 8.0], key=lambda p: p.location.distance(here))
        vehicles = [b for b in bl.filter("vehicle.*") if b.has_attribute("number_of_wheels")]
        batch = []
        for sp in rest[: self.args.vehicles]:
            b = self.rng.choice(vehicles)
            if b.has_attribute("color"):
                b.set_attribute("color", self.rng.choice(b.get_attribute("color").recommended_values))
            batch.append(carla.command.SpawnActor(b, sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)))
        self.actors += [r.actor_id for r in self.client.apply_batch_sync(batch, True) if not r.error]
        spots = [self.world.get_random_location_from_navigation() for _ in range(self.args.walkers)]
        res = self.client.apply_batch_sync([carla.command.SpawnActor(self.rng.choice(bl.filter("walker.pedestrian.*")),
                                                                     carla.Transform(l + carla.Location(z=1.0)))
                                            for l in spots if l is not None], True)
        walkers = [r.actor_id for r in res if not r.error]
        res = self.client.apply_batch_sync([carla.command.SpawnActor(bl.find("controller.ai.walker"), carla.Transform(), i)
                                            for i in walkers], True)
        self.controllers = [r.actor_id for r in res if not r.error]
        self.actors += walkers
        self.world.tick()
        for cid in self.controllers:
            c = self.world.get_actor(cid)
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + self.rng.random())

    def mount(self):
        bl = self.world.get_blueprint_library()
        ext = self.ego.bounding_box.extent
        pose = carla.Transform(carla.Location(x=ext.x + 0.05, z=self.args.camera_height), carla.Rotation(pitch=-2.0))
        self.queues = {}
        for kind, bp_name in (("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth"),
                              ("instance", "sensor.camera.instance_segmentation")):
            bp = bl.find(bp_name)
            bp.set_attribute("image_size_x", str(WIDTH))
            bp.set_attribute("image_size_y", str(HEIGHT))
            bp.set_attribute("fov", str(FOV))
            sensor = self.world.spawn_actor(bp, pose, attach_to=self.ego)
            q = queue.Queue()
            sensor.listen(q.put)
            self.sensors.append(sensor)
            self.queues[kind] = q

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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--towns", nargs="+", default=["Town10HD_Opt", "Town03_Opt", "Town05_Opt", "Town01_Opt", "Town02_Opt"])
    p.add_argument("--drives-per-town", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60.0, help="recorded length of each drive")
    p.add_argument("--every", type=int, default=10, help="keep one frame every N ticks (10 = 2 Hz)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--vehicles", type=int, default=50)
    p.add_argument("--walkers", type=int, default=120)
    p.add_argument("--cross", type=float, default=0.3, help="share of pedestrians that cross roads")
    p.add_argument("--camera-height", type=float, default=1.4)
    p.add_argument("--min-box", type=int, default=12)
    p.add_argument("--min-pixels", type=int, default=150)
    p.add_argument("--min-moving", type=float, default=0.5,
                   help="re-drive (new start) when the car moves in less than this share of frames")
    p.add_argument("--tries", type=int, default=4)
    p.add_argument("--pilot", action="store_true", help="one 10 s drive in the first town")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--out", type=Path, default=DATA_DIR / "carla_ego")
    args = p.parse_args()
    if args.pilot:
        args.towns, args.drives_per_town, args.seconds = args.towns[:1], 1, 10.0

    for sub in ("images", "depth", "labels"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    rows = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    done = {r["drive"] for r in rows}
    client = carla.Client("localhost", args.port)
    client.set_timeout(120.0)
    rec = Drive(client, args)
    start = time.time()
    try:
        for town in args.towns:
            short = town.split("_")[0].lower()
            todo = [f"{short}_d{d}" for d in range(args.drives_per_town)]
            if all(d in done for d in todo):
                continue
            rec.load(town, 8000 + args.towns.index(town))
            for drive in todo:
                sun, azimuth = rec.rng.uniform(15.0, 70.0), rec.rng.uniform(0.0, 360.0)
                if drive in done:
                    continue
                for attempt in range(args.tries):
                    new = record_drive(rec, args, town, drive, sun, azimuth)
                    moving = np.mean([r["speed"] > 0.5 for r in new]) if new else 0.0
                    if moving >= args.min_moving:
                        break
                    print(f"{drive}: car moving in only {moving:.0%} of frames, new start", flush=True)
                    for r in new:
                        for sub, ext in (("images", "png"), ("depth", "png"), ("labels", "json")):
                            (args.out / sub / f"{r['frame']}.{ext}").unlink(missing_ok=True)
                    new = []
                    rec.spawn_skip += 7
                rows += new
                pd.DataFrame(rows).to_csv(manifest, index=False)
                print(f"{drive}: {len(new)} frames, {sum(r['objects'] for r in new)} objects, "
                      f"{sum(r['in_path'] for r in new)} in path, moving {moving:.0%}, "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
            rec.cleanup()
    finally:
        try:
            rec.cleanup()
        except RuntimeError:
            pass
        if rec.world is not None:
            s = rec.world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds = False, None
            rec.world.apply_settings(s)
            rec.tm.set_synchronous_mode(False)


def record_drive(rec, args, town, drive, sun, azimuth):
    """One drive in the loaded town; returns its manifest rows (frames already written)."""
    new = []
    try:
        rec.cleanup()
        rec.populate(sun, azimuth)
        rec.mount()
        for _ in range(args.warmup):
            rec.tick()
        for k in range(int(args.seconds / 0.05)):
            data = rec.tick()
            if k % args.every:
                continue
            fid = f"{drive}_{k // args.every:04d}"
            rgb = np.frombuffer(data["rgb"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3]
            depth = decode_depth(np.frombuffer(data["depth"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4))
            inst = np.frombuffer(data["instance"].raw_data, np.uint8).reshape(HEIGHT, WIDTH, 4)
            objs = objects_from_instances(inst, depth, FOV, args.min_box, args.min_pixels, exclude_ids=[rec.ego.id])
            v = rec.ego.get_velocity()
            speed = float(np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
            cv2.imwrite(str(args.out / "images" / f"{fid}.png"), rgb)
            half = cv2.resize(depth, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(args.out / "depth" / f"{fid}.png"), np.clip(half * 100, 0, 65535).astype(np.uint16))
            (args.out / "labels" / f"{fid}.json").write_text(json.dumps({
                "objects": objs, "width": WIDTH, "height": HEIGHT, "fov": FOV, "town": town,
                "camera_height": args.camera_height, "speed": speed,
                "sun_altitude": sun, "sun_azimuth": azimuth}))
            new.append({"drive": drive, "town": town, "frame": fid, "time": round(k * 0.05, 2),
                        "path": str(args.out / "images" / f"{fid}.png"), "objects": len(objs),
                        "in_path": sum(o["in_path"] for o in objs), "speed": round(speed, 2)})
    except RuntimeError as e:
        print(f"{drive} stopped early: {e}", flush=True)
    return new


if __name__ == "__main__":
    main()
