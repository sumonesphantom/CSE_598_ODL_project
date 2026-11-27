"""Phase 3 of the driving study: a car that drives on what its perception reports.

The driving agent is deliberately simple and the same for every perception pipeline, so any
difference in safety between runs comes from perception alone:

  * lateral: pure pursuit on the lane's centre line (map waypoints; perception plays no part);
  * longitudinal: cruise at a target speed, and brake for the objects perception puts in the car's
    path. Every in-path object is treated as standing still (no tracking), so the deceleration
    needed to stop `margin` metres short of the nearest one is a = v^2 / (2 (d - margin)); the
    brake is applied in proportion once a exceeds `comfort` (a light, early brake) and fully when
    the object is inside the margin.
  * an object must be reported in two perception frames in a row before the car reacts (one
    spurious detection does not cause a hard stop), and a reported object is remembered for
    `memory` seconds (its distance shrinking by the distance driven) if it drops out of view.

Perception runs at 10 Hz and the decision it leads to is applied `latency` seconds later, the
same for every pipeline, including the ground-truth one.

Objects may carry a `source`. One reported only by the depth obstacle warning (source "depth",
bolero.car_depth) is a weaker signal than a detection, so braking for it alone is capped at
`warn_cap` of full braking (a gentle, early slow-down); as soon as the detector also reports an
object in the path, full braking is allowed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrakePolicy:
    target_speed: float = 10.0     # m/s (36 km/h, urban)
    margin: float = 5.0            # m: stop this far short of the object (camera is at the bumper)
    comfort: float = 1.0           # m/s^2: needed deceleration below this is ignored
    max_decel: float = 8.0         # m/s^2 at full brake (CARLA Lincoln MKZ, dry road)
    confirm: int = 2               # consecutive perception frames before reacting
    memory: float = 0.5            # s an object is remembered after it drops out
    warn_cap: float = 0.4          # brake fraction allowed for a depth-warning-only object (~3 m/s^2)
    streak: int = 0
    strong: bool = True            # the reacted-to object is backed by the detector (not depth alone)
    held: float | None = None      # remembered distance of the nearest in-path object
    held_age: float = 0.0
    log: list = field(default_factory=list)

    def reset(self):
        self.streak, self.held, self.held_age, self.strong = 0, None, 0.0, True

    def observe(self, objects: list[dict], dt_since_last: float, speed: float) -> float | None:
        """Update with one perception frame; returns the distance the car reacts to (or None)."""
        path = [o["distance"] for o in objects if o["in_path"]]
        detected = any(o["in_path"] and o.get("source", "detector") != "depth" for o in objects)
        if self.held is not None:
            self.held -= speed * dt_since_last
            self.held_age += dt_since_last
        if path:
            self.streak += 1
            if self.streak >= self.confirm:
                self.held, self.held_age, self.strong = min(path), 0.0, detected
        else:
            self.streak = 0
            if self.held is not None and self.held_age > self.memory:
                self.held = None
        return self.held

    def control(self, speed: float, distance: float | None, strong: bool = True) -> tuple[float, float]:
        """(throttle, brake) for the current speed and reacted-to distance; `strong` False caps the
        brake at `warn_cap` (the object comes from the depth warning alone)."""
        if distance is not None:
            room = distance - self.margin
            need = np.inf if room <= 0.3 else speed ** 2 / (2.0 * room)
            if need >= self.comfort:
                brake = float(np.clip(need / self.max_decel, 0.15, 1.0)) if np.isfinite(need) else 1.0
                return 0.0, brake if strong else min(brake, self.warn_cap)
            if room < 2.0 * speed:      # close but no braking needed yet: do not speed up
                return 0.0, 0.0
        err = self.target_speed - speed
        return (float(np.clip(0.3 + 0.4 * err, 0.0, 0.8)), 0.0) if err > 0 else (0.0, float(np.clip(-0.2 * err, 0.0, 0.3)))


def pure_pursuit(ego_xy: np.ndarray, ego_yaw_deg: float, target_xy: np.ndarray, wheelbase: float = 2.9,
                 max_steer_deg: float = 70.0) -> float:
    """Steering command in [-1, 1] towards a look-ahead point (CARLA: +y is to the right)."""
    d = np.asarray(target_xy, float) - np.asarray(ego_xy, float)
    yaw = np.deg2rad(ego_yaw_deg)
    alpha = np.arctan2(d[1], d[0]) - yaw
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    ld = max(np.hypot(*d), 1e-3)
    delta = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
    return float(np.clip(np.rad2deg(delta) / max_steer_deg, -1.0, 1.0))


def gap_and_offset(ego_xy, ego_yaw_deg, ego_half_length, obj_xy, obj_half_length) -> tuple[float, float]:
    """Longitudinal gap (front bumper to the object's near edge) and lateral offset of an object."""
    yaw = np.deg2rad(ego_yaw_deg)
    fwd, right = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = np.asarray(obj_xy, float) - np.asarray(ego_xy, float)
    return float(rel @ fwd - ego_half_length - obj_half_length), float(rel @ right)


def run_measures(trace: list[dict], in_path_width: float) -> dict:
    """Safety measures of one run from its per-tick trace (ground truth from the simulator).

    trace rows: t, speed, brake, gap, lateral, hazard_speed (along the ego's heading), collision.
    """
    t = np.array([r["t"] for r in trace])
    speed = np.array([r["speed"] for r in trace])
    brake = np.array([r["brake"] for r in trace])
    gap = np.array([r["gap"] for r in trace])
    lat = np.abs(np.array([r["lateral"] for r in trace]))
    hz = np.array([r["hazard_speed"] for r in trace])
    coll = np.array([r["collision"] for r in trace], bool)
    ahead = (lat < in_path_width) & (gap > -1.0)
    closing = speed - hz
    ttc = np.where(ahead & (closing > 0.1) & (gap > 0), gap / np.maximum(closing, 0.1), np.inf)
    first_coll = int(np.argmax(coll)) if coll.any() else None
    braking = brake > 0.1
    onset = int(np.argmax(braking)) if braking.any() else None
    return {
        "collision": bool(coll.any()),
        "impact_speed": float(speed[first_coll]) if first_coll is not None else 0.0,
        "min_gap": float(gap[ahead].min()) if ahead.any() else float("nan"),
        "min_ttc": float(ttc.min()) if np.isfinite(ttc).any() else float("inf"),
        "brake_onset_gap": float(gap[onset]) if onset is not None else float("nan"),
        "brake_onset_t": float(t[onset]) if onset is not None else float("nan"),
        "brake_seconds": float(braking.sum() * (t[1] - t[0])) if len(t) > 1 else 0.0,
        "min_speed": float(speed.min()),
        "final_speed": float(speed[-1]),
    }
