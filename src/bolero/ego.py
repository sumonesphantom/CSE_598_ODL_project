"""Car-side pieces of the driving study: ground truth from CARLA's car camera, and weather on it.

Coordinates follow the camera: z forward (CARLA's depth camera gives this planar depth),
x to the right. The camera is a pinhole with horizontal field of view `fov`.

Weather is the project's own physics effects (perturbations.WEATHER), built from CARLA's exact
depth in the same normalized inverse-depth convention as Depth-Anything's output (1 = nearest),
so the effects behave on car frames as they do everywhere else in the project.
"""

from __future__ import annotations

import numpy as np

from .depth import normalize
from .perturbations import WEATHER, rng_for

TAG_CLASS = {12: "person", 13: "person", 14: "car", 15: "truck", 16: "bus", 18: "motorcycle", 19: "bicycle"}

# An object is "in the car's path" when it is ahead, within this lateral offset (half a lane plus
# half the car's width) and this distance. Used for ground truth and for the isolation stage alike.
PATH_HALF_WIDTH = 2.0   # m
PATH_RANGE = 40.0       # m


def focal_px(width: int, fov: float) -> float:
    return width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))


def decode_depth(bgra: np.ndarray) -> np.ndarray:
    """CARLA depth camera (H, W, 4) BGRA -> planar depth in metres."""
    d = bgra.astype(np.float64)
    return ((d[..., 2] + d[..., 1] * 256 + d[..., 0] * 65536) / (256 ** 3 - 1) * 1000.0).astype(np.float32)


def objects_from_instances(inst: np.ndarray, depth_m: np.ndarray, fov: float, min_box: int = 12,
                           min_pixels: int = 150, exclude_ids=()) -> list[dict]:
    """Visible objects from CARLA's instance camera (BGRA; R = semantic tag, G/B = instance id).

    Each object: box (x1 y1 x2 y2 px), label, id, distance (median depth of its pixels, m),
    lateral (median signed offset to the right, m), pixels, in_path.
    """
    h, w = inst.shape[:2]
    f = focal_px(w, fov)
    tag = inst[..., 2].astype(np.int32)
    ident = inst[..., 1].astype(np.int32) | (inst[..., 0].astype(np.int32) << 8)
    excluded = {int(i) & 0xFFFF for i in exclude_ids}
    objects = []
    for t, name in TAG_CLASS.items():
        mask = tag == t
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        ids = ident[ys, xs]
        for i in np.unique(ids):
            if int(i) in excluded:
                continue
            sel = ids == i
            if sel.sum() < min_pixels:
                continue
            x1, x2, y1, y2 = xs[sel].min(), xs[sel].max() + 1, ys[sel].min(), ys[sel].max() + 1
            if (x2 - x1) < min_box or (y2 - y1) < min_box:
                continue
            z = depth_m[ys[sel], xs[sel]]
            lateral = (xs[sel] + 0.5 - w / 2.0) * z / f
            dist, lat = float(np.median(z)), float(np.median(lateral))
            objects.append({"box": [int(x1), int(y1), int(x2), int(y2)], "label": name, "id": int(i),
                            "distance": round(dist, 3), "lateral": round(lat, 3), "pixels": int(sel.sum()),
                            "in_path": bool(dist < PATH_RANGE and abs(lat) < PATH_HALF_WIDTH)})
    return objects


def inverse_depth(depth_m: np.ndarray) -> np.ndarray:
    """Metres -> the physics effects' convention (normalized inverse depth, 1 = nearest)."""
    return normalize(1.0 / np.clip(depth_m, 0.5, None))


def resize_to(depth_m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    if depth_m.shape[:2] == tuple(shape):
        return depth_m
    return cv2.resize(depth_m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def apply_weather(image: np.ndarray, depth_m: np.ndarray, effect: str, level: int, frame_id: str,
                  seed: int = 42) -> np.ndarray:
    """One physics weather effect at a level on a float [0, 1] RGB car frame. `effect` "clean" is identity.
    Deterministic per frame, effect and level (rain streaks and sand grains still change frame to frame)."""
    if effect == "clean" or level == 0:
        return image
    d = inverse_depth(resize_to(depth_m, image.shape[:2]))
    return WEATHER[effect](image, level, d, rng_for(frame_id, effect, level, seed))
