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


WEATHER_MODELS = ("v1", "v2")


def apply_weather(image: np.ndarray, depth_m: np.ndarray, effect: str, level: int, frame_id: str,
                  seed: int = 42, model: str = "v1", t: float | None = None) -> np.ndarray:
    """One physics weather effect at a level on a float [0, 1] RGB car frame.

    model "v1": the project's effects (perturbations.WEATHER); "clean" is the identity.
    model "v2": the harder, more realistic version (weather_v2): metric-depth dust with uneven
    plumes, a sky-dependent airlight and forward-scatter blur, and camera noise plus JPEG on every
    frame, clean ones included. `t` (seconds) makes v2's plumes drift smoothly in a live drive;
    without it the plume position comes from the frame id.
    Deterministic per frame, effect and level (rain streaks and sand grains change frame to frame)."""
    if model == "v2":
        return weather_v2(image, depth_m, effect, level, frame_id, seed, t)
    if model != "v1":
        raise ValueError(f"unknown weather model {model!r}; choose from {WEATHER_MODELS}")
    if effect == "clean" or level == 0:
        return image
    d = inverse_depth(resize_to(depth_m, image.shape[:2]))
    return WEATHER[effect](image, level, d, rng_for(frame_id, effect, level, seed))


# --- weather v2 ----------------------------------------------------------------------------------
# v1 dust is exactly the dark-channel dehazer's forward model (I = J t + A (1 - t), one airlight,
# transmission from depth alone), so the dehazer can undo it almost perfectly, which real dust does
# not allow (Phase 2's real-photo check). v2 breaks each of those assumptions the way real dust does:
#   * extinction from metric depth and a meteorological visibility per level;
#   * uneven density: a drifting low-frequency plume field scales the extinction 0.4x to 1.6x;
#   * airlight that is brighter and paler toward the sky, plus a glow on one side (sun direction);
#   * forward scattering: light from distant objects spreads, so they blur as well as fade;
#   * the camera: shot and read noise, then JPEG compression, on every frame, clean ones too.
# Rain keeps v1's streaks / mist / lens drops (they are not what the dehazer models) plus the camera.
DUST_VISIBILITY_M = {1: 200.0, 2: 90.0, 3: 45.0}
SKY_DISTANCE_M = 1000.0


def _plume_window(shape: tuple[int, int], effect: str, level: int, seed: int, offset: float) -> np.ndarray:
    """(H, W) field in [0, 1]: a window of a wide fixed plume texture, shifted by `offset` (0..1)."""
    h, w = shape
    base = _plume_texture(h, w, effect, level, seed)
    start = round((offset % 1.0) * 2 * w)
    return base[:, start:start + w]


_PLUMES: dict = {}


def _plume_texture(h: int, w: int, effect: str, level: int, seed: int) -> np.ndarray:
    import cv2

    key = (h, w, effect, level, seed)
    if key not in _PLUMES:
        rng = rng_for("plume", effect, level, seed)
        field = np.zeros((h, 3 * w), np.float32)
        for cells, weight in ((3, 0.65), (9, 0.35)):
            grid = rng.random((cells + 1, 3 * cells + 1)).astype(np.float32)
            grid[:, -1] = grid[:, 0]                       # wraps, so the drifting window has no seam
            field += weight * cv2.resize(grid, (3 * w, h), interpolation=cv2.INTER_CUBIC)
        field -= field.min()
        _PLUMES[key] = field / max(field.max(), 1e-6)
    return _PLUMES[key]


def _airlight_map(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    from .perturbations import DUST_AIRLIGHT

    yy = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]          # 1 at the top
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    sun = float(rng.uniform(0.0, 1.0))
    glow = 0.10 * np.exp(-((xx - sun) ** 2) / 0.08) * yy
    pale = np.array([0.88, 0.80, 0.66], np.float32)                          # washed-out sky dust
    a = DUST_AIRLIGHT * (0.80 + 0.20 * yy) * (1 - 0.5 * yy) + pale * 0.5 * yy
    return np.clip(a + glow, 0.0, 1.0)


def _camera(image: np.ndarray, level: int, rng: np.random.Generator) -> np.ndarray:
    """Shot + read noise (stronger in bad light), then JPEG at quality 75."""
    import cv2

    sigma = 0.006 + 0.004 * level
    noisy = image + rng.normal(0.0, 1.0, image.shape).astype(np.float32) * (sigma * np.sqrt(np.clip(image, 0.02, 1.0)) + 0.003)
    u8 = (np.clip(noisy, 0, 1) * 255 + 0.5).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def dust_v2(image: np.ndarray, depth_m: np.ndarray, level: int, rng: np.random.Generator,
            plume: np.ndarray, storm: bool) -> np.ndarray:
    import cv2

    from .perturbations import DUST_AIRLIGHT, DUST_BETA_RGB, _blur, _line_kernel

    h, w = image.shape[:2]
    dist = np.minimum(depth_m, SKY_DISTANCE_M).astype(np.float32)
    vis = DUST_VISIBILITY_M[level] * (0.6 if storm else 1.0)
    beta = 3.912 / vis * (0.4 + 1.2 * plume)
    t = np.exp(-(beta * dist)[..., None] * DUST_BETA_RGB)
    fs = np.clip(1.0 - t.mean(axis=2, keepdims=True), 0.0, 1.0) * 0.7         # forward scatter share
    spread = _blur(image, 1.5 + 2.0 * level)
    direct = image * (1.0 - fs) + spread * fs
    out = direct * t + _airlight_map(h, w, rng) * (1.0 - t)
    if storm:
        grains = (rng.random((h, w)) < 0.02 + 0.02 * level).astype(np.float32)
        length = max(3, round(min(h, w) * (0.02 + 0.015 * level)))
        grains = cv2.filter2D(grains, -1, _line_kernel(length, float(rng.uniform(70.0, 110.0))),
                              borderType=cv2.BORDER_REFLECT)
        grains = np.clip(grains / (np.quantile(grains[grains > 0], 0.5) + 1e-6) * (0.15 + 0.1 * level), 0.0, 0.8)
        out = out * (1.0 - grains[..., None]) + DUST_AIRLIGHT * 0.9 * grains[..., None]
    return out


def weather_v2(image: np.ndarray, depth_m: np.ndarray, effect: str, level: int, frame_id: str,
               seed: int = 42, t: float | None = None) -> np.ndarray:
    import zlib

    rng = rng_for(frame_id, f"v2_{effect}", level, seed)
    if effect == "clean" or level == 0:
        out = image
    elif effect.startswith("dust"):
        depth = resize_to(depth_m, image.shape[:2])
        offset = (t / 120.0) if t is not None else zlib.crc32(frame_id.encode()) / 2**32   # wind drift
        plume = _plume_window(image.shape[:2], effect, level, seed, offset)
        out = dust_v2(image, depth, level, rng, plume, storm=effect == "dust_storm")
    else:
        d = inverse_depth(resize_to(depth_m, image.shape[:2]))
        out = WEATHER[effect](image, level, d, rng)
    return _camera(np.clip(out, 0, 1).astype(np.float32), level, rng)
