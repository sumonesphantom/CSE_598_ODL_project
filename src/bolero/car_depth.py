"""The project's depth step on the car (the "forced depth perception" part of the algorithm).

Three ways depth enters the car pipeline, all estimated on the frame the car has (never CARLA's
true depth):

  * depth cues (bolero.cues): Depth-Anything-V2 depth on the repaired frame, one of the six
    cues written back into it at its tuned strength, then the detector. This is the algorithm's
    "optional depth cue" step, unchanged.
  * SAM structure (bolero.segmentation): the same cues, with a structure map from SAM masks in
    place of depth. The project's version keeps the one central entity of an Imagenette photo; a
    road scene has many objects, so here every object-sized mask is kept (`object_structure`).
  * depth obstacle warning (`DepthObstacle`, new): something standing in the car's lane, from the
    depth map alone, without the detector having to recognize it.

How the obstacle warning works. Depth-Anything gives relative inverse depth d (no scale). On a
flat road, the distance of every image row is known from the camera's height and tilt
(CameraModel.ground_distance), so d is fitted to a / Z_road + b on the lower part of the frame,
robustly (the road dominates there; the worst-fitting 40 % of pixels are dropped and the fit
repeated). With that scale, a pixel whose estimated distance is well short of the road distance
of its row is something standing up out of the road. Such pixels inside the car's corridor and
within range are grouped into blobs; the nearest blob big enough is the obstacle, and its distance
is the road distance of its lowest row (where it meets the road), as in the detector's isolation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .car_perception import PATH_HALF_WIDTH, PATH_RANGE, CameraModel


def object_structure(regions, shape: tuple[int, int], max_area: float = 0.1, sigma: float = 2.0) -> np.ndarray:
    """Union of the object-sized SAM masks (each under `max_area` of the frame) as a soft map in
    [0, 1], usable wherever the cues expect depth. Large masks (road, sky, buildings) are left out."""
    mask = np.zeros(shape, bool)
    for r in regions:
        if 0.0 < r.area <= max_area:
            mask |= r.mask
    if not mask.any():
        return np.zeros(shape, np.float32)
    soft = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (soft / (soft.max() + 1e-8)).astype(np.float32)


class DepthObstacle:
    """Nearest obstacle in the car's corridor from a relative depth map (1 = nearest)."""

    def __init__(self, camera: CameraModel, ratio: float = 0.7, min_area: int = 300,
                 max_range: float = PATH_RANGE, half_width: float = PATH_HALF_WIDTH, fit_range: float = 25.0,
                 contact: float = 0.9, scale: float = 1.0, opening: int = 1):
        self.camera, self.ratio, self.min_area, self.contact = camera, ratio, min_area, contact
        self.scale, self.opening = scale, opening    # scale: distance correction, chosen on training towns
        self.max_range, self.half_width, self.fit_range = max_range, half_width, fit_range
        h, w = camera.height, camera.width
        v = np.arange(h, dtype=np.float64)
        self.z_road = camera.ground_distance(v)                       # per row; inf above the horizon
        self.u = np.arange(w, dtype=np.float64)
        self.below = np.isfinite(self.z_road)                         # rows below the horizon
        z = np.where(self.below, self.z_road, 1e9)[:, None]
        lat = np.abs(camera.lateral(self.u[None, :], z))
        self.fit_rows = np.flatnonzero(self.below & (self.z_road < fit_range))
        self.fit_cols = (lat < 6.0)                                   # the road and its edges

    def fit(self, d: np.ndarray) -> tuple[float, float] | None:
        """d ≈ a / Z_road + b on the lower frame; None if the fit is unusable (a <= 0)."""
        rows = self.fit_rows
        sel = self.fit_cols[rows]
        dv = d[rows][sel]
        x = (1.0 / self.z_road[rows])[:, None].repeat(d.shape[1], 1)[sel]
        keep = np.ones(dv.size, bool)
        a = b = None
        for _ in range(3):
            A = np.stack([x[keep], np.ones(keep.sum())], 1)
            (a, b), *_ = np.linalg.lstsq(A, dv[keep], rcond=None)
            res = np.abs(dv - (a * x + b))
            keep = res <= np.quantile(res, 0.6)
        return (float(a), float(b)) if a is not None and a > 0 else None

    def __call__(self, depth: np.ndarray) -> dict:
        h, w = self.camera.height, self.camera.width
        d = depth if depth.shape == (h, w) else np.asarray(
            ndimage.zoom(depth, (h / depth.shape[0], w / depth.shape[1]), order=1), np.float32)
        ab = self.fit(d)
        out = {"distance": None, "lateral": None, "fit": ab, "pixels": 0}
        if ab is None:
            return out
        a, b = ab
        inv = (d - b) / a                                             # estimated 1 / Z
        z_est = np.where(inv > 1e-4, 1.0 / np.maximum(inv, 1e-4), np.inf)
        # Standing pixels: below the horizon, well short of the road distance of their row, and
        # (at their estimated distance) inside the car's corridor and range.
        lat = np.abs(self.camera.lateral(self.u[None, :], np.minimum(z_est, 1e6)))
        standing = (self.below[:, None] & (z_est < self.ratio * self.z_road[:, None])
                    & (z_est < self.max_range) & (lat < self.half_width))
        standing = ndimage.binary_opening(standing, iterations=self.opening) if self.opening else standing
        labels, n = ndimage.label(standing)
        best = None
        for k in range(1, n + 1):
            ys, xs = np.nonzero(labels == k)
            if ys.size < self.min_area:
                continue
            # Follow the object down to where it meets the road: below the blob, rows whose pixels
            # (in the blob's columns) still sit clearly in front of the road surface. The depth
            # model compresses distant objects, so the blob itself often stops above the contact.
            c0, c1 = int(xs.min()), int(xs.max()) + 1
            bottom = int(ys.max())
            while bottom + 1 < h and (z_est[bottom + 1, c0:c1] < self.contact * self.z_road[bottom + 1]).mean() > 0.5:
                bottom += 1
            z = self.scale * float(self.z_road[min(bottom + 1, h - 1)])
            lat = float(self.camera.lateral(np.median(xs), z))
            if z < self.max_range and abs(lat) < self.half_width and (best is None or z < best[0]):
                best = (z, lat, ys.size)
        if best is not None:
            out.update({"distance": best[0], "lateral": best[1], "pixels": best[2]})
        return out
