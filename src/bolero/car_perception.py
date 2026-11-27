"""The algorithm on the car: route on the weather, repair the frame, detect, isolate what matters.

  frame -> ConditionRouter (clean / rain / dust, smoothed over time)
        -> repairer chosen for that condition (learned_restoration)
        -> CarDetector (DETR, the project's frozen detector, or Mask R-CNN for instance masks)
        -> isolate(): distance and lateral offset of every object from the ground plane,
           which objects are in the car's path, and the most critical one (nearest in path)

Distances use only the camera's height and pitch (flat-road assumption), not ground-truth depth,
so the whole pipeline uses information a real car has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .config import get_device
from .detection import CLEAN, PerturbationDetector, extract_features
from .detection_eval import Detections
from .ego import PATH_HALF_WIDTH, PATH_RANGE, focal_px
from .weather_data import CLASSES

ROUTER_SIDE = 320       # short side the condition detector sees (as in train_weather_detector.py)
FAMILIES = (CLEAN, "rain", "dust")


def router_input(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    s = ROUTER_SIDE / min(h, w)
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def router_features(image: np.ndarray) -> np.ndarray:
    return extract_features(router_input(image))


# --- routing -----------------------------------------------------------------------------------
@dataclass
class ConditionRouter:
    """Per-frame condition detector plus smoothing over time.

    Probabilities are averaged with an exponential moving average (weight `alpha` on the new frame).
    The route switches to rain or dust only when that class's smoothed probability reaches `enter`,
    and back to clean only when it falls below `exit` (hysteresis), so the route does not flicker.
    With alpha=1, enter=exit=threshold it is the detector's plain per-frame rule.
    """

    detector: PerturbationDetector
    alpha: float = 0.4
    enter: float = 0.6
    exit: float = 0.45
    state: str = CLEAN
    smoothed: np.ndarray | None = None
    last_seconds: float = 0.0

    def __post_init__(self):
        if self.detector is not None:
            self.detector.model.n_jobs = 1   # one frame at a time: parallel trees only add overhead

    def reset(self):
        self.state, self.smoothed = CLEAN, None

    def proba(self, features: np.ndarray) -> np.ndarray:
        """(n, 3) probabilities in FAMILIES order."""
        p = self.detector.predict_proba(np.atleast_2d(features))
        cls = list(self.detector.classes)
        return np.stack([p[:, cls.index(c)] if c in cls else np.zeros(len(p)) for c in FAMILIES], axis=1)

    def step_proba(self, p: np.ndarray) -> str:
        self.smoothed = p if self.smoothed is None else self.alpha * p + (1 - self.alpha) * self.smoothed
        weather = {c: self.smoothed[i] for i, c in enumerate(FAMILIES) if c != CLEAN}
        best = max(weather, key=weather.get)
        if self.state == CLEAN:
            if weather[best] >= self.enter:
                self.state = best
        elif weather[self.state] < self.exit:
            self.state = best if weather[best] >= self.enter else CLEAN
        elif best != self.state and weather[best] >= self.enter and weather[best] > weather[self.state]:
            self.state = best
        return self.state

    def __call__(self, image: np.ndarray) -> str:
        start = time.perf_counter()
        state = self.step_proba(self.proba(router_features(image))[0])
        self.last_seconds = time.perf_counter() - start
        return state


def smooth_sequence(probs: np.ndarray, alpha: float, enter: float, exit: float) -> list[str]:
    """Route for every frame of a sequence of (n, 3) probabilities."""
    r = ConditionRouter(detector=None, alpha=alpha, enter=enter, exit=exit)
    return [r.step_proba(p) for p in probs]


# --- detection ---------------------------------------------------------------------------------
@dataclass
class CarDetections(Detections):
    masks: list | None = field(default=None)   # per detection (H, W) bool, Mask R-CNN only


class CarDetector:
    """`detr`: the project's frozen facebook/detr-resnet-50. `maskrcnn`: torchvision Mask R-CNN
    ResNet-50 FPN v2 (COCO), which also gives instance masks. Both kept to the six classes."""

    def __init__(self, kind: str = "detr", device=None, min_score: float = 0.05, fp16: bool = False):
        self.kind, self.min_score, self.fp16 = kind, min_score, fp16
        self.device = get_device(device)
        if kind == "detr":
            from .detection_eval import FrozenDetector

            self.model = FrozenDetector("detr", device=device, min_score=min_score)
        elif kind == "maskrcnn":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

            weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            self.model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=min_score).to(self.device).eval()
            self.categories = weights.meta["categories"]
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, want_masks: bool = False) -> CarDetections:
        start = time.perf_counter()
        if self.kind == "detr":
            from PIL import Image

            d = self.model(Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)))
            return CarDetections(d.boxes, d.scores, d.labels, time.perf_counter() - start)
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(self.device, torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16 and self.device.type == "cuda"):
            out = self.model([x])[0]
        names = [self.categories[i] for i in out["labels"].tolist()]
        keep = [i for i, n in enumerate(names) if n in CLASSES]
        boxes = out["boxes"][keep].cpu().numpy().astype(np.float32).reshape(-1, 4)
        scores = out["scores"][keep].cpu().numpy().astype(np.float32)
        masks = list((out["masks"][keep, 0] > 0.5).cpu().numpy()) if want_masks else None
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return CarDetections(boxes, scores, [names[i] for i in keep], time.perf_counter() - start, masks)


# --- isolation ---------------------------------------------------------------------------------
@dataclass
class CameraModel:
    width: int = 1280
    height: int = 720
    fov: float = 90.0
    camera_height: float = 1.4   # m above the road
    pitch: float = -2.0          # degrees, negative = looking down

    @property
    def f(self) -> float:
        return focal_px(self.width, self.fov)

    def ground_distance(self, v_bottom: np.ndarray) -> np.ndarray:
        """Planar depth (along the optical axis) of a road point imaged at row v (flat road)."""
        theta = np.deg2rad(-self.pitch)
        below = np.arctan((np.asarray(v_bottom, np.float64) - self.height / 2.0) / self.f) + theta
        horizontal = np.where(below > 1e-3, self.camera_height / np.tan(np.maximum(below, 1e-3)), np.inf)
        return horizontal * np.cos(theta) + self.camera_height * np.sin(theta)

    def lateral(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (np.asarray(u, np.float64) - self.width / 2.0) * z / self.f


def isolate(dets: Detections, camera: CameraModel, score: float = 0.5) -> list[dict]:
    """Objects the car should know about: confident detections with distance, lateral offset and
    an in-path flag; sorted nearest first, the first in-path one marked critical."""
    keep = np.flatnonzero(dets.scores >= score)
    if len(keep) == 0:
        return []
    boxes = dets.boxes[keep]
    z = np.minimum(camera.ground_distance(boxes[:, 3]), 200.0)
    lat = camera.lateral((boxes[:, 0] + boxes[:, 2]) / 2.0, z)
    objects = []
    for j, i in enumerate(keep):
        objects.append({"box": boxes[j].tolist(), "label": dets.labels[i], "score": float(dets.scores[i]),
                        "distance": float(z[j]), "lateral": float(lat[j]),
                        "in_path": bool(z[j] < PATH_RANGE and abs(lat[j]) < PATH_HALF_WIDTH),
                        "mask_index": int(i), "critical": False})
    objects.sort(key=lambda o: o["distance"])
    for o in objects:
        if o["in_path"]:
            o["critical"] = True
            break
    return objects


# --- the whole thing ---------------------------------------------------------------------------
class CarPerception:
    """Router -> repairer for the routed condition -> detector -> isolation, with per-stage timings.

    With `async_route`, the router runs in a background thread on the newest frame while the
    repair and detection run; each frame uses the latest finished route. Weather changes over
    seconds, so a route one or two frames old costs nothing, and no frame waits for the router."""

    def __init__(self, router: ConditionRouter, repairers: dict, detector: CarDetector,
                 camera: CameraModel = CameraModel(), score: float = 0.5, want_masks: bool = False,
                 async_route: bool = False):
        self.router, self.repairers, self.detector = router, repairers, detector
        self.camera, self.score, self.want_masks = camera, score, want_masks
        self.async_route = async_route
        self.pool, self.pending = None, None
        if async_route:
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(max_workers=1)

    def _route(self, image: np.ndarray) -> str:
        if not self.async_route:
            return self.router(image)
        if self.pending is None or self.pending.done():
            if self.pending is not None:
                self.pending.result()   # surface errors
            self.pending = self.pool.submit(self.router, image)
        return self.router.state

    def __call__(self, image: np.ndarray) -> dict:
        t0 = time.perf_counter()
        route = self._route(image)
        t1 = time.perf_counter()
        repaired = self.repairers.get(route, lambda x: x)(image)
        t2 = time.perf_counter()
        dets = self.detector(repaired, want_masks=self.want_masks)
        t3 = time.perf_counter()
        objects = isolate(dets, self.camera, self.score)
        t4 = time.perf_counter()
        return {"route": route, "repaired": repaired, "detections": dets, "objects": objects,
                "ms": {"route": 1e3 * (t1 - t0), "repair": 1e3 * (t2 - t1), "detect": 1e3 * (t3 - t2),
                       "isolate": 1e3 * (t4 - t3), "total": 1e3 * (t4 - t0)}}
