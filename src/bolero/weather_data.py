"""Loaders for real adverse-weather data: DAWN, XWOD (detection) and SDSI / VIRADA (condition labels).

Detection samples carry pixel boxes (x1, y1, x2, y2) and class names from the six
traffic / security classes shared by DAWN and XWOD. `condition` uses the weather
taxonomy names (`rain`, `dust`), so DAWN's "Sand" becomes `dust`.

Data layout (all git-ignored under data/):

  data/dawn/Rain/*.jpg + Rain/Rain_PASCAL_VOC/*.xml     DAWN, Mendeley 10.17632/766ygrbt8y.3,
  data/dawn/Sand/*.jpg + Sand/Sand_PASCAL_VOC/*.xml     CC BY-NC 3.0 (attribute, non-commercial)
  data/xwod/XWOD_dataset.v1.1/XWOD_dataset.v1.1/        XWOD v1.1 from Kaggle (kuantinglai/exwod,
      data.v1.1.yaml, <split>.v1.1/{images,labels}/      CC BY-NC-SA 4.0, needs a Kaggle API key).
                                                          Splits train / valid / test; conditions come
                                                          from the filename (heavy_rain_*, dust_*; the
                                                          other five are dropped). Class "bike" = bicycle
  data/sdsi/<label>/*.jpg, data/virada/<label>/*.jpg      image-level condition labels, one folder
                                                          per label (e.g. storm/ and clear/)

DAWN's YOLO .txt files use numeric ids with no names file, so the VOC XML (explicit
class names) is the source of truth here.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR

CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
# COCO category ids as used by facebook/detr-resnet-50's id2label.
COCO_ID = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8}
# Dataset class names that mean one of CLASSES (XWOD calls bicycles "bike").
CLASS_ALIASES = {"bike": "bicycle", "motorbike": "motorcycle"}

DAWN_CONDITIONS = {"rain": "Rain", "dust": "Sand", "fog": "Fog", "snow": "Snow"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class DetectionSample:
    image_id: str
    path: Path
    dataset: str
    condition: str
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # (N, 4) x1 y1 x2 y2 px
    labels: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None


@dataclass
class ConditionImage:
    path: Path
    dataset: str
    label: str


def parse_voc(xml_path: Path, keep: tuple[str, ...] = CLASSES) -> tuple[np.ndarray, list[str], int, int]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes, labels = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in keep:
            continue
        bb = obj.find("bndbox")
        boxes.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(name)
    return np.asarray(boxes, np.float32).reshape(-1, 4), labels, width, height


def load_dawn(root: Path | None = None, conditions: tuple[str, ...] = ("rain", "dust")) -> list[DetectionSample]:
    """DAWN images with VOC boxes; images without an annotation file are skipped."""
    root = Path(root or DATA_DIR / "dawn")
    samples = []
    for condition in conditions:
        folder = root / DAWN_CONDITIONS[condition]
        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found; see bolero/weather_data.py for the layout")
        voc = folder / f"{folder.name}_PASCAL_VOC"
        for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            xml = voc / f"{image.stem}.xml"
            if not xml.exists():
                continue
            boxes, labels, w, h = parse_voc(xml)
            samples.append(DetectionSample(image.stem, image, "dawn", condition, boxes, labels, w, h))
    return samples


def open_rgb(path: Path):
    """Open an image as RGB with its EXIF orientation applied, the view the box labels are drawn on.

    Some XWOD images (35, e.g. dust_test_00034) carry an EXIF orientation tag, and their YOLO
    boxes match the oriented view. transformers' pipeline applies the tag to a PIL image from
    disk but not to an image rebuilt from an array, so without this the same photo could reach
    the detector mirrored in one method and unmirrored in another.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _yolo_yaml(root: Path) -> Path | None:
    """data.yaml, or a versioned name such as XWOD v1.1's data.v1.1.yaml."""
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    return next(iter(sorted(root.glob("data*.yaml"))), None)


def _yolo_names(root: Path) -> list[str]:
    import yaml

    data = yaml.safe_load(_yolo_yaml(root).read_text())
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    return [CLASS_ALIASES.get(str(n).lower(), str(n).lower()) for n in names]


def _yolo_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """(images, labels) for a split. Accepts images/<split>/ (Ultralytics), <split>/images/
    (Roboflow export, as in XWOD) and versioned <split>.v1.1/images/; "val" and "valid" are aliases."""
    names = ("val", "valid") if split in ("val", "valid") else (split,)
    candidates = []
    for s in names:
        candidates += [(root / "images" / s, root / "labels" / s), (root / s / "images", root / s / "labels")]
        candidates += [(d / "images", d / "labels") for d in sorted(root.glob(f"{s}.*")) if d.is_dir()]
    for images, labels in candidates:
        if images.is_dir():
            return images, labels
    raise FileNotFoundError(f"No images folder for split {split!r} under {root}")


def _xwod_condition(name: str) -> str | None:
    """Map an XWOD condition (folder or filename token) onto the weather taxonomy."""
    # Whole tokens only: a substring test matches the "rain" in "train" and tags every
    # training image (flooding_train_00001, snow_train_00001, ...) as rain.
    tokens = set(re.split(r"[^a-z]+", name.lower()))
    if "rain" in tokens:
        return "rain"
    if tokens & {"sand", "dust", "haze"}:
        return "dust"
    return None


def load_yolo_dataset(root: Path, split: str, dataset: str, condition_of=None,
                      keep: tuple[str, ...] = CLASSES) -> list[DetectionSample]:
    """Ultralytics layout: data.yaml (names), images/<split>/..., labels/<split>/... (normalized cx cy w h).

    `condition_of(path) -> str | None` assigns the weather condition (None drops the image);
    by default it looks for rain / sand / dust / haze in the relative path.
    """
    from PIL import Image, ImageOps

    root = Path(root)
    names = _yolo_names(root)
    image_dir, label_dir = _yolo_split_dirs(root, split)
    condition_of = condition_of or (lambda p: _xwod_condition(str(p.relative_to(image_dir))))
    samples = []
    for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        condition = condition_of(image)
        if condition is None:
            continue
        with Image.open(image) as im:
            w, h = ImageOps.exif_transpose(im).size if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else im.size
        label_file = (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
        rows = np.loadtxt(label_file, ndmin=2) if label_file.exists() and label_file.stat().st_size else np.zeros((0, 5))
        boxes, labels = [], []
        for cls, cx, cy, bw, bh in rows:
            name = names[int(cls)]
            if name in keep:
                # Clipped: some XWOD boxes run a few pixels past the image edge.
                boxes.append([max((cx - bw / 2) * w, 0.0), max((cy - bh / 2) * h, 0.0),
                              min((cx + bw / 2) * w, w), min((cy + bh / 2) * h, h)])
                labels.append(name)
        samples.append(DetectionSample(image.stem, image, dataset, condition,
                                       np.asarray(boxes, np.float32).reshape(-1, 4), labels, w, h))
    return samples


def load_xwod(root: Path | None = None, split: str = "test") -> list[DetectionSample]:
    """XWOD rain and haze/sand/dust images (other conditions dropped)."""
    root = Path(root or DATA_DIR / "xwod")
    if _yolo_yaml(root) is None:
        # The Kaggle zip nests the release two folders down (XWOD_dataset.v1.1/XWOD_dataset.v1.1/).
        found = sorted(root.rglob("data*.yaml"), key=lambda p: (len(p.parts), str(p))) if root.exists() else []
        if not found:
            raise FileNotFoundError(
                f"No data*.yaml under {root}. Download XWOD with "
                f"`kaggle datasets download kuantinglai/exwod -p {root} --unzip`.")
        root = found[0].parent
    return load_yolo_dataset(root, split, "xwod")


def load_condition_folders(root: Path, dataset: str) -> list[ConditionImage]:
    """Image-level labels, one sub-folder per label (SDSI: storm / clear; VIRADA: rain / no_rain)."""
    root = Path(root)
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        out += [ConditionImage(p, dataset, folder.name.lower())
                for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return out


VEHICLES = ("bicycle", "car", "motorcycle", "bus", "truck")


def load_coco(root: Path | None = None, n: int = 300, seed: int = 42, keep: tuple[str, ...] = CLASSES,
              require_any: tuple[str, ...] = VEHICLES) -> list[DetectionSample]:
    """Clean COCO val2017 street-type images: the undegraded source for synthetic rain / dust.

    Keeps images with at least one vehicle box (`require_any`, so the scenes are traffic
    scenes like XWOD / DAWN rather than COCO's many indoor portraits) and no crowd region
    of the six classes (a crowd box cannot be scored as one object), then samples `n`.
    Layout: data/coco/val2017/*.jpg + data/coco/annotations/instances_val2017.json.
    """
    import json

    root = Path(root or DATA_DIR / "coco")
    ann_path = root / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} not found. Download val2017.zip and the 2017 annotations "
                                "from images.cocodataset.org into data/coco/.")
    ann = json.loads(ann_path.read_text())
    names = {c["id"]: c["name"] for c in ann["categories"]}
    boxes, crowd = {}, set()
    for a in ann["annotations"]:
        name = names[a["category_id"]]
        if name not in keep:
            continue
        if a["iscrowd"]:
            crowd.add(a["image_id"])
            continue
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0:
            boxes.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], name))
    images = {im["id"]: im for im in ann["images"]}
    eligible = sorted(i for i, b in boxes.items()
                      if i not in crowd and any(c in require_any for _, c in b))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=min(n, len(eligible)), replace=False).tolist())
    samples = []
    for i in chosen:
        im = images[i]
        b = boxes[i]
        samples.append(DetectionSample(f"coco_{i:012d}", root / "val2017" / im["file_name"], "coco", "clean",
                                       np.asarray([x for x, _ in b], np.float32).reshape(-1, 4),
                                       [c for _, c in b], im["width"], im["height"]))
    return samples


def load_carla(root: Path | None = None, condition: str = "clean", level: int = 0,
               towns: tuple[str, ...] | None = None) -> list[DetectionSample]:
    """CARLA traffic-camera frames from scripts/record_carla_weather.py.

    Every scene is rendered clean and under rain / dust levels 1-3 with all actors frozen, so the
    boxes (from the clean frame's instance segmentation) hold for every condition. `image_id` is
    the scene name, shared across conditions, so sets pair up image by image.
    """
    import json

    import pandas as pd

    root = Path(root or DATA_DIR / "carla")
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found; record frames with scripts/record_carla_weather.py")
    m = pd.read_csv(manifest)
    m = m[(m["condition"] == condition) & (m["level"] == level)]
    if towns:
        m = m[m["town"].isin(towns)]
    samples = []
    for row in m.sort_values("scene").itertuples():
        lab = json.loads((root / "labels" / f"{row.scene}.json").read_text())
        samples.append(DetectionSample(row.scene, Path(row.path), "carla", condition,
                                       np.asarray(lab["boxes"], np.float32).reshape(-1, 4), list(lab["labels"]),
                                       lab["width"], lab["height"]))
    return samples


def carla_depth(root: Path, scene: str) -> np.ndarray:
    """CARLA ground-truth depth in metres for a scene (saved as 16-bit centimetres)."""
    import cv2

    return cv2.imread(str(Path(root) / "depth" / f"{scene}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
