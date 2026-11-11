"""Loaders for real adverse-weather data: DAWN, XWOD (detection) and SDSI / VIRADA (condition labels).

Detection samples carry pixel boxes (x1, y1, x2, y2) and class names from the six
traffic / security classes shared by DAWN and XWOD. `condition` uses the weather
taxonomy names (`rain`, `dust`), so DAWN's "Sand" becomes `dust`.

Data layout (all git-ignored under data/):

  data/dawn/Rain/*.jpg + Rain/Rain_PASCAL_VOC/*.xml     DAWN, Mendeley 10.17632/766ygrbt8y.3,
  data/dawn/Sand/*.jpg + Sand/Sand_PASCAL_VOC/*.xml     CC BY-NC 3.0 (attribute, non-commercial)
  data/xwod/data.yaml, images/<split>/, labels/<split>/  XWOD in Ultralytics YOLO layout (Kaggle;
                                                          needs ~/.kaggle/kaggle.json to download)
  data/sdsi/<label>/*.jpg, data/virada/<label>/*.jpg      image-level condition labels, one folder
                                                          per label (e.g. storm/ and clear/)

DAWN's YOLO .txt files use numeric ids with no names file, so the VOC XML (explicit
class names) is the source of truth here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR

CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
# COCO category ids as used by facebook/detr-resnet-50's id2label.
COCO_ID = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8}

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


def _yolo_names(root: Path) -> list[str]:
    import yaml

    data = yaml.safe_load((root / "data.yaml").read_text())
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def _xwod_condition(name: str) -> str | None:
    """Map an XWOD condition (folder or filename token) onto the weather taxonomy."""
    n = name.lower()
    if "rain" in n:
        return "rain"
    if any(t in n for t in ("sand", "dust", "haze")):
        return "dust"
    return None


def load_yolo_dataset(root: Path, split: str, dataset: str, condition_of=None,
                      keep: tuple[str, ...] = CLASSES) -> list[DetectionSample]:
    """Ultralytics layout: data.yaml (names), images/<split>/..., labels/<split>/... (normalized cx cy w h).

    `condition_of(path) -> str | None` assigns the weather condition (None drops the image);
    by default it looks for rain / sand / dust / haze in the relative path.
    """
    from PIL import Image

    root = Path(root)
    names = [n.lower() for n in _yolo_names(root)]
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    condition_of = condition_of or (lambda p: _xwod_condition(str(p.relative_to(image_dir))))
    samples = []
    for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        condition = condition_of(image)
        if condition is None:
            continue
        with Image.open(image) as im:
            w, h = im.size
        label_file = (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
        rows = np.loadtxt(label_file, ndmin=2) if label_file.exists() and label_file.stat().st_size else np.zeros((0, 5))
        boxes, labels = [], []
        for cls, cx, cy, bw, bh in rows:
            name = names[int(cls)]
            if name in keep:
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                labels.append(name)
        samples.append(DetectionSample(image.stem, image, dataset, condition,
                                       np.asarray(boxes, np.float32).reshape(-1, 4), labels, w, h))
    return samples


def load_xwod(root: Path | None = None, split: str = "test") -> list[DetectionSample]:
    """XWOD rain and haze/sand/dust images (other conditions dropped)."""
    root = Path(root or DATA_DIR / "xwod")
    if not (root / "data.yaml").exists():
        raise FileNotFoundError(
            f"{root}/data.yaml not found. Download XWOD from Kaggle (kuantinglai/exwod) into {root}; "
            "this loader expects the Ultralytics YOLO layout and may need adjusting to the release.")
    return load_yolo_dataset(root, split, "xwod")


def load_condition_folders(root: Path, dataset: str) -> list[ConditionImage]:
    """Image-level labels, one sub-folder per label (SDSI: storm / clear; VIRADA: rain / no_rain)."""
    root = Path(root)
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        out += [ConditionImage(p, dataset, folder.name.lower())
                for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return out
