"""Project-wide constants: paths, class mapping, model names, device selection."""

from __future__ import annotations

import os
from pathlib import Path

import torch

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = REPO_ROOT / "results"

CLASSIFIER_MODEL = "microsoft/resnet-50"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# Imagenette wnid -> class name.
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# Imagenette wnid -> ImageNet-1k index. Verified against the model's id2label in
# notebooks/02_bolero_audit.ipynb §1 and in tests/test_mapping.py.
IMAGENETTE_INDEX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def find_imagenette_root() -> Path | None:
    """Locate imagenette2-160: $IMAGENETTE_ROOT, then ./data, then the sibling course repo."""
    candidates = []
    if os.environ.get("IMAGENETTE_ROOT"):
        candidates.append(Path(os.environ["IMAGENETTE_ROOT"]))
    candidates += [
        DATA_DIR / "imagenette2-160",
        REPO_ROOT.parent / "clip_corruption_calibration" / "data" / "imagenette2-160",
    ]
    for candidate in candidates:
        if (candidate / "val").is_dir():
            return candidate.resolve()
    return None


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
