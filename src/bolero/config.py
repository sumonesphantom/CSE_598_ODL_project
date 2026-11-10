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
# Frozen ImageNet-1k recognizers that share one label space (see classifier.resolve_backbone).
BACKBONES = {
    "resnet50": "microsoft/resnet-50",
    "vit_b16": "google/vit-base-patch16-224",
    "convnext_t": "facebook/convnext-tiny-224",
}
# Spatial-evidence models (see segmentation.py).
SEGMENTERS = {
    "sam": "facebook/sam-vit-base",
    "detr": "facebook/detr-resnet-50",
}
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

IMAGENETTE_VERSIONS = {
    "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
}
IMAGENETTE_URL = IMAGENETTE_VERSIONS["160"]
DEFAULT_IMAGENETTE_VERSION = "160"

# Decision models on OpenRouter's Decisions API (see openrouter.py).
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.14"
DECISION_MODELS = {
    "jev": "typesafe/jev-1.14",
    "clef-flash": "cloudflare/clef-flash",
    "pplx-decider-flash": "perplexity/pplx-decider-flash-v1.2",
    "d1": "liquid/d1",
    "joey-1": "baseten/joey-1",
}

# TypeSafe image limits (docs.typesafe.ai/images). The API rejects images
# outside these bounds; it never resizes them.
JEV_IMAGE_MIN_SIDE = 280
JEV_IMAGE_MAX_SIDE = 1568
JEV_IMAGE_MIN_PIXELS = 78_400
JEV_IMAGE_MAX_PIXELS = 2_458_624

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


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding the real environment.

    Values are never printed or logged. Missing files are ignored so the rest of
    the pipeline keeps working without an OpenRouter key.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if key and key not in os.environ:
            os.environ[key] = value


def imagenette_dirname(version: str | None = None) -> str:
    """Directory name for an Imagenette release, e.g. imagenette2-320."""
    version = version or os.environ.get("IMAGENETTE_VERSION", DEFAULT_IMAGENETTE_VERSION)
    return f"imagenette2-{version}"


def find_imagenette_root(version: str | None = None) -> Path | None:
    """Locate Imagenette: $IMAGENETTE_ROOT, then ./data, then the sibling course repo."""
    candidates = []
    if os.environ.get("IMAGENETTE_ROOT"):
        candidates.append(Path(os.environ["IMAGENETTE_ROOT"]))
    candidates += [
        DATA_DIR / imagenette_dirname(version),
        REPO_ROOT.parent / "clip_corruption_calibration" / "data" / imagenette_dirname(version),
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
