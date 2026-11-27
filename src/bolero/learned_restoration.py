"""Weather repair for the car's camera: pretrained deraining / dehazing models and a classical dehazer.

  restormer_derain    Restormer (Zamir et al., CVPR 2022), official deraining weights (Rain13K)
  instructir_derain   InstructIR (Conde et al., ECCV 2024), 7-task model, told to remove rain
  instructir_dehaze   the same model, told to remove haze (it was trained on outdoor haze, SOTS)
  dcp_dehaze          dark channel prior (He et al., CVPR 2009) with a guided-filter refined
                      transmission and a per-channel airlight, so yellow-brown dust is handled
  autocontrast        the project's depth-free baseline (baselines.autocontrast), for reference

Weights come only from the authors' official releases and are checked against pinned SHA-256
hashes. Model code is vendored in bolero/third_party (MIT, see LICENSES.txt).

Every repairer maps a float32 RGB image in [0, 1] (H, W, 3) to the same. `max_side` lets the
learned models run on a downscaled copy for speed; only the predicted correction is upsampled and
added to the full-resolution input, so fine detail is kept.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import CACHE_DIR, get_device

WEIGHTS = {
    "restormer_deraining.pth": ("https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth",
                                "225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681"),
    "im_instructir-7d.pt": ("https://huggingface.co/marcosv/InstructIR/resolve/main/im_instructir-7d.pt",
                            "f28d8f0f66ff57449ebe2be52241dfdd53a3dfab1003d63e65493f96ea152fd0"),
    "lm_instructir-7d.pt": ("https://huggingface.co/marcosv/InstructIR/resolve/main/lm_instructir-7d.pt",
                            "b239e5d5dbc811813a90e709f9647dead0e35a96a294a7d6c5263da549016fe6"),
}
INSTRUCTIR_TEXT_MODEL = "TaylorAI/bge-micro-v2"
PROMPTS = {"derain": "Clear the rain from my picture.", "dehaze": "Get rid of the fog in my image."}


def weight_path(name: str, root: Path = CACHE_DIR / "restoration") -> Path:
    """Local copy of an official weight file; downloaded and hash-checked on first use."""
    url, sha = WEIGHTS[name]
    path = Path(root) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise RuntimeError(f"{path} does not match the pinned hash ({digest}); delete it and retry")
    return path


# --- classical ---------------------------------------------------------------------------------
def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 40, eps: float = 1e-3) -> np.ndarray:
    """He et al. guided filter with a gray guide (float32 inputs)."""
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mean_i * mean_p
    var = _box(guide * guide, r) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def dcp_dehaze(image: np.ndarray, patch: int = 15, omega: float = 0.95, t_min: float = 0.15,
               work_side: int | None = None) -> np.ndarray:
    """Dark channel prior dehazing with per-channel airlight (brightest 0.1 % of the dark channel).

    With `work_side`, airlight and transmission are estimated on a copy downscaled to that longest
    side and the (smooth) transmission is upsampled: about 3x faster at 1280x720, and within about
    1 % of the full-resolution result (run_car_live.py --benchmark checks detection is unchanged)."""
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    scale = 1.0 if not work_side else min(1.0, work_side / max(h, w))
    x = image if scale == 1.0 else cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    patch = max(3, int(round(patch * scale)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(x.min(axis=2), kernel)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    airlight = np.clip(x.reshape(-1, 3)[idx].mean(axis=0), 0.05, 1.0)
    t = 1.0 - omega * cv2.erode((x / airlight).min(axis=2), kernel)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    r = max(8, int(round(min(x.shape[:2]) / 18)))
    t = guided_filter(gray, t.astype(np.float32), r=r)
    if scale < 1.0:
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.clip(t, t_min, 1.0)
    return np.clip((image - airlight) / t[..., None] + airlight, 0.0, 1.0).astype(np.float32)


def autocontrast(image: np.ndarray) -> np.ndarray:
    from .baselines import apply_baseline

    return apply_baseline("autocontrast", image)


# --- learned -----------------------------------------------------------------------------------
class _Learned:
    """Shared wrapper: optional downscale, pad to a multiple, fp16 autocast, residual upsampling."""

    multiple = 8

    def __init__(self, device=None, max_side: int | None = None):
        self.device = get_device(device)
        self.max_side = max_side
        self.last_seconds = 0.0

    def run(self, x: torch.Tensor) -> torch.Tensor:  # (1, 3, h, w) in [0, 1]
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        h, w = image.shape[:2]
        full = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(self.device, torch.float32)
        x = full
        scale = 1.0 if not self.max_side else min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            x = F.interpolate(full, size=(round(h * scale), round(w * scale)), mode="area")
        sh, sw = x.shape[-2:]
        ph, pw = (-sh) % self.multiple, (-sw) % self.multiple
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            y = self.run(xp).float()[..., :sh, :sw]
        if scale < 1.0:
            y = full + F.interpolate(y - x, size=(h, w), mode="bilinear", align_corners=False)
        out = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_seconds = time.perf_counter() - start
        return out


class RestormerDerain(_Learned):
    def __init__(self, device=None, max_side=None):
        super().__init__(device, max_side)
        from .third_party.restormer_arch import Restormer

        self.model = Restormer(dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8],
                               ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")
        state = torch.load(weight_path("restormer_deraining.pth"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["params"])
        self.model.to(self.device).eval()

    def run(self, x):
        return self.model(x)


def instructir_embedding(task: str, device) -> torch.Tensor:
    """Text embedding for an InstructIR instruction: bge-micro-v2 mean pooling + the released LM head."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(INSTRUCTIR_TEXT_MODEL)
    lm = AutoModel.from_pretrained(INSTRUCTIR_TEXT_MODEL).eval()
    inputs = tok([PROMPTS[task]], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        tokens = lm(**inputs)[0]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sentence = F.normalize((tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)
        head = torch.nn.ModuleDict({"fc1": torch.nn.Linear(384, 256), "fc2": torch.nn.Linear(256, 7)})
        state = torch.load(weight_path("lm_instructir-7d.pt"), map_location="cpu", weights_only=True)
        head.load_state_dict(state)
        embd = F.normalize(head["fc1"](sentence), p=2, dim=1)
    return embd.to(device)


class InstructIR(_Learned):
    multiple = 16

    def __init__(self, task: str, device=None, max_side=None):
        super().__init__(device, max_side)
        from .third_party.instructir_arch import create_model

        self.model = create_model(input_channels=3, width=32, enc_blks=[2, 2, 4, 8], middle_blk_num=4,
                                  dec_blks=[2, 2, 2, 2], txtdim=256)
        state = torch.load(weight_path("im_instructir-7d.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.embedding = instructir_embedding(task, self.device)

    def run(self, x):
        return self.model(x, self.embedding)


REPAIRERS = ("none", "autocontrast", "dcp_dehaze", "restormer_derain", "instructir_derain", "instructir_dehaze")
# Trained in this project (task_repair.py); kept out of REPAIRERS so the Phase 2 runs stay as they were.
TRAINED_REPAIRERS = ("task_repair",)


def build(name: str, device=None, max_side: int | None = None):
    """Repairer by name: a callable float RGB -> float RGB."""
    if name == "none":
        return lambda x: x
    if name == "autocontrast":
        return autocontrast
    if name == "dcp_dehaze":
        return dcp_dehaze if not max_side else (lambda x: dcp_dehaze(x, work_side=max_side))
    if name == "restormer_derain":
        return RestormerDerain(device, max_side)
    if name.startswith("instructir_"):
        return InstructIR(name.split("_", 1)[1], device, max_side)
    if name == "task_repair":
        from .task_repair import TaskRepair

        return TaskRepair(device=device)
    raise ValueError(f"unknown repairer {name!r}; choose from {REPAIRERS}")
