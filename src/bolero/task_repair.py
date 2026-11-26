"""A small repair network trained for the frozen detector, on real weather photos.

Phase 2 found that the off-the-shelf repairs (autocontrast, dark-channel dehazing, Restormer) give
little or nothing on real rain and dust photos, and dehazing even hurts real dust. This repairer is
trained instead of hand-designed: a light residual U-Net is placed in front of the car's detector
(torchvision Mask R-CNN, COCO weights) and trained to lower that detector's own training loss on
real weather photos with box labels (XWOD train split). The detector never changes (all its weights
frozen, batch-norm statistics fixed), so this keeps the project's rule that the recognizer is not
retrained; only the preprocessing step is learned.

Safeguards against harming the detector:
  * the output layers start at zero, so before training the network is exactly the identity;
  * clean COCO street photos are mixed into every batch, with an extra penalty for changing them;
  * a penalty on the size of the change keeps the repair small on weather photos too.

The network runs at the frame's resolution and costs a few milliseconds per frame on the GPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import CACHE_DIR, get_device

TASK_REPAIR_PATH = CACHE_DIR / "restoration" / "task_repair.pt"


def _block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True))


class RepairNet(nn.Module):
    """Residual U-Net (3 scales) plus a global per-channel gain and offset (for colour casts and
    lost contrast). Output = clamp(gain * x + offset + residual), identity at initialization."""

    def __init__(self, width: int = 24):
        super().__init__()
        c = width
        self.enc1, self.enc2, self.enc3 = _block(3, c), _block(c, 2 * c, 2), _block(2 * c, 4 * c, 2)
        self.dec2, self.dec1 = _block(4 * c + 2 * c, 2 * c), _block(2 * c + c, c)
        self.out = nn.Conv2d(c, 3, 3, 1, 1)
        self.affine = nn.Linear(4 * c, 6)
        for layer in (self.out, self.affine):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.enc1(xp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        residual = self.out(d1)[..., :h, :w]
        g = self.affine(e3.mean(dim=(2, 3)))
        gain, offset = 1.0 + torch.tanh(g[:, :3])[..., None, None], 0.5 * torch.tanh(g[:, 3:])[..., None, None]
        return (gain * (x - 0.5) + 0.5 + offset + residual).clamp(0.0, 1.0)


class TaskRepair:
    """Float RGB (H, W, 3) in [0, 1] -> same, using trained weights (scripts/train_task_repair.py)."""

    def __init__(self, path: Path = TASK_REPAIR_PATH, device=None, fp16: bool = True):
        self.device = get_device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net = RepairNet(ckpt["width"]).to(self.device).eval()
        self.net.load_state_dict(ckpt["state"])
        self.fp16 = fp16 and self.device.type == "cuda"
        self.last_seconds = 0.0

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(self.device, torch.float32)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            y = self.net(x).float()
        out = y[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        self.last_seconds = time.perf_counter() - start
        return out
