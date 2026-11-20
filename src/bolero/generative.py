"""Generative weather: rain and dust storms painted onto clean images by InstructPix2Pix.

Severity is the edit strength. InstructPix2Pix's image guidance scale controls how closely
the output follows the input; lower means a stronger edit. Levels 1-3 use 2.0 / 1.75 / 1.5,
chosen on a pilot of COCO street scenes: below 1.5 the model starts replacing the scene
(new buildings, new people) instead of adding weather to it.

Whether the content survived the edit is judged by CLIP image similarity to the clean
image, never by the detector or classifier under test. Filtering on "the detector still
finds the boxes" would drop exactly the hard cases and inflate the scores.

The heavy models are imported lazily so the rest of the package works without diffusers.
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import torch
from PIL import Image

from .config import get_device

EDIT_MODEL = "timbrooks/instruct-pix2pix"
CLIP_MODEL = "openai/clip-vit-base-patch32"
PROMPTS = {"rain": "make it raining heavily", "dust": "make it a sandstorm"}
IMAGE_GUIDANCE = {1: 2.0, 2: 1.75, 3: 1.5}   # lower = stronger edit
TEXT_GUIDANCE = 7.5
STEPS = 20
EDIT_SIDE = 512                               # Stable Diffusion 1.5 resolution (short side)


def edit_size(width: int, height: int, side: int = EDIT_SIDE) -> tuple[int, int]:
    """Size to edit at: short side `side`, both sides multiples of 8 (the VAE's stride)."""
    scale = side / min(width, height)
    return max(8, int(round(width * scale / 8)) * 8), max(8, int(round(height * scale / 8)) * 8)


def edit_seed(image_id: str, condition: str, level: int, seed: int = 42) -> int:
    """Deterministic per-(image, condition, level) diffusion seed."""
    return (seed * 1_000_003 + zlib.crc32(f"{image_id}|{condition}|{level}".encode())) % (2**31)


class WeatherEditor:
    """InstructPix2Pix with fixed prompts per condition; returns an image the size of the input."""

    def __init__(self, device: str | None = None, steps: int = STEPS):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        self.device = get_device(device)
        half = self.device.type == "cuda"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            EDIT_MODEL, torch_dtype=torch.float16 if half else torch.float32,
            variant="fp16" if half else None, safety_checker=None, requires_safety_checker=False,
        ).to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps
        self.last_seconds = 0.0

    @torch.no_grad()
    def __call__(self, image: Image.Image, condition: str, level: int, seed: int) -> Image.Image:
        start = time.perf_counter()
        size = image.size
        work = image.convert("RGB").resize(edit_size(*size), Image.BICUBIC)
        generator = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(PROMPTS[condition], image=work, num_inference_steps=self.steps,
                        image_guidance_scale=IMAGE_GUIDANCE[level], guidance_scale=TEXT_GUIDANCE,
                        generator=generator).images[0]
        self.last_seconds = time.perf_counter() - start
        return out.resize(size, Image.BICUBIC)


class ClipEmbedder:
    """Unit-length CLIP image embeddings, for content-preservation checks and set distances."""

    def __init__(self, device: str | None = None, model: str = CLIP_MODEL):
        from transformers import CLIPModel, CLIPProcessor

        self.device = get_device(device)
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=[im.convert("RGB") for im in images[i:i + batch_size]],
                                    return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb if isinstance(emb, torch.Tensor) else emb.pooler_output
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
