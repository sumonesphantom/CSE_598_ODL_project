"""Lazy local VLM client (Qwen2.5-VL by default) with an on-disk answer cache.

The client loads weights only on the first real generation, supports multiple
images per prompt (overlay plus depth map), and never prints. Tests inject a
``generate_fn`` so the suite stays offline and model-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import RESULTS_DIR, get_device
from .data import to_pil


@dataclass
class VLMResponse:
    text: str
    latency_ms: float
    cached: bool
    raw: dict | None


class VLMClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        cache_dir: str | Path = RESULTS_DIR / "vlm" / "cache",
        max_new_tokens: int = 160,
        generate_fn: Callable[[list[np.ndarray], str], str] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.generate_fn = generate_fn
        self._processor = None
        self._model = None
        self._model_device = None

    def available(self) -> tuple[bool, str]:
        """Whether the runtime libraries are installed. Does not load weights."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            return False, str(exc)
        return True, ""

    def _load(self):
        if self._model is not None:
            return self._processor, self._model, self._model_device

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = get_device(self.device)
        dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
        try:
            processor = AutoProcessor.from_pretrained(self.model)
            model = AutoModelForImageTextToText.from_pretrained(self.model, dtype=dtype)
            model = model.to(device).eval()
        except Exception as exc:
            raise RuntimeError(f"could not load {self.model}: {exc}") from exc

        self._processor = processor
        self._model = model
        self._model_device = device
        return processor, model, device

    def _cache_path(self, cache_key: str, prompt: str = "") -> Path:
        payload = f"{self.model}|{self.max_new_tokens}|{cache_key}|{prompt}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _write_cache(self, path: Path, text: str, raw: dict | None) -> None:
        """Best-effort atomic cache write; failures never break generation."""
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"text": text, "raw": raw}))
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _generate(self, images: list[np.ndarray], prompt: str) -> str:
        import torch

        processor, model, device = self._load()
        pils = [to_pil(np.asarray(image, dtype=np.float32)) for image in images]
        content = [{"type": "image", "image": pil} for pil in pils]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=pils, return_tensors="pt")
        inputs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        prompt_length = inputs["input_ids"].shape[-1]
        return processor.decode(generated[0][prompt_length:], skip_special_tokens=True)

    def ask(
        self,
        images: list[np.ndarray],
        prompt: str,
        cache_key: str,
        use_cache: bool = True,
    ) -> VLMResponse:
        start = time.perf_counter()
        path = self._cache_path(cache_key, prompt)

        if use_cache and path.is_file():
            try:
                payload = json.loads(path.read_text())
                return VLMResponse(
                    text=str(payload.get("text", "")),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    cached=True,
                    raw=payload.get("raw"),
                )
            except (OSError, ValueError, TypeError):
                pass  # corrupt cache entry: regenerate below

        if self.generate_fn is not None:
            text = str(self.generate_fn(images, prompt))
        else:
            text = self._generate(images, prompt)

        latency_ms = (time.perf_counter() - start) * 1000.0
        if use_cache:
            self._write_cache(path, text, None)
        return VLMResponse(text=text, latency_ms=latency_ms, cached=False, raw=None)
