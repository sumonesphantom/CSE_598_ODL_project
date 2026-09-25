"""The system under test: a frozen ImageNet-pretrained ResNet-50."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .config import CLASSIFIER_MODEL, get_device
from .data import to_pil


@dataclass
class Predictions:
    index: np.ndarray        # top-1 ImageNet-1k index
    confidence: np.ndarray   # top-1 softmax probability
    true_prob: np.ndarray    # softmax probability of the true class
    seconds: float           # wall time for the batch


class FrozenClassifier:
    """ResNet-50 with frozen weights. Never trained or fine-tuned in this project."""

    def __init__(self, model_name: str = CLASSIFIER_MODEL, device: str | None = None):
        self.device = get_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, images: list[np.ndarray], true_index: np.ndarray) -> Predictions:
        start = time.perf_counter()
        inputs = self.processor(images=[to_pil(im) for im in images], return_tensors="pt")
        logits = self.model(pixel_values=inputs["pixel_values"].to(self.device)).logits
        probs = torch.softmax(logits.float(), dim=-1)
        conf, index = probs.max(dim=-1)
        true_prob = probs.gather(1, torch.as_tensor(true_index, device=probs.device).view(-1, 1))
        if self.device.type == "mps":
            torch.mps.synchronize()
        return Predictions(
            index=index.cpu().numpy(),
            confidence=conf.cpu().numpy(),
            true_prob=true_prob.squeeze(1).cpu().numpy(),
            seconds=time.perf_counter() - start,
        )


class BatchPredictor:
    """Accumulates (metadata, image) pairs and classifies them in batches."""

    def __init__(self, classifier: FrozenClassifier, batch_size: int = 64):
        self.classifier = classifier
        self.batch_size = batch_size
        self._meta: list[dict] = []
        self._images: list[np.ndarray] = []
        self.rows: list[dict] = []

    def add(self, meta: dict, image: np.ndarray) -> None:
        self._meta.append(meta)
        self._images.append(image)
        if len(self._images) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._images:
            return
        true_index = np.array([m["label_index"] for m in self._meta])
        preds = self.classifier.predict(self._images, true_index)
        per_image_ms = 1000.0 * preds.seconds / len(self._images)
        for i, meta in enumerate(self._meta):
            self.rows.append({
                **meta,
                "pred_index": int(preds.index[i]),
                "pred_conf": float(preds.confidence[i]),
                "true_prob": float(preds.true_prob[i]),
                "correct": bool(preds.index[i] == meta["label_index"]),
                "classify_ms": per_image_ms,
            })
        self._meta, self._images = [], []
