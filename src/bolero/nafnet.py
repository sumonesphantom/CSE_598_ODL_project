
from pathlib import Path

import numpy as np
import torch

from basicsr.models.archs.NAFNet_arch import NAFNet
from bolero.config import get_device


class NAFNetRestorer:
    """Inference adapter for the official NAFNet-SIDD width-32 checkpoint.

    Input and output images use HWC RGB format with float32 values in [0, 1].
    """

    def __init__(self, checkpoint: Path, device: str | None = None):
        self.checkpoint = Path(checkpoint)

        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"NAFNet checkpoint not found: {self.checkpoint}")

        self.device = torch.device(get_device(device))

        self.model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2],
        )

        saved = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(saved, dict) or "params" not in saved:
            raise ValueError(
                "Unexpected NAFNet checkpoint format. Expected a dictionary "
                "containing the 'params' state dictionary."
            )

        self.model.load_state_dict(saved["params"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Denoise one HWC RGB image and return HWC RGB float32 output."""

        image = np.asarray(image, dtype=np.float32)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected an HWC RGB image with 3 channels; got {image.shape}."
            )

        if not np.isfinite(image).all():
            raise ValueError("Input image contains NaN or infinite values.")

        if image.min() < -1e-5 or image.max() > 1.00001:
            raise ValueError(
                "Expected input pixel values in [0, 1]. "
                f"Observed range: [{image.min()}, {image.max()}]."
            )

        tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1).unsqueeze(0).to(self.device)

        restored = self.model(tensor)
        restored = restored.clamp(0.0, 1.0)

        output = (
            restored.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if output.shape != image.shape:
            raise RuntimeError(
                f"NAFNet changed image dimensions: "
                f"{image.shape} -> {output.shape}"
            )

        return output
