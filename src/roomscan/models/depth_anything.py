"""Depth Anything V2 via HuggingFace transformers."""

from __future__ import annotations

import numpy as np

from roomscan.models.base import DepthModel, pick_torch_device

HF_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "metric_indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "metric_indoor_small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}


class _HFDepth(DepthModel):
    def __init__(self, hf_id: str, device: str = "auto"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = pick_torch_device(device)
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(self.device).eval()
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self._torch = torch

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs).predicted_depth          # (1, h', w')
        out = torch.nn.functional.interpolate(out[:, None], size=(h, w), mode="bilinear",
                                              align_corners=False)[0, 0]
        return out.float().cpu().numpy().astype(np.float32)


class DepthAnythingV2(_HFDepth):
    """Relative (affine-invariant) variants. Output is disparity-like (larger = closer)."""

    is_metric = False
    output_kind = "disparity"

    def __init__(self, size: str = "large", device: str = "auto"):
        self.name = f"depth_anything_v2_{size}"
        super().__init__(HF_IDS[size], device)


class DepthAnythingV2Metric(_HFDepth):
    """Metric variant fine-tuned on Hypersim (indoor). Output in metres, no alignment."""

    is_metric = True
    output_kind = "depth"

    def __init__(self, size: str = "metric_indoor", device: str = "auto"):
        self.name = f"depth_anything_v2_{size}"
        super().__init__(HF_IDS[size], device)
