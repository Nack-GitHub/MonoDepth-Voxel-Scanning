"""Depth Anything V2 backends via HuggingFace `transformers` (Phase 2)."""

from __future__ import annotations

import numpy as np

from roomscan.models.base import DepthModel


class DepthAnythingV2(DepthModel):
    """Relative (affine-invariant) variants: small / base / large."""

    is_metric = False
    output_kind = "disparity"

    def __init__(self, size: str = "large", device: str = "cpu"):
        self.name = f"depth_anything_v2_{size}"
        # TODO(Phase 2): AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-{Size}-hf")
        raise NotImplementedError("Phase 2")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DepthAnythingV2Metric(DepthModel):
    """Metric variant fine-tuned on Hypersim (indoor). No alignment needed."""

    name = "depth_anything_v2_metric_indoor"
    is_metric = True
    output_kind = "depth"

    def __init__(self, device: str = "cpu"):
        raise NotImplementedError("Phase 2")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError
