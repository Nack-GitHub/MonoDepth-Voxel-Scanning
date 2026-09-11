"""MiDaS via torch.hub (Exp4 small-model row)."""

from __future__ import annotations

import numpy as np

from roomscan.models.base import DepthModel, pick_torch_device


class MiDaS(DepthModel):
    is_metric = False
    output_kind = "disparity"

    def __init__(self, variant: str = "MiDaS_small", device: str = "auto"):
        import torch

        self.name = f"midas_{variant.lower().removeprefix('midas_')}"
        self.device = pick_torch_device(device)
        self.model = torch.hub.load("intel-isl/MiDaS", variant, trust_repo=True).to(self.device).eval()
        tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = tf.small_transform if "small" in variant.lower() else tf.dpt_transform
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self._torch = torch

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        x = self.transform(rgb).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        out = torch.nn.functional.interpolate(out[:, None], size=(h, w), mode="bilinear",
                                              align_corners=False)[0, 0]
        return out.float().cpu().numpy().astype(np.float32)
