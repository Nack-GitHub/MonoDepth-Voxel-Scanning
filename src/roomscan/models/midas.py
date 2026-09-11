"""MiDaS backends via torch.hub (Phase 3 / Exp4)."""

from __future__ import annotations

import numpy as np

from roomscan.models.base import DepthModel


class MiDaS(DepthModel):
    is_metric = False
    output_kind = "disparity"

    def __init__(self, variant: str = "MiDaS_small", device: str = "cpu"):
        self.name = f"midas_{variant.lower().removeprefix('midas_')}"
        raise NotImplementedError("Phase 3")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError
