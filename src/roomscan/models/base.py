"""DepthModel — a network mapping RGB -> depth-ish map. Inference only (ADR-008).
Kept separate from DepthSource so Exp4 (model size) is a different backend behind
the same MonocularDepth source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class DepthModel(ABC):
    name: str
    is_metric: ClassVar[bool]            # True -> metres; False -> affine-invariant
    output_kind: ClassVar[str] = "depth"  # "depth" | "disparity"
    param_count: int | None = None

    @abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB, upright -> (H, W) float32 (same H, W)."""


def pick_torch_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
