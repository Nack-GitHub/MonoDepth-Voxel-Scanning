"""DepthModel — a neural network that maps RGB -> depth-ish map.

Kept separate from DepthSource so Exp4 (model size) is just a different
backend behind the same `MonocularDepth` source. Inference only (ADR-008).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class DepthModel(ABC):
    name: ClassVar[str]
    is_metric: ClassVar[bool]        # True -> output is already metres; False -> relative/affine
    param_count: ClassVar[int | None] = None   # for the model-size table

    @abstractmethod
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) float32.

        Relative models return whatever the network outputs (often inverse
        depth / disparity). Say so in `output_kind` so the aligner can invert.
        """

    output_kind: ClassVar[str] = "depth"   # "depth" | "disparity"
