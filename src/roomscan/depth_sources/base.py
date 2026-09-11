"""DepthSource — THE swap point of the whole project (ADR-001).

The paper's experiment and the future product are the same code: change
`depth.source` in the config, nothing else moves.

Contract:
  * returns (H, W) float32 at the dataset's intrinsics resolution
  * 0 / NaN = no data
  * `is_metric` tells the pipeline whether a ScaleAligner is required.
    A DepthSource must NOT align itself — that's the aligner's job (ADR-002),
    otherwise the {source} x {aligner} grid in Exp1 is not a clean ablation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from roomscan.types import Frame


class DepthSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_metric(self) -> bool: ...

    @abstractmethod
    def get_depth(self, frame: Frame) -> np.ndarray: ...
