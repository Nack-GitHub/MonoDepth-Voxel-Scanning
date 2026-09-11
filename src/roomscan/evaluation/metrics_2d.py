"""Per-frame depth quality vs GT depth (Phase 2). Diagnostic, not the headline.

Tells you whether a bad mesh comes from bad depth or from the fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics2D:
    rmse: float          # metres
    abs_rel: float
    delta1: float        # % pixels with max(d/gt, gt/d) < 1.25
    delta2: float        # < 1.25^2
    delta3: float        # < 1.25^3
    n_valid: int


def depth_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1,
                  max_depth: float = 5.0) -> Metrics2D:
    """TODO(Phase 2): mask = finite & in [min,max] on both; standard KITTI/NYU formulas."""
    raise NotImplementedError("Phase 2")


def mean_metrics(items: list[Metrics2D]) -> Metrics2D:
    raise NotImplementedError("Phase 2")
