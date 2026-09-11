"""Upper bound: the dataset's own depth sensor (Phase 1)."""

from __future__ import annotations

import numpy as np

from roomscan.depth_sources.base import DepthSource
from roomscan.types import Frame


class GroundTruthDepth(DepthSource):
    name = "gt"
    is_metric = True

    def get_depth(self, frame: Frame) -> np.ndarray:
        if frame.gt_depth is None:
            raise ValueError(f"frame {frame.idx} has no GT depth; dataset does not provide it")
        return frame.gt_depth
