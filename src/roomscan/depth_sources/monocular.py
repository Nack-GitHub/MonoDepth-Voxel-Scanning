"""Monocular depth from a pretrained network (Phase 2). The variable under study."""

from __future__ import annotations

import numpy as np

from roomscan.depth_sources.base import DepthSource
from roomscan.models.base import DepthModel
from roomscan.types import Frame


class MonocularDepth(DepthSource):
    def __init__(self, model: DepthModel):
        self.model = model

    @property
    def name(self) -> str:
        return f"mono:{self.model.name}"

    @property
    def is_metric(self) -> bool:
        return self.model.is_metric

    def get_depth(self, frame: Frame) -> np.ndarray:
        # TODO(Phase 2): run model, convert disparity -> depth if output_kind == "disparity",
        # resize to frame.rgb resolution, mask invalid (<=0, inf).
        raise NotImplementedError("Phase 2")
