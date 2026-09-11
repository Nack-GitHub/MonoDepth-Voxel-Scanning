"""ARCore Depth API — depth-from-motion on Android (Phase 3, optional).

Only meaningful with a custom capture: the dataset must put the ARCore depth
map into `frame.extra["arcore_depth"]`. Kept here so the config option exists.
"""

from __future__ import annotations

import numpy as np

from roomscan.depth_sources.base import DepthSource
from roomscan.types import Frame


class ARCoreDepth(DepthSource):
    name = "arcore"
    is_metric = True

    def get_depth(self, frame: Frame) -> np.ndarray:
        raise NotImplementedError("needs an Android capture; see docs/architecture/README.md")
