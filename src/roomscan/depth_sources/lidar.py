"""Consumer LiDAR depth as captured by ARKit on iPad/iPhone Pro (ADR-009).

The dataset puts ARKit's lowres_depth (metres) into frame.extra["lidar_depth"]
and the 0/1/2 confidence map into frame.extra["lidar_confidence"].
This row is the direct competitor of the mono rows: "what iPhone Pro does today".
"""

from __future__ import annotations

import numpy as np

from roomscan.depth_sources.base import DepthSource
from roomscan.types import Frame


class LiDARDepth(DepthSource):
    name = "lidar"
    is_metric = True

    def __init__(self, min_confidence: int = 0):
        # ARKit confidence: 0 = low, 1 = medium, 2 = high. Default keeps everything;
        # Phase 4 failure analysis can sweep this.
        self.min_confidence = min_confidence

    def get_depth(self, frame: Frame) -> np.ndarray:
        try:
            depth = frame.extra["lidar_depth"]
        except KeyError:
            raise ValueError(f"frame {frame.idx} has no extra['lidar_depth']; "
                             "dataset must be ARKitScenes or a LiDAR capture") from None
        conf = frame.extra.get("lidar_confidence")
        if conf is not None and self.min_confidence > 0:
            depth = np.where(conf >= self.min_confidence, depth, 0.0).astype(np.float32)
        return depth
