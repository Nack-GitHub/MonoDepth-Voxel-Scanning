"""Monocular depth from a pretrained network. The variable under study.

Responsibilities that live HERE (not in the model, not in the aligner):
  * rotate the image upright using frame.extra['sky_direction'] (ARKitScenes
    stores landscape frames from a portrait-held iPad) and rotate the result back
  * convert disparity -> depth-like (1/x) so every aligner sees "bigger = farther"
  * mask non-finite / non-positive values to 0
Output is at the RGB's native resolution; the pipeline resamples to the fusion grid.
"""

from __future__ import annotations

import numpy as np

from roomscan.depth_sources.base import DepthSource
from roomscan.models.base import DepthModel
from roomscan.types import Frame

_ROT = {"Up": 0, "Left": 1, "Down": 2, "Right": 3}   # k for np.rot90 to make the image upright


class MonocularDepth(DepthSource):
    def __init__(self, model: DepthModel, max_input_width: int | None = 1024):
        self.model = model
        self.max_input_width = max_input_width

    @property
    def name(self) -> str:
        return f"mono:{self.model.name}"

    @property
    def is_metric(self) -> bool:
        return self.model.is_metric

    def get_depth(self, frame: Frame) -> np.ndarray:
        import cv2

        rgb = frame.rgb
        h, w = rgb.shape[:2]
        scale = 1.0
        if self.max_input_width and w > self.max_input_width:
            scale = self.max_input_width / w
            rgb = cv2.resize(rgb, (self.max_input_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

        k = _ROT.get(str(frame.extra.get("sky_direction", "Up")), 0)
        pred = self.model.predict(np.ascontiguousarray(np.rot90(rgb, k)))
        pred = np.rot90(pred, -k)

        if self.model.output_kind == "disparity":
            with np.errstate(divide="ignore", invalid="ignore"):
                pred = np.where(pred > 1e-6, 1.0 / pred, 0.0)
        pred = np.where(np.isfinite(pred) & (pred > 0), pred, 0.0).astype(np.float32)
        if scale != 1.0:
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(pred)
