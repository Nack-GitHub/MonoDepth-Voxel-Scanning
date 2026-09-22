"""MonocularDepth upright rotation (no torch: a fake DepthModel records what it was fed)."""

from __future__ import annotations

import numpy as np
import pytest

from roomscan.depth_sources.monocular import MonocularDepth
from roomscan.models.base import DepthModel
from roomscan.types import Frame


class _EchoModel(DepthModel):
    """Returns the red channel + 1 as 'depth' and keeps the input it saw."""

    name = "echo"
    is_metric = True

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        self.seen = rgb.copy()
        return rgb[..., 0].astype(np.float32) + 1.0


def _frame(sky: str) -> Frame:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, 0, 0] = 255                          # mark the left column
    return Frame(idx=0, rgb=rgb, pose_c2w=np.eye(4), extra={"sky_direction": sky})


def test_sky_left_turns_clockwise():
    """sky=Left: the left edge is the sky -> the model must see it as the top row."""
    model = _EchoModel()
    MonocularDepth(model).get_depth(_frame("Left"))
    assert model.seen.shape == (6, 4, 3)
    assert np.all(model.seen[0, :, 0] == 255) and model.seen[1:, :, 0].sum() == 0


def test_sky_right_turns_counter_clockwise():
    model = _EchoModel()
    f = _frame("Right")
    f.rgb = np.ascontiguousarray(f.rgb[:, ::-1])  # marker on the right column = the sky
    MonocularDepth(model).get_depth(f)
    assert np.all(model.seen[0, :, 0] == 255) and model.seen[1:, :, 0].sum() == 0


@pytest.mark.parametrize("sky", ["Up", "Left", "Down", "Right"])
def test_prediction_rotated_back_to_frame(sky):
    f = _frame(sky)
    depth = MonocularDepth(_EchoModel()).get_depth(f)
    assert depth.shape == f.rgb.shape[:2]
    assert np.array_equal(depth, f.rgb[..., 0].astype(np.float32) + 1.0)
