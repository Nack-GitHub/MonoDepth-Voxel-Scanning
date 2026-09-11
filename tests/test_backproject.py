import numpy as np
import pytest

from roomscan.geometry.backproject import backproject, to_world
from roomscan.types import Intrinsics


def test_principal_point_maps_to_optical_axis(intr):
    depth = np.zeros((intr.height, intr.width), dtype=np.float32)
    depth[int(intr.cy), int(intr.cx)] = 3.0
    pts = backproject(depth, intr)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 3.0])


def test_flat_wall_keeps_depth(flat_wall_frame, intr):
    pts = backproject(flat_wall_frame.gt_depth, intr)
    assert pts.shape == (intr.width * intr.height, 3)
    np.testing.assert_allclose(pts[:, 2], 2.0)


def test_invalid_pixels_dropped(intr):
    depth = np.full((intr.height, intr.width), 1.0, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    assert len(backproject(depth, intr)) == intr.width * intr.height - 2


def test_resolution_mismatch_is_loud(intr):
    with pytest.raises(ValueError, match="scaled"):
        backproject(np.ones((10, 10), dtype=np.float32), intr)


def test_to_world_applies_translation():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    out = to_world(np.array([[0.0, 0.0, 1.0]]), pose)
    np.testing.assert_allclose(out, [[1.0, 2.0, 4.0]])


def test_intrinsics_scaled_halves_everything():
    k = Intrinsics(1000, 1000, 640, 480, 1280, 960).scaled(640, 480)
    assert (k.fx, k.fy, k.cx, k.cy, k.width, k.height) == (500, 500, 320, 240, 640, 480)
