"""Pixel + depth -> 3D. The one formula used everywhere in this project.

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
"""

from __future__ import annotations

import numpy as np

from roomscan.types import Intrinsics


def backproject(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """(H, W) metric depth -> (N, 3) points in CAMERA frame. Invalid (<=0, non-finite) dropped."""
    h, w = depth.shape
    if (w, h) != (intr.width, intr.height):
        raise ValueError(f"depth is {w}x{h} but intrinsics are {intr.width}x{intr.height}; "
                         "use Intrinsics.scaled()")
    v, u = np.mgrid[0:h, 0:w]
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    x = (u[valid] - intr.cx) * d[valid] / intr.fx
    y = (v[valid] - intr.cy) * d[valid] / intr.fy
    return np.stack([x, y, d[valid]], axis=1)


def to_world(points_cam: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    """(N, 3) camera-frame points -> world frame using a 4x4 camera-to-world pose."""
    homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (pose_c2w @ homo.T).T[:, :3]
