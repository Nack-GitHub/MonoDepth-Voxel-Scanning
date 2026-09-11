"""Core data types shared by every stage.

Conventions (violating these is the #1 cause of broken meshes):
  * Depth is float32, in METERS, shape (H, W). 0 or NaN = no data.
  * Poses are 4x4 float64 camera-to-world (T_cw). Open3D's integrate() wants
    world-to-camera, so the pipeline passes inv(pose) — nowhere else.
  * Intrinsics are tied to a resolution. If you resize the image, call
    Intrinsics.scaled() — never reuse K across resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> Intrinsics:
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                   width=width, height=height)

    def scaled(self, width: int, height: int) -> Intrinsics:
        """Intrinsics for the same camera after resizing the image to (width, height)."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)

    def to_open3d(self):
        import open3d as o3d  # local import keeps `types` import-light for tests

        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


@dataclass
class Frame:
    """One time step of a scan. Everything a DepthSource might need."""

    idx: int
    rgb: np.ndarray                       # (H, W, 3) uint8, RGB order
    pose_c2w: np.ndarray                  # (4, 4) float64, camera -> world
    gt_depth: np.ndarray | None = None    # (H, W) float32 meters; None if dataset has no GT
    extra: dict[str, Any] = field(default_factory=dict)  # sensor side-channels (ARCore/LiDAR)

    @property
    def has_valid_pose(self) -> bool:
        return bool(np.all(np.isfinite(self.pose_c2w)))


@dataclass
class Timing:
    """Wall-clock seconds per stage. Feeds the 'time / scene' column of every table."""

    depth: float = 0.0
    align: float = 0.0
    fusion: float = 0.0
    extract: float = 0.0
    postprocess: float = 0.0
    eval: float = 0.0

    @property
    def total(self) -> float:
        return self.depth + self.align + self.fusion + self.extract + self.postprocess + self.eval
