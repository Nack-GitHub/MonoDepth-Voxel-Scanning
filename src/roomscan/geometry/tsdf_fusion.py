"""TSDF fusion wrapper around Open3D ScalableTSDFVolume (Phase 1, ADR-004).

Hides the Open3D API so `pipeline.py` reads like the diagram in the workflow doc,
and so a different backend (e.g. Open3D tensor VoxelBlockGrid on GPU) is one class.
"""

from __future__ import annotations

import numpy as np

from roomscan.types import Intrinsics


class TSDFFusion:
    def __init__(self, voxel_size: float, sdf_trunc: float, depth_trunc: float,
                 depth_min: float, intrinsics: Intrinsics):
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_trunc = depth_trunc
        self.depth_min = depth_min
        self.intrinsics = intrinsics
        self._n_frames = 0
        # TODO(Phase 1):
        # import open3d as o3d
        # self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
        #     voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        # self._intr_o3d = intrinsics.to_open3d()

    def integrate(self, rgb: np.ndarray, depth_m: np.ndarray, pose_c2w: np.ndarray) -> None:
        """depth_m: float32 metres, same HxW as rgb. pose: camera-to-world (inverted here, only here)."""
        # TODO(Phase 1):
        # depth = depth_m.copy(); depth[(depth < self.depth_min) | ~np.isfinite(depth)] = 0
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
        #     depth_scale=1.0, depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)
        # self.volume.integrate(rgbd, self._intr_o3d, np.linalg.inv(pose_c2w))
        self._n_frames += 1
        raise NotImplementedError("Phase 1")

    def extract_mesh(self):
        """-> open3d.geometry.TriangleMesh with vertex normals."""
        raise NotImplementedError("Phase 1")

    @property
    def n_frames(self) -> int:
        return self._n_frames
