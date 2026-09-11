"""TSDF fusion wrapper around Open3D ScalableTSDFVolume (ADR-004).

Hides the Open3D API so pipeline.py reads like the block diagram, and so a
GPU backend can be swapped in behind the same two methods.
"""

from __future__ import annotations

import cv2
import numpy as np

from roomscan.types import Intrinsics


class TSDFFusion:
    def __init__(self, voxel_size: float, sdf_trunc: float, depth_trunc: float,
                 depth_min: float, intrinsics: Intrinsics):
        import open3d as o3d

        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc)
        self.depth_trunc = float(depth_trunc)
        self.depth_min = float(depth_min)
        self.intrinsics = intrinsics
        self._intr_o3d = intrinsics.to_open3d()
        self._n_frames = 0
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size, sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def integrate(self, rgb: np.ndarray, depth_m: np.ndarray, pose_c2w: np.ndarray) -> None:
        """depth_m: float32 metres on the intrinsics grid. pose: camera-to-world — inverted HERE, only here."""
        import open3d as o3d

        w, h = self.intrinsics.width, self.intrinsics.height
        if depth_m.shape != (h, w):
            raise ValueError(f"depth {depth_m.shape[::-1]} != intrinsics grid {(w, h)}")
        depth = np.ascontiguousarray(depth_m, dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < self.depth_min)] = 0.0
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)
        self.volume.integrate(rgbd, self._intr_o3d, np.linalg.inv(pose_c2w))
        self._n_frames += 1

    def extract_mesh(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    @property
    def n_frames(self) -> int:
        return self._n_frames
