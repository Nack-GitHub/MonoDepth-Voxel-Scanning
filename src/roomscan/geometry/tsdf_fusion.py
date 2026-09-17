"""TSDF fusion wrapper around Open3D ScalableTSDFVolume (ADR-004).

Hides the Open3D API so pipeline.py reads like the block diagram, and so a
GPU backend can be swapped in behind the same two methods.

Per-pixel integer weights (ADR-012): the legacy volume adds weight 1 per
integrate() call, so a pixel that should count `w` times is integrated `w`
times. `integrate(..., weights=)` decomposes a (H, W) int map into nested
masks — pixels with weight >= level — and integrates each level once, so the
cost is max(weights) passes, not one pass per distinct value.
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

    def integrate(self, rgb: np.ndarray, depth_m: np.ndarray, pose_c2w: np.ndarray,
                  weights: np.ndarray | None = None) -> None:
        """depth_m: float32 metres on the intrinsics grid. pose: camera-to-world — inverted HERE, only here.
        weights: optional (H, W) non-negative ints; 0 drops the pixel, w integrates it w times."""
        import open3d as o3d

        w, h = self.intrinsics.width, self.intrinsics.height
        if depth_m.shape != (h, w):
            raise ValueError(f"depth {depth_m.shape[::-1]} != intrinsics grid {(w, h)}")
        depth = np.ascontiguousarray(depth_m, dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < self.depth_min)] = 0.0
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        extrinsic = np.linalg.inv(pose_c2w)

        if weights is None:
            levels = [(depth, 1)]
        else:
            if weights.shape != (h, w):
                raise ValueError(f"weights {weights.shape[::-1]} != intrinsics grid {(w, h)}")
            wi = np.rint(weights).astype(np.int32)
            levels = [(np.where(wi >= lv, depth, 0.0).astype(np.float32), 1)
                      for lv in range(1, int(wi.max()) + 1)]
        for d, _ in levels:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), o3d.geometry.Image(np.ascontiguousarray(d)),
                depth_scale=1.0, depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)
            self.volume.integrate(rgbd, self._intr_o3d, extrinsic)
        self._n_frames += 1

    def extract_mesh(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    @property
    def n_frames(self) -> int:
        return self._n_frames


def confidence_weights(confidence: np.ndarray | None, table) -> np.ndarray | None:
    """ARKit confidence {0,1,2} -> integer integration weight via `table` = [w0, w1, w2].
    None table (the default) or no confidence map -> None = unweighted (every pixel once)."""
    if table is None or confidence is None:
        return None
    lut = np.asarray([int(v) for v in table], dtype=np.int32)
    if lut.size == 0 or (lut < 0).any():
        raise ValueError(f"fusion.confidence_weights must be non-negative ints, got {list(table)}")
    return lut[np.clip(confidence.astype(np.int64), 0, lut.size - 1)]
