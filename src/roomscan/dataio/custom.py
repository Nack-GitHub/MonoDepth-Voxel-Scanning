"""Custom capture loader — a real room recorded by our own app (MVP path, ADR-003).

The app (or a converter from ARKit / ARCore / COLMAP output) writes one folder per scan:

    <root>/<scene_id>/
        intrinsics.json        {"fx","fy","cx","cy","width","height"}  (of the rgb images)
        poses.json             {"<stem>": [[4x4 camera-to-world]], ...} OpenCV camera (x right, y down, z fwd)
        rgb/<stem>.jpg|png     required
        depth/<stem>.png       optional, uint16 millimetres  -> extra["lidar_depth"]   (LiDAR / ARCore)
        confidence/<stem>.png  optional, uint8               -> extra["lidar_confidence"]
        sparse/<stem>.png      optional, uint16 millimetres  -> extra["sparse_depth"]  (VIO points, ADR-011)
        reference.ply          optional GT mesh (e.g. a laser scan) for evaluation; None otherwise

Frames are the sorted stems of rgb/; a stem without a pose is skipped. Every side-channel
is resampled (nearest) to `fusion_resolution`, and intrinsics are scaled to match — the same
contract the ARKitScenes loader honours, so `depth.source` / `depth.aligner` swap unchanged.
There is no GT depth (`gt_depth=None`): `oracle_frame` / `per_scene` cannot run on a capture,
only `identity` (lidar / arcore / metric mono) and `sparse_points` can — which is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from roomscan.dataio.base import SceneDataset
from roomscan.types import Frame, Intrinsics

RGB_SUFFIXES = (".jpg", ".jpeg", ".png")


class CustomCaptureScene(SceneDataset):
    def __init__(self, root: str | Path, scene_id: str, *, fusion_resolution: tuple[int, int] = (256, 192),
                 **_ignored):
        self.root = Path(root)
        self.scene_id = str(scene_id)
        self.scene_dir = self.root / self.scene_id
        self.fusion_resolution = (int(fusion_resolution[0]), int(fusion_resolution[1]))
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"scene dir not found: {self.scene_dir}")

        meta = json.loads((self.scene_dir / "intrinsics.json").read_text())
        native = Intrinsics(float(meta["fx"]), float(meta["fy"]), float(meta["cx"]), float(meta["cy"]),
                            int(meta["width"]), int(meta["height"]))
        self._intrinsics = native.scaled(*self.fusion_resolution)

        poses = json.loads((self.scene_dir / "poses.json").read_text())
        self._poses = {k: np.asarray(v, dtype=np.float64).reshape(4, 4) for k, v in poses.items()}

        rgb_dir = self.scene_dir / "rgb"
        stems = sorted(p.stem for p in rgb_dir.iterdir() if p.suffix.lower() in RGB_SUFFIXES) \
            if rgb_dir.is_dir() else []
        if not stems:
            raise FileNotFoundError(f"no images in {rgb_dir}")
        self._rgb_paths = {p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in RGB_SUFFIXES}
        self._stems = [s for s in stems if s in self._poses]

    # ------------------------------------------------------------------ SceneDataset
    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def __len__(self) -> int:
        return len(self._stems)

    def frame(self, idx: int) -> Frame:
        stem = self._stems[idx]
        bgr = cv2.imread(str(self._rgb_paths[stem]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read {self._rgb_paths[stem]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        extra = {"sky_direction": "Up"}
        d = self._read_mm("depth", stem)
        if d is not None:
            extra["lidar_depth"] = d
        c = self._read_u8("confidence", stem)
        if c is not None:
            extra["lidar_confidence"] = c
        s = self._read_mm("sparse", stem)
        if s is not None:
            extra["sparse_depth"] = s
        return Frame(idx=idx, rgb=rgb, pose_c2w=self._poses[stem], gt_depth=None, extra=extra)

    def gt_mesh(self):
        path = self.scene_dir / "reference.ply"
        if not path.is_file():
            return None
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        return mesh if mesh.has_triangles() else None

    # ------------------------------------------------------------------ internals
    def _read_mm(self, sub: str, stem: str) -> np.ndarray | None:
        p = self.scene_dir / sub / f"{stem}.png"
        if not p.is_file():
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError(f"{p}: expected uint16 millimetre PNG")
        return self._to_grid(raw.astype(np.float32) / 1000.0)

    def _read_u8(self, sub: str, stem: str) -> np.ndarray | None:
        p = self.scene_dir / sub / f"{stem}.png"
        if not p.is_file():
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return self._to_grid(raw.astype(np.uint8))

    def _to_grid(self, img: np.ndarray) -> np.ndarray:
        w, h = self.fusion_resolution
        if img.shape[1] == w and img.shape[0] == h:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
