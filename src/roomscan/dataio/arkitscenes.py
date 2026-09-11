"""ARKitScenes scene loader (Phase 0). See ADR-009 for every decision below.

Expected on-disk layout (raw assets, downloaded via Apple's download_data.py):

    data/arkitscenes/raw/<split>/<video_id>/
    ├── wide/                       <video_id>_<ts>.png   1920x1440 RGB
    ├── wide_intrinsics/            <video_id>_<ts>.pincam
    ├── lowres_wide/                256x192 RGB (same camera, downscaled)
    ├── lowres_wide_intrinsics/
    ├── lowres_depth/               <video_id>_<ts>.png   256x192, ARKit (LiDAR-fused)  -> extra["lidar_depth"]
    ├── confidence/                 <video_id>_<ts>.png   0/1/2                          -> extra["lidar_confidence"]
    ├── highres_depth/              <video_id>_<ts>.png   Faro laser GT, SUBSET of scans -> gt_depth
    ├── lowres_wide.traj            "ts rx ry rz tx ty tz" per line (axis-angle rotation)
    ├── <video_id>_3dod_mesh.ply    ARKit-generated mesh (NOT laser GT)
    └── reference_mesh.ply          built by scripts/build_reference_mesh.py (cache)

Frame mapping decided in ADR-009:
    rgb        = wide, resized to the fusion resolution (lowres_depth size)
    gt_depth   = highres_depth  (None if this scan / frame has none)
    pose_c2w   = from .traj      ** VERIFY c2w vs w2c on first sanity check **
    extra      = {"lidar_depth": lowres_depth (m), "lidar_confidence": confidence}
    intrinsics = wide .pincam scaled to fusion resolution (Intrinsics.scaled)

Pitfalls to handle here and nowhere else:
    * depth PNGs are uint16 millimetres (VERIFY) -> /1000 -> float32 metres
    * frame <-> pose matched by timestamp in filename, tolerance ~1e-3 s; unmatched -> skipped
    * .pincam: "width height fx fy cx cy" (single line) per frame; assume constant per scan
"""

from __future__ import annotations

from pathlib import Path

from roomscan.dataio.base import SceneDataset
from roomscan.types import Frame, Intrinsics


class ARKitScenesScene(SceneDataset):
    REFERENCES = ("faro_fused", "arkit_mesh")

    def __init__(self, root: str | Path, scene_id: str, *, split: str = "Training",
                 reference: str = "faro_fused", fusion_resolution: tuple[int, int] = (256, 192)):
        if reference not in self.REFERENCES:
            raise ValueError(f"reference must be one of {self.REFERENCES}")
        self.root = Path(root) / "raw" / split / scene_id
        self.scene_id = scene_id
        self.reference = reference
        self.fusion_resolution = fusion_resolution
        # TODO(Phase 0): parse .traj -> {ts: pose}; list lowres_depth/*.png -> timestamps;
        # keep only frames with a pose; read one .pincam -> self._intr
        raise NotImplementedError("Phase 0: implement ARKitScenes loader")

    @property
    def intrinsics(self) -> Intrinsics:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def frame(self, idx: int) -> Frame:
        raise NotImplementedError

    def gt_mesh(self):
        """faro_fused -> reference_mesh.ply (raise with a helpful message if not built yet);
        arkit_mesh -> <video_id>_3dod_mesh.ply"""
        raise NotImplementedError

    def has_faro_depth(self) -> bool:
        """True if highres_depth/ exists and is non-empty. Scene selection uses this."""
        d = self.root / "highres_depth"
        return d.is_dir() and any(d.iterdir())
