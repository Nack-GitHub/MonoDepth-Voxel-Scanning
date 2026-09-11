"""ScanNet scene loader — FALLBACK only (ADR-009 chose ARKitScenes).

Kept as a registry entry so switching back is a config change. Layout & pitfalls
if this is ever implemented:
  * depth PNGs uint16 millimetres -> /1000
  * color 1296x968 vs depth 640x480 -> resize color down, use intrinsic_depth.txt
  * pose/*.txt 4x4 camera-to-world, may contain -inf (tracking lost)
  * GT mesh: <scene_id>_vh_clean_2.ply
"""

from __future__ import annotations

from pathlib import Path

from roomscan.dataio.base import SceneDataset
from roomscan.types import Frame, Intrinsics


class ScanNetScene(SceneDataset):
    def __init__(self, root: str | Path, scene_id: str, **options):
        self.root = Path(root) / scene_id
        self.scene_id = scene_id
        raise NotImplementedError("fallback dataset; not implemented (see ADR-009)")

    @property
    def intrinsics(self) -> Intrinsics:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def frame(self, idx: int) -> Frame:
        raise NotImplementedError

    def gt_mesh(self):
        raise NotImplementedError
