"""SceneDataset — the only thing the pipeline knows about where frames come from.

ScanNet today; Replica / 7-Scenes as fallbacks if ScanNet access is late;
a COLMAP/ARKit capture of a real room for the MVP. Same interface for all (ADR-003).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from roomscan.types import Frame, Intrinsics


class SceneDataset(ABC):
    scene_id: str

    @property
    @abstractmethod
    def intrinsics(self) -> Intrinsics:
        """Intrinsics at the resolution `Frame.rgb` / `Frame.gt_depth` are returned in."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def frame(self, idx: int) -> Frame:
        """Load one frame. May return a frame with a non-finite pose; callers use `frames()`."""

    def frames(self, stride: int = 1, max_frames: int | None = None) -> Iterator[Frame]:
        """Iterate valid frames only (skips tracking-lost poses). Every stage consumes this."""
        yielded = 0
        for idx in range(0, len(self), stride):
            f = self.frame(idx)
            if not f.has_valid_pose:
                continue
            yield f
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return

    @abstractmethod
    def gt_mesh(self):
        """Ground-truth mesh as open3d.geometry.TriangleMesh, or None if the dataset has none."""
