"""Scene frame -> (rgb, depth_m, mask) training pair on the 640x480 vga_wide grid.

Reuses ARKitScenesScene unchanged: with fusion_resolution=(640, 480) its _to_fusion_grid resamples
every depth asset with NEAREST (Faro 1920x1440 down, LiDAR 256x192 up), and vga_wide is read at its
native 640x480, so rgb/depth/mask line up pixel for pixel. target="faro" iterates highres_depth
timestamps (~10 fps; vga_wide at 30 fps is always within the 0.02 s match tolerance); target="lidar"
iterates vga_wide timestamps and masks by ARKit confidence (ADR-013).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roomscan.dataio.arkitscenes import ARKitScenesScene

TARGETS = ("faro", "lidar")
TRAIN_RESOLUTION = (640, 480)   # (W, H) of vga_wide
FRAME_SOURCE = {"faro": "highres_depth", "lidar": "vga_wide"}


@dataclass(frozen=True)
class DepthPair:
    rgb: np.ndarray      # (H, W, 3) uint8, as stored (not yet upright)
    depth: np.ndarray    # (H, W) float32 metres, 0 = invalid
    mask: np.ndarray     # (H, W) bool
    sky_direction: str = "Up"


def open_training_scene(root: str | Path, video_id: str, *, fold: str = "Training",
                        target: str = "faro") -> ARKitScenesScene:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    return ARKitScenesScene(root, str(video_id), split=fold, fusion_resolution=TRAIN_RESOLUTION,
                            rgb_asset="vga_wide", frame_source=FRAME_SOURCE[target])


REQUIRED_ASSETS = {"faro": ("highres_depth",), "lidar": ("lowres_depth", "confidence")}


def iter_indices(scene: ARKitScenesScene, stride: int = 1, target: str | None = None) -> list[int]:
    """Every `stride`-th frame; with `target`, only frames whose label assets exist within tolerance
    (LiDAR streams have gaps — a vga_wide frame without lowres_depth cannot be a training pair)."""
    idx = range(0, len(scene), max(1, int(stride)))
    if target is None:
        return list(idx)
    need = REQUIRED_ASSETS[target]
    return [i for i in idx if all(scene.has_asset_at(a, i) for a in need)]


def make_pair(scene: ARKitScenesScene, idx: int, *, target: str, max_depth: float = 10.0,
              lidar_min_confidence: int = 1) -> DepthPair:
    frame = scene.frame(idx)
    if target == "faro":
        depth = frame.gt_depth
        if depth is None:
            raise ValueError(f"{scene.scene_id} frame {idx}: no highres_depth within tolerance")
        mask = depth > 0
    elif target == "lidar":
        depth = frame.extra.get("lidar_depth")
        conf = frame.extra.get("lidar_confidence")
        if depth is None or conf is None:
            raise ValueError(f"{scene.scene_id} frame {idx}: no lowres_depth/confidence within tolerance")
        mask = (depth > 0) & (conf >= lidar_min_confidence)
    else:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    mask &= depth <= max_depth   # metres

    rgb = frame.rgb
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        # loader fell back to another RGB asset (e.g. lowres_wide, same camera): resample RGB only
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return DepthPair(rgb=rgb, depth=np.where(mask, depth, 0.0).astype(np.float32), mask=mask,
                     sky_direction=str(frame.extra.get("sky_direction", "Up")))
