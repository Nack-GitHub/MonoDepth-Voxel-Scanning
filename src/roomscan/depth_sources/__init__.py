from __future__ import annotations

from roomscan.depth_sources.base import DepthSource


def build_depth_source(cfg, device: str = "cpu") -> DepthSource:
    """cfg = the `depth:` block. Mono needs `cfg.model`; everything else is arg-less."""
    if cfg.source == "gt":
        from roomscan.depth_sources.ground_truth import GroundTruthDepth

        return GroundTruthDepth()
    if cfg.source == "mono":
        from roomscan.depth_sources.monocular import MonocularDepth
        from roomscan.models import build_depth_model

        if not cfg.model:
            raise ValueError("depth.source=mono requires depth.model")
        return MonocularDepth(build_depth_model(cfg.model, device=device))
    if cfg.source == "arcore":
        from roomscan.depth_sources.arcore import ARCoreDepth

        return ARCoreDepth()
    if cfg.source == "lidar":
        from roomscan.depth_sources.lidar import LiDARDepth

        return LiDARDepth(min_confidence=int(cfg.get("lidar_min_confidence", 0)))
    raise ValueError(f"Unknown depth.source '{cfg.source}' (gt | mono | arcore | lidar)")


__all__ = ["DepthSource", "build_depth_source"]
