from __future__ import annotations

from roomscan.models.base import DepthModel

# config name -> ("module:Class", kwargs)
_REGISTRY: dict[str, tuple[str, dict]] = {
    "depth_anything_v2_small": ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "small"}),
    "depth_anything_v2_base": ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "base"}),
    "depth_anything_v2_large": ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "large"}),
    "depth_anything_v2_metric_indoor": ("roomscan.models.depth_anything:DepthAnythingV2Metric",
                                        {"size": "metric_indoor"}),
    "depth_anything_v2_metric_indoor_small": ("roomscan.models.depth_anything:DepthAnythingV2Metric",
                                              {"size": "metric_indoor_small"}),
    # ARKitScenes fine-tunes of metric_indoor (ADR-013) — private HF repos pushed by roomscan.training.push
    "depth_anything_v2_ft_faro": ("roomscan.models.depth_anything:DepthAnythingV2Metric",
                                  {"size": "ft_faro", "hf_id": "NackPanupong/roomscan-dav2-metric-large-ft-faro"}),
    "depth_anything_v2_ft_lidar": ("roomscan.models.depth_anything:DepthAnythingV2Metric",
                                   {"size": "ft_lidar", "hf_id": "NackPanupong/roomscan-dav2-metric-large-ft-lidar"}),
    "depth_anything_v2_ft_lidar_all": ("roomscan.models.depth_anything:DepthAnythingV2Metric",
                                       {"size": "ft_lidar_all",
                                        "hf_id": "NackPanupong/roomscan-dav2-metric-large-ft-lidar-all"}),
    # Level-0 baseline (ADR-013): a metric model that conditions on / predicts the camera intrinsics
    "depth_pro": ("roomscan.models.depth_pro:DepthPro", {}),                      # predicts its own FOV
    # ARKitScenes vga_wide is one camera: fx = fy = 533 px at 640x480 in all six test scenes
    "depth_pro_intrinsics": ("roomscan.models.depth_pro:DepthPro", {"focal_px": 532.8}),
    "midas_small": ("roomscan.models.midas:MiDaS", {"variant": "MiDaS_small"}),
    "midas_dpt_hybrid": ("roomscan.models.midas:MiDaS", {"variant": "DPT_Hybrid"}),
}


def build_depth_model(name: str, device: str = "auto") -> DepthModel:
    from roomscan._registry import load_class

    cls = load_class({k: v[0] for k, v in _REGISTRY.items()}, name, kind="depth model")
    return cls(device=device, **_REGISTRY[name][1])


__all__ = ["DepthModel", "build_depth_model"]
