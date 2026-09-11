from __future__ import annotations

from roomscan.models.base import DepthModel

# name used in config  ->  "module:Class" + constructor kwargs
_REGISTRY: dict[str, tuple[str, dict]] = {
    "depth_anything_v2_small":         ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "small"}),
    "depth_anything_v2_base":          ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "base"}),
    "depth_anything_v2_large":         ("roomscan.models.depth_anything:DepthAnythingV2", {"size": "large"}),
    "depth_anything_v2_metric_indoor": ("roomscan.models.depth_anything:DepthAnythingV2Metric", {}),
    "midas_small":                     ("roomscan.models.midas:MiDaS", {"variant": "MiDaS_small"}),
    # "mobilevit_depth": (...)   # Project 1 bridge
}


def build_depth_model(name: str, device: str = "cpu") -> DepthModel:
    from roomscan._registry import load_class

    paths = {k: v[0] for k, v in _REGISTRY.items()}
    cls = load_class(paths, name, kind="depth model")
    return cls(device=device, **_REGISTRY[name][1])


__all__ = ["DepthModel", "build_depth_model"]
