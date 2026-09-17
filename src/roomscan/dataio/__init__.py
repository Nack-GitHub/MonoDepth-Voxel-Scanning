from __future__ import annotations

from roomscan.dataio.base import SceneDataset

_REGISTRY: dict[str, str] = {
    "arkitscenes": "roomscan.dataio.arkitscenes:ARKitScenesScene",   # primary (ADR-009)
    "scannet":     "roomscan.dataio.scannet:ScanNetScene",           # fallback
    # "replica": "roomscan.dataio.replica:ReplicaScene",             # synthetic fallback
    "custom":      "roomscan.dataio.custom:CustomCaptureScene",     # MVP: our own app's capture folder
}


def build_dataset(cfg) -> SceneDataset:
    """cfg = the `dataset:` block. Dataset-specific knobs go under `dataset.options`."""
    from roomscan._registry import load_class

    cls = load_class(_REGISTRY, cfg.name, kind="dataset")
    options = dict(cfg.get("options") or {})
    return cls(root=cfg.root, scene_id=str(cfg.scene), **options)


__all__ = ["SceneDataset", "build_dataset"]
