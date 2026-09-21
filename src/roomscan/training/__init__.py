"""Fine-tuning DA-v2 metric on ARKitScenes with a Faro or LiDAR teacher (ADR-013).

Dependency rule: this package imports only dataio, evaluation.metrics_2d, types and config —
never pipeline, geometry, depth_sources or open3d. Torch-only modules (dataset, loss, train,
validate, push) import torch at module level; pairs and transforms are numpy-only.
"""
