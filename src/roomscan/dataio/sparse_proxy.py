"""Sparse metric points as a proxy for a VIO/SLAM point cloud (ADR-011).

ARKitScenes ships no ARKit feature points, so the `sparse_points` aligner is
evaluated against a proxy: `n` pixels sampled from the LiDAR frame (high
confidence only), which is what a phone's tracker would hand the pipeline —
a few hundred metric depths per frame, nothing dense. The sample is
deterministic per frame so every run sees the same points.

This is optimistic in one way (LiDAR points are cleaner than triangulated
VIO features; `noise` lets you degrade them) and pessimistic in another
(VIO points sit on texture, ours are uniform over the LiDAR image).
"""

from __future__ import annotations

import numpy as np


def sample_sparse_depth(depth: np.ndarray, confidence: np.ndarray | None, n: int, *,
                        min_confidence: int = 2, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """-> (H, W) float32 with `n` metric depths kept and 0 everywhere else."""
    valid = np.isfinite(depth) & (depth > 0)
    if confidence is not None and min_confidence > 0:
        valid &= confidence >= min_confidence
    idx = np.flatnonzero(valid)
    out = np.zeros(depth.shape, dtype=np.float32)
    if idx.size == 0 or n <= 0:
        return out
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    vals = depth.ravel()[pick].astype(np.float32)
    if noise > 0:
        vals = vals * (1.0 + rng.normal(0.0, noise, vals.shape)).astype(np.float32)
    out.ravel()[pick] = np.maximum(vals, 1e-3)
    return out
