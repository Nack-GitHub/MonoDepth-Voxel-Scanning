"""Per-frame depth quality vs GT depth. Diagnostic: tells whether a bad mesh comes
from bad depth or from the fusion. Standard NYU/KITTI definitions."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Metrics2D:
    rmse: float          # metres
    abs_rel: float
    delta1: float        # fraction with max(d/gt, gt/d) < 1.25
    delta2: float        # < 1.25^2
    delta3: float        # < 1.25^3
    n_valid: int


def depth_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1,
                  max_depth: float = 5.0) -> Metrics2D:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    m = (np.isfinite(pred) & np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth)
         & (pred >= min_depth) & (pred <= max_depth))
    n = int(m.sum())
    if n == 0:
        return Metrics2D(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return Metrics2D(
        rmse=float(np.sqrt(np.mean((p - g) ** 2))),
        abs_rel=float(np.mean(np.abs(p - g) / g)),
        delta1=float(np.mean(ratio < 1.25)),
        delta2=float(np.mean(ratio < 1.25 ** 2)),
        delta3=float(np.mean(ratio < 1.25 ** 3)),
        n_valid=n,
    )


def mean_metrics(items: list[Metrics2D]) -> Metrics2D:
    """Pixel-weighted mean over frames (frames with no valid pixels are ignored)."""
    items = [i for i in items if i.n_valid > 0]
    if not items:
        return Metrics2D(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    w = np.array([i.n_valid for i in items], dtype=np.float64)
    out = {}
    for f in fields(Metrics2D):
        if f.name == "n_valid":
            continue
        v = np.array([getattr(i, f.name) for i in items])
        out[f.name] = float((v * w).sum() / w.sum())
    return Metrics2D(n_valid=int(w.sum()), **out)
