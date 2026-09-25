"""Per-frame depth quality vs GT depth. Diagnostic: tells whether a bad mesh comes
from bad depth or from the fusion. Standard NYU/KITTI definitions.

Protocol 2 (ADR-014): the pixel set is chosen by the GT alone (min_depth <= gt <= max_depth) and by
whether the source produced a value at all (pred > 0 and finite; 0 = "no value", e.g. a LiDAR pixel
dropped by confidence). The prediction is then CLIPPED to [min_depth, max_depth], never masked by it.
Protocol 1 also dropped pixels whose *prediction* fell outside the range, which hid exactly the
largest errors of a model that over-estimates scale and let each row be scored on its own pixel set.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

PROTOCOL = 2   # bump when the definition changes; carried into every metrics.json via Metrics2D


@dataclass
class Metrics2D:
    rmse: float          # metres
    abs_rel: float
    delta1: float        # fraction with max(d/gt, gt/d) < 1.25
    delta2: float        # < 1.25^2
    delta3: float        # < 1.25^3
    n_valid: int
    protocol_2d: int = PROTOCOL


def depth_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1,
                  max_depth: float = 5.0) -> Metrics2D:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    m = (np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth)
         & np.isfinite(pred) & (pred > 0))
    n = int(m.sum())
    if n == 0:
        return Metrics2D(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    p = np.clip(pred[m].astype(np.float64), min_depth, max_depth)
    g = gt[m].astype(np.float64)
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
        if f.name in ("n_valid", "protocol_2d"):
            continue
        v = np.array([getattr(i, f.name) for i in items])
        out[f.name] = float((v * w).sum() / w.sum())
    return Metrics2D(n_valid=int(w.sum()), **out)
