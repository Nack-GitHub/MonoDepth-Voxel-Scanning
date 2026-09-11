"""ScaleAligner — turns relative depth into metric depth (ADR-002).

    depth_metric ~= s * depth_pred + t

Strategies (Exp1 rows):
  identity      no-op; for GT and metric models
  oracle_frame  fit (s, t) per frame against GT  -> model's ceiling, not deployable
  per_scene     fit once on the first N frames, freeze -> realistic, shows drift
  sparse_points fit against sparse SLAM points (ARKit/COLMAP) -> the MVP path

The gap between oracle_frame and per_scene is "the price of not knowing scale".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from roomscan.types import Frame


def fit_scale_shift(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None
                    ) -> tuple[float, float]:
    """Closed-form least squares for ref ~= s * pred + t over valid pixels.

    TODO(Phase 2): mask = finite(pred) & finite(ref) & ref>0 & pred>0 (& mask);
    solve via np.linalg.lstsq on [pred, 1]. Consider RANSAC/median-based variant
    if outliers at depth discontinuities dominate.
    """
    raise NotImplementedError("Phase 2")


class ScaleAligner(ABC):
    name: str

    def fit(self, frames: list[Frame], preds: list[np.ndarray]) -> None:  # noqa: B027
        """Optional one-off calibration (per_scene / sparse_points). Default: no-op."""

    @abstractmethod
    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        """Relative -> metric depth for one frame."""


class IdentityAligner(ScaleAligner):
    name = "identity"

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        return pred


class OraclePerFrameAligner(ScaleAligner):
    name = "oracle_frame"

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        s, t = fit_scale_shift(pred, frame.gt_depth)
        return (s * pred + t).astype(np.float32)


class PerSceneAligner(ScaleAligner):
    name = "per_scene"

    def __init__(self, fit_frames: int = 10):
        self.fit_frames = fit_frames
        self.s: float | None = None
        self.t: float | None = None

    def fit(self, frames: list[Frame], preds: list[np.ndarray]) -> None:
        # TODO(Phase 2): stack the first `fit_frames` (pred, gt) pairs, one lstsq
        raise NotImplementedError("Phase 2")

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        if self.s is None:
            raise RuntimeError("PerSceneAligner.fit() must run before align()")
        return (self.s * pred + self.t).astype(np.float32)


class SparsePointsAligner(ScaleAligner):
    """MVP path: uses frame.extra['sparse_points'] (N, 3) from SLAM instead of GT."""

    name = "sparse_points"

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        raise NotImplementedError("MVP: needs COLMAP/ARKit sparse points")


def build_aligner(cfg) -> ScaleAligner:
    """cfg = the `depth:` block."""
    table = {
        "identity": lambda: IdentityAligner(),
        "oracle_frame": lambda: OraclePerFrameAligner(),
        "per_scene": lambda: PerSceneAligner(fit_frames=int(cfg.get("aligner_fit_frames", 10))),
        "sparse_points": lambda: SparsePointsAligner(),
    }
    try:
        return table[cfg.aligner]()
    except KeyError:
        raise ValueError(f"Unknown depth.aligner '{cfg.aligner}'. Known: {sorted(table)}") from None
