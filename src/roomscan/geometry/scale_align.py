"""ScaleAligner — turns relative depth into metric depth (ADR-002).

    depth_metric ~= s * depth_pred + t

Strategies (Exp1 rows):
  identity      no-op; for GT / LiDAR / metric models
  oracle_frame  fit (s, t) per frame against GT  -> model's ceiling, not deployable
  per_scene     fit once on the first N frames, freeze -> realistic, shows drift
  sparse_points fit against sparse SLAM points (ARKit/COLMAP) -> the MVP path

The gap between oracle_frame and per_scene is "the price of not knowing scale".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from roomscan.types import Frame


def _valid(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    m = np.isfinite(pred) & np.isfinite(ref) & (pred > 0) & (ref > 0)
    if mask is not None:
        m &= mask.astype(bool)
    return m


def fit_scale_shift(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None,
                    robust_iters: int = 3, trim: float = 0.2, min_points: int = 100) -> tuple[float, float]:
    """Least squares for ref ~= s * pred + t over valid pixels, with iterative
    trimming of the worst `trim` fraction of residuals (depth discontinuities are
    heavy-tailed; plain LS gets dragged by them)."""
    if pred.shape != ref.shape:
        raise ValueError("pred/ref shape mismatch")
    m = _valid(pred, ref, mask)
    if m.sum() < min_points:
        raise ValueError(f"too few valid pixels to fit scale/shift ({int(m.sum())} < {min_points})")
    x, y = pred[m].astype(np.float64).ravel(), ref[m].astype(np.float64).ravel()
    keep = np.ones_like(x, dtype=bool)
    s = t = 0.0
    for _ in range(max(1, robust_iters)):
        A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
        (s, t), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = np.abs(s * x + t - y)
        keep = r <= np.quantile(r, 1.0 - trim)
    return float(s), float(t)


def fit_scale_shift_stacked(preds: list[np.ndarray], refs: list[np.ndarray]) -> tuple[float, float]:
    """One (s, t) for many frames: subsample each frame so no frame dominates."""
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for p, r in zip(preds, refs, strict=True):
        m = _valid(p, r, None)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        idx = rng.choice(idx, size=min(idx.size, 20_000), replace=False)
        xs.append(p.ravel()[idx])
        ys.append(r.ravel()[idx])
    if not xs:
        raise ValueError("no valid pixels in fit frames")
    return fit_scale_shift(np.concatenate(xs), np.concatenate(ys))


class ScaleAligner(ABC):
    name: str

    def fit(self, frames: list[Frame], preds: list[np.ndarray]) -> None:  # noqa: B027
        """Optional one-off calibration (per_scene / sparse_points). Default: no-op."""

    @abstractmethod
    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        """Relative -> metric depth for one frame (float32, invalid stays 0)."""


def _apply(pred: np.ndarray, s: float, t: float) -> np.ndarray:
    out = (s * pred + t).astype(np.float32)
    out[~np.isfinite(pred) | (pred <= 0) | (out <= 0)] = 0.0
    return out


def _inv(x: np.ndarray) -> np.ndarray:
    """1/x with invalid (<=0, non-finite) mapped to 0 so it stays invalid downstream."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(x) & (x > 0), 1.0 / x, 0.0)
    return out.astype(np.float32)


class _AffineAligner(ScaleAligner):
    """Shared machinery: fit/apply an affine map in `space` = 'depth' or 'inverse'.

    Affine-invariant models (Depth Anything, MiDaS) are affine in DISPARITY, so the
    right model is  gt ≈ 1 / (s * (1/pred) + t)  — `space='inverse'`. Fitting in
    depth space is wrong for those models and only right for near-metric ones.
    """

    def __init__(self, space: str = "inverse"):
        if space not in ("depth", "inverse"):
            raise ValueError("space must be 'depth' or 'inverse'")
        self.space = space

    def _xy(self, pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (_inv(pred), _inv(ref)) if self.space == "inverse" else (pred, ref)

    def _map(self, pred: np.ndarray, s: float, t: float) -> np.ndarray:
        if self.space == "inverse":
            return _inv(_apply(_inv(pred), s, t))
        return _apply(pred, s, t)


class IdentityAligner(ScaleAligner):
    name = "identity"

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        return pred


class OraclePerFrameAligner(_AffineAligner):
    name = "oracle_frame"

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        if frame.gt_depth is None:
            raise ValueError("oracle_frame needs frame.gt_depth")
        x, y = self._xy(pred, _match(frame.gt_depth, pred))
        s, t = fit_scale_shift(x, y)
        return self._map(pred, s, t)


class PerSceneAligner(_AffineAligner):
    """One (s, t) for the whole scan, fitted on `fit_frames` frames spread evenly over it
    (the pipeline picks them). Mimics a one-time calibration against a few known points."""

    name = "per_scene"

    def __init__(self, fit_frames: int = 10, space: str = "inverse"):
        super().__init__(space)
        self.fit_frames = fit_frames
        self.s: float | None = None
        self.t: float | None = None

    def fit(self, frames: list[Frame], preds: list[np.ndarray]) -> None:
        pairs = [self._xy(p, _match(f.gt_depth, p)) for f, p in zip(frames, preds, strict=True)
                 if f.gt_depth is not None]
        if not pairs:
            raise ValueError("per_scene needs gt_depth on the fit frames")
        self.s, self.t = fit_scale_shift_stacked([x for x, _ in pairs], [y for _, y in pairs])

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        if self.s is None:
            raise RuntimeError("PerSceneAligner.fit() must run before align()")
        return self._map(pred, self.s, self.t)


class SparsePointsAligner(_AffineAligner):
    """MVP path: per-frame fit against frame.extra['sparse_depth'] (H,W, 0 = no point) — a few
    hundred metric points from the device tracker (ADR-011 uses a LiDAR-sampled proxy).
    Frames with fewer than `min_points` usable points reuse the previous frame's (s, t), which is
    what a deployed system would do when tracking momentarily loses features."""

    name = "sparse_points"

    def __init__(self, space: str = "inverse", min_points: int = 20, trim: float = 0.1):
        super().__init__(space)
        self.min_points = int(min_points)
        self.trim = float(trim)
        self._last: tuple[float, float] | None = None
        self.n_fallback = 0

    def align(self, pred: np.ndarray, frame: Frame) -> np.ndarray:
        sparse = frame.extra.get("sparse_depth")
        if sparse is None:
            raise ValueError("sparse_points needs frame.extra['sparse_depth']")
        x, y = self._xy(pred, _match(sparse, pred))
        try:
            s, t = fit_scale_shift(x, y, robust_iters=2, trim=self.trim, min_points=self.min_points)
            self._last = (s, t)
        except ValueError:
            if self._last is None:
                raise
            self.n_fallback += 1
            s, t = self._last
        return self._map(pred, s, t)


def _match(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Resample ref (nearest) onto pred's grid if they differ."""
    if ref.shape == pred.shape:
        return ref
    import cv2

    return cv2.resize(ref, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)


def build_aligner(cfg) -> ScaleAligner:
    space = str(cfg.get("align_space", "inverse"))
    table = {
        "identity": lambda: IdentityAligner(),
        "oracle_frame": lambda: OraclePerFrameAligner(space),
        "per_scene": lambda: PerSceneAligner(int(cfg.get("aligner_fit_frames", 10)), space),
        "sparse_points": lambda: SparsePointsAligner(space, int(cfg.get("sparse_min_points", 20))),
    }
    try:
        return table[cfg.aligner]()
    except KeyError:
        raise ValueError(f"Unknown depth.aligner '{cfg.aligner}'. Known: {sorted(table)}") from None
