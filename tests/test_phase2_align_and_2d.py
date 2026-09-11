import numpy as np
import pytest

from roomscan.evaluation.metrics_2d import depth_metrics, mean_metrics
from roomscan.geometry.scale_align import (
    OraclePerFrameAligner,
    PerSceneAligner,
    fit_scale_shift,
    fit_scale_shift_stacked,
)
from roomscan.types import Frame


def _pair(seed=0, s=2.0, t=0.5, outliers=0.05):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 5.0, (192, 256)).astype(np.float32)
    pred = ((gt - t) / s).astype(np.float32)
    bad = rng.random(gt.shape) < outliers
    pred[bad] = rng.uniform(0.01, 5.0, bad.sum())      # depth-discontinuity style outliers
    return pred, gt


def test_fit_recovers_scale_shift_despite_outliers():
    pred, gt = _pair()
    s, t = fit_scale_shift(pred, gt)
    assert abs(s - 2.0) < 0.02 and abs(t - 0.5) < 0.02


def test_stacked_fit_and_per_scene_aligner():
    pairs = [_pair(i) for i in range(4)]
    s, t = fit_scale_shift_stacked([p for p, _ in pairs], [g for _, g in pairs])
    assert abs(s - 2.0) < 0.02 and abs(t - 0.5) < 0.02
    frames = [Frame(i, np.zeros((1, 1, 3), np.uint8), np.eye(4), gt_depth=g) for i, (_, g) in enumerate(pairs)]
    al = PerSceneAligner(fit_frames=4)
    al.fit(frames, [p for p, _ in pairs])
    out = al.align(pairs[0][0], frames[0])
    m = depth_metrics(out, pairs[0][1])
    assert m.abs_rel < 0.05 and m.delta1 > 0.9


def test_oracle_aligner_and_invalid_pixels_stay_zero():
    pred, gt = _pair(7)
    pred[:10] = 0.0
    out = OraclePerFrameAligner().align(pred, Frame(0, np.zeros((1, 1, 3), np.uint8), np.eye(4), gt_depth=gt))
    assert (out[:10] == 0).all()
    assert depth_metrics(out, gt).delta1 > 0.9


def test_metrics_2d_identity_and_mean():
    gt = np.full((4, 4), 2.0, np.float32)
    perfect = depth_metrics(gt, gt)
    assert perfect.rmse == 0 and perfect.delta1 == 1.0
    off = depth_metrics(gt * 1.3, gt)
    assert off.delta1 == 0.0 and off.delta2 == 1.0
    m = mean_metrics([perfect, off])
    assert m.delta1 == pytest.approx(0.5) and m.n_valid == 32
