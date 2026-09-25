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
    al = PerSceneAligner(fit_frames=4, space="depth")
    al.fit(frames, [p for p, _ in pairs])
    out = al.align(pairs[0][0], frames[0])
    m = depth_metrics(out, pairs[0][1])
    # the 5 % injected outliers are scored under protocol 2 (clipped, not dropped) -> ~0.06
    assert m.abs_rel < 0.08 and m.delta1 > 0.9


def test_oracle_aligner_and_invalid_pixels_stay_zero():
    pred, gt = _pair(7)
    pred[:10] = 0.0
    f = Frame(0, np.zeros((1, 1, 3), np.uint8), np.eye(4), gt_depth=gt)
    out = OraclePerFrameAligner(space="depth").align(pred, f)
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
    assert m.protocol_2d == 2


def test_metrics_2d_clips_out_of_range_predictions_instead_of_dropping_them():
    # ADR-014: pixels are chosen by the GT; a prediction beyond max_depth is scored (clipped), not hidden
    gt = np.full((2, 2), 4.0, np.float32)
    pred = np.array([[4.0, 4.0], [4.0, 12.0]], np.float32)
    m = depth_metrics(pred, gt, 0.1, 5.0)
    assert m.n_valid == 4
    assert m.abs_rel == pytest.approx((5.0 - 4.0) / 4.0 / 4)     # 12 m clipped to 5 m
    assert m.delta1 == pytest.approx(0.75)
    # a pixel the source left empty (0 / NaN) is not a prediction and is not scored
    pred[0, 0], pred[0, 1] = 0.0, np.nan
    assert depth_metrics(pred, gt, 0.1, 5.0).n_valid == 2
    # GT outside the range is still excluded
    gt[1, 0] = 7.0
    assert depth_metrics(pred, gt, 0.1, 5.0).n_valid == 1


def test_inverse_space_recovers_disparity_affine():
    rng = np.random.default_rng(3)
    gt = rng.uniform(0.5, 5.0, (192, 256)).astype(np.float32)
    disp = 3.0 / gt + 0.2                       # model output: affine in disparity
    pred = (1.0 / disp).astype(np.float32)      # what MonocularDepth hands the aligner
    f = Frame(0, np.zeros((1, 1, 3), np.uint8), np.eye(4), gt_depth=gt)
    out = OraclePerFrameAligner(space="inverse").align(pred, f)
    assert depth_metrics(out, gt).abs_rel < 0.01
    bad = OraclePerFrameAligner(space="depth").align(pred, f)
    assert depth_metrics(bad, gt).abs_rel > depth_metrics(out, gt).abs_rel


def test_sparse_points_aligner_from_proxy_and_fallback():
    from roomscan.dataio.sparse_proxy import sample_sparse_depth
    from roomscan.geometry.scale_align import SparsePointsAligner

    pred, gt = _pair(11, outliers=0.0)
    conf = np.full(gt.shape, 2, np.uint8)
    sparse = sample_sparse_depth(gt, conf, 200, seed=0)
    assert (sparse > 0).sum() == 200 and np.allclose(sparse[sparse > 0], gt[sparse > 0])
    f = Frame(0, np.zeros((1, 1, 3), np.uint8), np.eye(4), extra={"sparse_depth": sparse})
    al = SparsePointsAligner(space="depth")
    out = al.align(pred, f)
    assert depth_metrics(out, gt).abs_rel < 0.02
    # a frame with too few points reuses the last fit instead of failing
    few = Frame(1, np.zeros((1, 1, 3), np.uint8), np.eye(4),
                extra={"sparse_depth": sample_sparse_depth(gt, conf, 5, seed=1)})
    out2 = al.align(pred, few)
    assert al.n_fallback == 1 and np.allclose(out, out2)
    # low-confidence pixels never become proxy points
    conf[:, :128] = 0
    left_only = sample_sparse_depth(gt, conf, 50, min_confidence=2, seed=2)
    assert (left_only[:, :128] == 0).all() and (left_only > 0).sum() == 50
