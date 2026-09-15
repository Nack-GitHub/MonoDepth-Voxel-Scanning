"""Phase 4 diagnostic: WHY does per_scene lose to oracle_frame, and why is the
"metric" model off? Frame-level view of the scale problem, no fusion involved.

For every k-th frame of a scene it runs a mono model and reports, per frame:
  * the oracle (s, t) fitted in inverse-depth (ADR-010)  -> how much the model's
    affine parameters move from view to view
  * abs_rel under oracle / per_scene (10 spread fit frames, same rule as the
    pipeline) / scale-only (median gt/pred, no shift)
  * gt median depth, so close-ups can be separated from room views

    python scripts/analyze_scale_drift.py --stride 5 --out paper/analysis/42444474_scale_drift.json
    python scripts/analyze_scale_drift.py --model depth_anything_v2_metric_indoor

Reading it: if oracle abs_rel is small but per_scene is large, the model is fine and
the *per-view* scale is the problem (a single scene-wide (s, t) cannot fix it).
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from roomscan.config import load_config
from roomscan.dataio import build_dataset
from roomscan.depth_sources import build_depth_source
from roomscan.evaluation.metrics_2d import depth_metrics
from roomscan.geometry.scale_align import _inv, fit_scale_shift, fit_scale_shift_stacked


def _grid(d: np.ndarray, h: int, w: int) -> np.ndarray:
    return d if d.shape == (h, w) else cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)


def _apply_inv(pred: np.ndarray, s: float, t: float) -> np.ndarray:
    d = _inv((s * _inv(pred) + t).astype(np.float32))
    d[~np.isfinite(d) | (d <= 0)] = 0.0
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/depth/mono_scene.yaml")
    ap.add_argument("--model", default=None, help="override depth.model")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fit_frames", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="json with per-frame rows")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.model:
        cfg.depth.model = a.model
    ds = build_dataset(cfg.dataset)
    intr = ds.intrinsics
    src = build_depth_source(cfg.depth, device=a.device)
    frames = [f for f in ds.frames(stride=1) if f.gt_depth is not None]
    print(f"{ds.scene_id}: {len(frames)} frames with GT, model {src.name}")

    # per_scene (S, T): same selection rule as ReconstructionPipeline
    k = max(1, len(frames) // a.fit_frames)
    fit = frames[::k][: a.fit_frames]
    fit_preds = [_grid(src.get_depth(f), intr.height, intr.width) for f in fit]
    S, T = fit_scale_shift_stacked([_inv(p) for p in fit_preds], [_inv(f.gt_depth) for f in fit])

    rows = []
    for f in frames[:: a.stride]:
        p = _grid(src.get_depth(f), intr.height, intr.width)
        g = f.gt_depth
        m = (g > 0) & (p > 0)
        if m.sum() < 100:
            continue
        s, t = fit_scale_shift(_inv(p), _inv(g))
        ratio = float(np.median(g[m] / p[m]))
        rows.append({
            "timestamp": float(f.extra.get("timestamp", 0.0)),
            "oracle_s": s, "oracle_t": t, "gt_over_pred_median": ratio,
            "gt_median": float(np.median(g[m])), "gt_p90": float(np.percentile(g[m], 90)),
            "abs_rel_raw": depth_metrics(p, g).abs_rel if src.is_metric else None,
            "abs_rel_oracle": depth_metrics(_apply_inv(p, s, t), g).abs_rel,
            "abs_rel_per_scene": depth_metrics(_apply_inv(p, S, T), g).abs_rel,
            "abs_rel_scale_only": depth_metrics(np.where(p > 0, p * ratio, 0).astype(np.float32), g).abs_rel,
        })

    def col(key):
        return np.array([r[key] for r in rows if r[key] is not None], dtype=float)

    print(f"per_scene (S, T) = ({S:.4g}, {T:.4g}) from {len(fit)} frames")
    for key in ("oracle_s", "oracle_t", "gt_over_pred_median"):
        v = col(key)
        print(f"  {key:22s} median {np.median(v):.4g}  IQR {np.percentile(v, 25):.4g}..{np.percentile(v, 75):.4g}"
              f"  range {v.min():.4g}..{v.max():.4g}")
    for key in ("abs_rel_raw", "abs_rel_oracle", "abs_rel_per_scene", "abs_rel_scale_only"):
        v = col(key)
        if v.size:
            print(f"  {key:22s} mean {v.mean():.3f}  median {np.median(v):.3f}"
                  f"  p90 {np.percentile(v, 90):.3f}  max {v.max():.3f}")
    gm, ps = col("gt_median"), col("abs_rel_per_scene")
    near = gm < 1.0
    near_err = ps[near].mean() if near.any() else float("nan")
    print(f"  close-ups (gt median < 1 m): {near.sum()}/{len(gm)} frames,"
          f" per_scene abs_rel {near_err:.3f} vs {ps[~near].mean():.3f} on the rest")

    if a.out:
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"scene": ds.scene_id, "model": src.name, "stride": a.stride,
                                           "per_scene": {"s": S, "t": T, "fit_frames": len(fit)},
                                           "rows": rows}, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
