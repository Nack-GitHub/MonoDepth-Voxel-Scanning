# Depth Pro on ARKitScenes: the post-processing does not give metric depth (2026-09-23)

Status: **blocked, no numbers reported.** `src/roomscan/models/depth_pro.py` exists and runs, the Exp6
row is commented out in `configs/experiments/exp6_finetune.yaml`.

## What we wanted

A Level-0 baseline for ADR-013: a metric model that is conditioned on (or predicts) the camera
intrinsics, so the paper can answer "why not just use Depth Pro / UniDepth / Metric3D?".

## What we measured

`apple/DepthPro-hf`, transformers 5.17, the call from the library's own docstring
(`processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])`), ARKitScenes test scenes,
Faro `highres_depth` as the reference, all at the loader's 256×192 grid:

| frame (47115299, sky=Up, no rotation) | measurement |
|---|---|
| post-processed depth vs Faro | median pred/gt **2.5–4.6×**, `corr(1/pred, 1/gt) = -0.95`, `corr(pred, gt) = -0.50` |
| model's raw `predicted_depth` vs Faro | `corr(raw, gt) = +0.92` (one frame), `+0.19` (another) |
| after per-frame median scaling | abs_rel 0.24–0.54 (DA-v2 Metric on the same frames: 0.04–0.13) |
| predicted FOV | 48.0–48.2° → focal 716 px, while the camera's real focal is **533 px** |

Passing the true focal (the wrapper's `focal_px`) scales depth by exactly 533/716 = 0.74 as expected,
which reduces but does not remove the error (median pred/gt 1.9 instead of 2.6).

## Reading

The library's post-processing computes `depth = 1 / (raw · width / focal)`, i.e. it treats
`predicted_depth` as *inverse* depth. The sign of the correlations says the checkpoint's output behaves
like depth, so the inversion turns near into far. Pre-processing is not the problem (1536², [-1, 1],
mean/std 0.5 — as documented).

## Before this row goes into the paper

1. Compare one frame against Apple's reference implementation (`apple/ml-depth-pro`) to fix the
   convention — that is the ground truth for which quantity `predicted_depth` holds.
2. If the library is wrong, invert in `DepthPro.predict` and re-check `corr(pred, gt) > 0` plus
   `median(pred/gt) ≈ 1` on a few frames of both a `sky=Up` and a `sky=Left` scene.
3. Only then run `sweep configs/experiments/exp6_finetune.yaml --skip-existing` for that row.

Do not report a "Depth Pro is N cm" number from the current code path: it would be a wrapper artefact,
not a property of the model.
