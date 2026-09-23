# Depth Pro on ARKitScenes: the model, not the wrapper (2026-09-23)

First conclusion in this note was wrong and is corrected below. Keep both, the false lead is the
useful part: a near/far inversion looked obvious from correlations alone on real frames.

## The false lead

On ARKitScenes test frames, the post-processed depth from `apple/DepthPro-hf` (transformers 5.17,
the call from the library's own docstring) gave `corr(1/pred, 1/gt) = -0.95` and `corr(pred, gt) = -0.50`
against Faro, while the model's raw `predicted_depth` appeared to correlate *positively* with depth.
Read as "the library treats the output as inverse depth when it is not", i.e. a convention bug.

## The control that settles it

The synthetic room (`data/synthetic`, exact GT, sharp render, no motion blur):

| frame | corr(raw, 1/gt) | corr(post, gt) | abs_rel of post after one scale |
|---|---|---|---|
| 0 | **+0.993** | **+0.996** | **0.015** |
| 30 | +0.991 | +0.994 | 0.008 |

So `predicted_depth` *is* inverse depth, `post_process_depth_estimation` *does* return depth, and our
wrapper's call is right. The correlations on real frames were low because the prediction there is bad,
not flipped: Pearson on a narrow GT band (0.7--3.4 m) against a prediction compressed into 3.1--6.4 m
is near zero and can take either sign.

## What Depth Pro actually does on this data

Looking at one frame (`scratchpad/dp_panel.png`: RGB | Faro | Depth Pro) the *structure* is right --
the dresser, the box, the ornament and the wall are all where they should be -- and the *distances*
are not: Faro 0.68--3.38 m, Depth Pro 3.10--6.39 m.

| | value |
|---|---|
| median pred/gt, 5 frames of 47115299 | 1.8--4.6x (with the FOV it predicts) |
| the same with the camera's true focal (533 px vs its predicted ~716) | 1.35--3.4x |
| predicted FOV | 48.0--48.7 deg; the true FOV at fx 533 on 640 px is **61.6 deg** |
| abs_rel after a per-frame median scale | 0.24--0.54 (DA-v2 Metric on the same frames: 0.04--0.13) |

Two failures stack: the FOV head is wrong on these frames (a 1.3x scale error on its own), and what is
left after removing all scale is still several times worse than DA-v2 Metric. The shape is not the
problem; the metric interpretation of close indoor VGA frames is.

## Consequence for the paper

This is exactly the Level-0 question ("why not a metric model that knows the camera?") and the answer
is reportable: **conditioning on intrinsics does not rescue metric depth on real phone captures.**
The Exp6 row therefore runs with the camera's true focal length (`depth_pro_intrinsics`), which is the
strongest form of the baseline -- anything weaker would be a straw man. Depth Pro at $1536^2$ costs
~8 s/frame on MPS, so the row is ~5 h for six scenes.
