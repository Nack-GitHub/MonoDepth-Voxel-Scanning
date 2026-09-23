# Does a LiDAR teacher replace a laser scanner? (Exp6, ADR-013)

Six held-out ARKitScenes scenes, one pipeline, one variable: **which metric model predicts depth**.
Every fine-tuned row is RGB-only at test time with `depth.aligner: identity` — no scale is ever fitted,
so these rows are what a phone *without* a depth sensor could actually run.
Training used 10 (R1, R2) or 24 (R3) other scenes, split by `visit_id`; the six scenes below were
excluded from training, validation and checkpoint selection (`configs/training/splits.yaml`).

## Headline (mean ± sd over the six scenes)

| row | teacher | Chamfer ↓ | F@5cm ↑ | AbsRel ↓ | δ1 ↑ | needs a sensor at test time? |
|---|---|---|---|---|---|---|
| `gt` (Faro) | — | 2.06 ± 0.54 cm | 0.97 | 0.000 | 1.000 | laser scanner |
| `lidar` (ARKit) | — | 3.04 ± 0.57 cm | 0.93 | 0.018 | 0.997 | iPhone/iPad Pro LiDAR |
| `mono_oracle` | — | 5.33 ± 1.33 cm | 0.79 | 0.032 | 0.986 | per-frame GT scale (upper bound) |
| `mono_sparse` | — | 5.96 ± 1.68 cm | 0.74 | 0.036 | 0.985 | ~200 sparse metric points (VIO proxy) |
| `mono_metric` (pretrained) | — | 54.27 ± 10.59 cm | 0.03 | 0.382 | 0.277 | **no** |
| `mono_ft_faro` (R1) | Faro, 10 scenes | **14.59 ± 3.96 cm** | **0.31** | 0.136 | 0.780 | **no** |
| `mono_ft_lidar` (R2) | ARKit LiDAR, 10 scenes | 16.55 ± 4.65 cm | 0.23 | 0.136 | **0.848** | **no** |
| `mono_ft_lidar_all` (R3) | ARKit LiDAR, 24 scenes | 15.68 ± 4.04 cm | 0.24 | **0.134** | 0.812 | **no** |

## What it says

1. **Fine-tuning is what makes a metric model usable at all.** Pretrained DA-v2 Metric-Indoor is
   54 cm off; the same architecture fine-tuned on ARKitScenes is 15–17 cm — a **3.3×** cut in Chamfer
   and AbsRel 0.382 → 0.134, with no alignment anywhere in the pipeline.
2. **The LiDAR teacher costs almost nothing.** R2 / R1 = **1.13×** in Chamfer, and in 2D the two are
   indistinguishable (AbsRel 0.136 vs 0.136; δ1 0.848 vs 0.780 *favours* the LiDAR teacher). A phone's
   own depth sensor is a good enough teacher for the phones that lack one — which is the point: the
   labels can be collected by anyone with an iPhone Pro instead of a $20k laser scanner.
3. **More LiDAR scenes ≈ the same.** R3 doubled the scene count (24 vs 10) and moved Chamfer 16.6 → 15.7
   and AbsRel 0.136 → 0.134. R3 was the *best* run on the validation fold (0.133 vs 0.142) but is not
   clearly better here, so the val-selected ranking did not transfer; report it as a null result.
4. **Alignment still beats fine-tuning.** `mono_oracle` (5.3 cm) and the deployable `mono_sparse`
   (6.0 cm) remain ~11 cm better than the best fine-tune. Fine-tuning removes the *need* for metric
   anchors; it does not match having them. The honest framing for the paper: three regimes —
   sensor (3 cm), monocular + metric anchors (6 cm), monocular alone (15 cm) — and fine-tuning moves
   the last one from "unusable" (54 cm, F@5cm 0.03) to "a coarse but correctly-scaled room".
5. **2D and 3D disagree on the teacher.** R1 wins on Chamfer/F@5cm, R2 wins on δ1 and per-scene
   AbsRel. Per-scene Chamfer (`experiments/results/exp6_finetune/per_scene.md`) shows why: R2 wins on only 2 of the
   6 scenes (42897743, 47115299) and is much worse on 45261556 (24.6 vs 11.9 cm). With six scenes and this spread, the
   teacher difference is within scene-to-scene variance — claim "no significant loss", not "as good".

## Caveat recorded for the write-up

The fine-tunes were selected on the ARKitScenes Validation fold (3 scenes, Faro), which ranked
R3 > R2 > R1; the test fold ranks R1 ≥ R3 ≥ R2 in 3D. Three validation scenes are too few to pick
between checkpoints that differ by ~0.01 AbsRel, and the paper should say so rather than quietly
using the test fold to choose.

## Related: the upright-rotation fix (PR #1)

The `sky_direction → np.rot90` table was wrong for `Left`/`Right`, so four of the six test scenes were
fed to every monocular model upside-down. Every `mono_*` number in Exp1 was re-run after the fix
(the two `Up`/`Down` scenes are unchanged, which is the check that the fix is what moved them):

| run | Chamfer before | after | F@5cm before | after |
|---|---|---|---|---|
| `mono_oracle` | 6.5 cm | **5.3 cm** | 0.74 | **0.79** |
| `mono_sparse` | 7.2 cm | **6.0 cm** | 0.69 | **0.74** |
| `mono_scene` | 22.7 cm | 22.9 cm | 0.22 | 0.23 |
| `mono_metric` | 49.0 cm | 54.3 cm | 0.06 | 0.03 |

Correcting the orientation helps the rows whose scale is fitted (oracle −1.2 cm, sparse −1.2 cm) and
*hurts* the pretrained metric row (+5.3 cm): its absolute scale error is erratic rather than systematic,
so an upside-down image was not consistently worse. `paper/analysis/42444474_failure_analysis.md` and the
scale-drift JSONs were produced before the fix — re-run and re-read before quoting them.
