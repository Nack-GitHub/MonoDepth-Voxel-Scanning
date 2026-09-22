# Fine-tuning results on V (validation, not test)

All numbers: ARKitScenes **Validation** fold, 3 scenes, every 5th Faro frame (214 frames), `metrics_2d.depth_metrics` (0.1–5 m), pixel-weighted mean. V picked the checkpoints, so these are optimistic; the paper's numbers come from the six held-out Exp1 test scenes (evaluated on the Mac).

| run | teacher | train sets | scenes / frames | steps | pretrained abs_rel | best abs_rel (step) | δ1 @best | RMSE m @best | last abs_rel | best / R1 |
|---|---|---|---|---|---|---|---|---|---|---|
| r1_ft_faro | faro | train_faro | 10 / 3594 | 6000 | 0.461 | **0.145** (2500) | 0.806 | 0.179 | 0.173 | 1.00 |
| r2_ft_lidar | lidar | train_faro | 10 / 8656 | 6000 | 0.461 | **0.142** (2000) | 0.834 | 0.154 | 0.163 | 0.98 |
| r3_ft_lidar_all | lidar | train_faro+train_lidar | 24 / 19896 | 12000 | 0.461 | **0.133** (2000) | 0.858 | 0.151 | 0.166 | 0.92 |

Per V scene, abs_rel at the chosen (best) checkpoint:

| run | 47331068 | 45663164 | 42445028 |
|---|---|---|---|
| pretrained | 0.776 | 0.211 | 0.322 |
| r1_ft_faro | 0.151 | 0.154 | 0.123 |
| r2_ft_lidar | 0.221 | 0.068 | 0.124 |
| r3_ft_lidar_all | 0.186 | 0.087 | 0.117 |

Validation curve (abs_rel every 500 steps):

| step | r1_ft_faro | r2_ft_lidar | r3_ft_lidar_all |
|---|---|---|---|
| 0 | 0.461 | 0.461 | 0.461 |
| 500 | 0.152 | 0.167 | 0.134 |
| 1000 | 0.158 | 0.178 | 0.213 |
| 1500 | 0.149 | 0.146 | 0.175 |
| 2000 | 0.149 | 0.142 | 0.133 |
| 2500 | 0.145 | 0.143 | 0.159 |
| 3000 | 0.154 | 0.159 | 0.154 |
| 3500 | 0.212 | 0.172 | 0.150 |
| 4000 | 0.150 | 0.160 | 0.143 |
| 4500 | 0.164 | 0.162 | 0.144 |
| 5000 | 0.179 | 0.164 | 0.242 |
| 5500 | 0.167 | 0.162 | 0.221 |
| 6000 | 0.173 | 0.163 | 0.218 |
| 6500 |  |  | 0.154 |
| 7000 |  |  | 0.220 |
| 7500 |  |  | 0.183 |
| 8000 |  |  | 0.189 |
| 8500 |  |  | 0.161 |
| 9000 |  |  | 0.163 |
| 9500 |  |  | 0.186 |
| 10000 |  |  | 0.175 |
| 10500 |  |  | 0.172 |
| 11000 |  |  | 0.164 |
| 11500 |  |  | 0.166 |
| 12000 |  |  | 0.166 |

Cost: r1_ft_faro 0.6282 s/step, 3.2 GB reserved, r2_ft_lidar 0.6204 s/step, 3.2 GB reserved, r3_ft_lidar_all 0.6158 s/step, 3.2 GB reserved (RTX 3070 Ti, bf16, batch 2×4).

## Checkpoint load check on the Mac (T15 / SPEC §10.7)

`build_depth_model(...)` pulls each checkpoint from the Hub and predicts through `MonocularDepth`
(upright rotation from PR #1) on **held-out test scenes**, 3 frames each, vs Faro depth (0.1–5 m):

| model | 42444474 (`Left`) median pred/gt · abs_rel | 47115299 (`Up`) median pred/gt · abs_rel |
|---|---|---|
| pretrained metric_indoor | 1.511 · 0.432 | 1.173 · 0.156 |
| ft_faro (R1) | 0.931 · 0.160 | 0.862 · 0.184 |
| ft_lidar (R2) | 1.110 · **0.146** | 0.914 · **0.138** |
| ft_lidar_all (R3) | 0.813 · 0.205 | 0.788 · 0.190 |

All four load and return (H, W) float32 metres; every median ratio is inside the [0.7, 1.4] gate. Two
things to keep in mind for the paper: the LiDAR teacher (R2) beats the Faro teacher (R1) on both test
scenes, and R3 — the best run on V — is the *worst* fine-tune here, i.e. V picked it optimistically.
Six frames are a smoke test, not a result; the numbers that go in the paper come from `exp6_finetune`.
