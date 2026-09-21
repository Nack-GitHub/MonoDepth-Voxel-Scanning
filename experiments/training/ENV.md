# Training environment (for the paper's Setup / Limitations)

Recorded 2026-09-21 on the training machine. Per-run details (device, versions, frame counts, s/step,
peak VRAM) are also in the first (`env`) and last (`summary`) rows of each `experiments/training/<run>/log.jsonl`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GB (driver 591.86, CUDA 13.1 driver API) |
| OS | Windows 11 + WSL2 Ubuntu 22.04.3 (kernel 6.18.33.2-microsoft-standard-WSL2), 16 vCPU, 7 GB RAM in WSL |
| Python / torch | 3.12.14 (uv) / torch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1) |
| Libraries | transformers 5.17.0, huggingface_hub 1.32.0, open3d 0.19.0 (tests only), numpy 2.5.2 |
| Data | `~/data/arkitscenes` on the Linux filesystem (not `/mnt/c`), 32.7 GB — see `DATA.md` |

## R0 smoke (gate, SPEC §10.4) — passed

Large, frozen backbone, batch 2 × accum 4, bf16 autocast, 518² crops, 1 scene of A, 30 steps.

| | |
|---|---|
| OOM fallback needed | none (batch 2 × 4 as in `base.yaml`) |
| peak VRAM (torch) | 2.7 GB allocated / 3.2 GB reserved — well under the 7.5 GB gate |
| speed | 0.61 s / optimizer step (8 samples) → R1/R2 6k steps ≈ 1 h, R3 12k ≈ 2 h (+ val ≈ 27 s each) |
| loss | 1.97 (step 1) → 0.90 (step 30) |
| val abs_rel on V | 0.461 (pretrained, step 0) → 0.180 (step 30) |

Step-0 abs_rel 0.46 is above the plan's 0.2–0.4 sanity band because of one close-range Validation scene
(47331068, Faro median depth 0.78 m, pretrained over-estimates ×1.6–1.9 in every orientation → abs_rel 0.78);
the other two V scenes are 0.21 / 0.32. The label is sane: median LiDAR/Faro = 0.98 on all three V scenes.

## Upright rotation (found during R0)

The `sky_direction` → `np.rot90` table copied from `depth_sources/monocular.py` turned every `Left` scene
upside-down. Checked against gravity from the poses (world is z-up): with the corrected table
(`Left` = clockwise) all 27 scenes score g ≈ -0.7…-0.9 (`DATA.md`, column "sky (upright g)").
The training code uses the corrected table; the pipeline's `monocular.py` is unchanged here (outside
ADR-013) and 4 of the 6 Exp1 test scenes are `Left` — reported separately.

## Local setup notes (not needed on a normal Ubuntu with sudo)

- `open3d` needs `libEGL.so.1` / `libGL.so.1`. Without root they were unpacked from the Ubuntu `.deb`s
  (`apt-get download libegl1 libgl1 libglvnd0 libglx0 ...` + `dpkg -x`) into `~/.local/syslibs` and
  symlinked next to `open3d/libOpen3D.so` (it loads with `RPATH=$ORIGIN`). With sudo:
  `apt-get install -y libegl1 libgl1 libgomp1`.
- `open3d` 0.20 returns an empty TSDF mesh in `tests/test_custom_capture.py`; `pyproject.toml` pins `<0.20`.
- Long jobs run as `systemd-run --user` units (equivalent to `nohup … &`) so they outlive the shell.
