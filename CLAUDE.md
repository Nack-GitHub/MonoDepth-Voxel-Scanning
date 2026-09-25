# roomscan — project notes for Claude Code

- Architecture and every design decision: `docs/architecture/README.md` + `adr-*.md`. Read §3 invariants and §4 dependency rule before editing `src/`.
- Dataset is **ARKitScenes** (ADR-009), not ScanNet. Faro `highres_depth` = GT, ARKit `lowres_depth` = the `lidar` depth source.
- Phase order is strict: do not implement `models/` or `depth_sources/monocular.py` until `make run-gt` produces a sane mesh (Phase 1 gate). (Phases 0–6 are done as of 2026-09-17.)
- `pipeline.py` must stay free of Open3D/torch/file-format calls; stages own those.
- New depth source / model / aligner / dataset = new file + registry line. If it requires editing `pipeline.py`, stop and write an ADR.
- Results are files: `experiments/results/<exp>/<scene>_<run>/{config.yaml,metrics.json}` are committed; meshes are not.
- Never commit anything under `data/`. Never push unless asked.
- Tests: `make test` — must run without real data (they generate a synthetic scene) and without torch.
- Synthetic scene (`make synthetic`, `data/synthetic/`) is for wiring checks only; never report its numbers as results.
- Dev env is `./.venv` (Python 3.12, open3d 0.19, torch 2.14 MPS OK, fastapi for `roomscan_web`); if missing, `make setup`.
- Long sweeps (mono on 6 scenes ≈ 30 min) must be launched detached (`nohup ... &`); the Bash tool's 10-min timeout kills background jobs, and never run two sweeps on the same experiment concurrently (they race on `metrics.json`).
- `sparse_points` numbers are a LiDAR-sampled VIO proxy (ADR-011) — always label them as an upper bound of real VIO.
- Fine-tuning (ADR-013, supersedes ADR-008 "pretrained-only"): `src/roomscan/training/`, `make train RUN=<yaml>`, env `pip install -e ".[train,dev]"` on WSL2/CUDA. Weights (`experiments/training/*/best|last`) never committed — HF Hub private. The 6 test scenes of Exp1 are never used for training/val/checkpoint choice; splits are visit-disjoint (`configs/training/splits.yaml`).
- Paper's main question (ADR-014): R3 (`ft_lidar_all`, LiDAR teacher) vs pretrained DA-v2 Metric, measured against Faro. 2D metrics are protocol 2 (mask by GT, clip pred; `protocol_2d` in every metrics.json); `roomscan reeval2d` recomputes them in place, `make eval-exp7` = 2D-only on 20 Validation-fold rooms.
- Paper: Thai draft `paper/draft/`, English IEEE manuscript `paper/latex/` (`make paper` needs tectonic), `refs.bib` there; keep both in sync when numbers change.
