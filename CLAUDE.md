# roomscan — project notes for Claude Code

- Architecture and every design decision: `docs/architecture/README.md` + `adr-*.md`. Read §3 invariants and §4 dependency rule before editing `src/`.
- Dataset is **ARKitScenes** (ADR-009), not ScanNet. Faro `highres_depth` = GT, ARKit `lowres_depth` = the `lidar` depth source.
- Phase order is strict: do not implement `models/` or `depth_sources/monocular.py` until `make run-gt` produces a sane mesh (Phase 1 gate).
- `pipeline.py` must stay free of Open3D/torch/file-format calls; stages own those.
- New depth source / model / aligner / dataset = new file + registry line. If it requires editing `pipeline.py`, stop and write an ADR.
- Results are files: `experiments/results/<exp>/<scene>_<run>/{config.yaml,metrics.json}` are committed; meshes are not.
- Never commit anything under `data/`. Never push unless asked.
- Tests: `make test` — must run without real data (they generate a synthetic scene) and without torch.
- Synthetic scene (`make synthetic`, `data/synthetic/`) is for wiring checks only; never report its numbers as results.
- Dev env used on 2026-09-12 lived in the session scratchpad and is gone; recreate with `make setup` (Python 3.12, open3d 0.19, torch 2.14 MPS OK).
