# Common entrypoints. Every target is one command so the paper's
# "reproduce" section can point here.
# All targets run inside ./.venv (Homebrew Python refuses global pip installs).
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
ROOMSCAN := $(BIN)/roomscan

.PHONY: venv setup setup-gt synthetic smoke figures-smoke figures paper-figures paper capture-zip sanity reference run-gt run-lidar run-mono sweep-exp1 sweep-exp2 sweep-exp3 sweep-exp4 sweep-exp5 sweep-exp6 report web train test lint

venv:
	test -d $(VENV) || $(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

setup: venv       ## full env (GT + monocular models + dev tools + web)
	$(BIN)/pip install -e ".[mono,dev,web]"
	@echo "done — run tools via 'make ...' or 'source .venv/bin/activate'"

setup-gt: venv    ## minimal env, enough for Phase 0-1 (no torch)
	$(BIN)/pip install -e ".[dev]"

synthetic:        ## synthetic ARKitScenes-format room for testing without real data
	$(BIN)/python scripts/make_synthetic_scene.py --root data/synthetic --frames 60

smoke:            ## all Exp1 row types on the synthetic room + tables
	$(ROOMSCAN) sweep configs/experiments/exp0_synthetic_smoke.yaml
	$(ROOMSCAN) report experiments/results

figures-smoke:
	$(BIN)/python scripts/make_figures.py experiments/results/exp0_synthetic_smoke --out outputs/figures_smoke

sanity:           ## Phase 0: one frame -> point cloud -> outputs/sanity_<scene>_<frame>_<source>.ply
	$(BIN)/python scripts/sanity_check.py --frame 0 --source gt
	$(BIN)/python scripts/sanity_check.py --frame 0 --source lidar

reference:        ## Phase 1: fuse Faro depth at 1cm -> data/.../reference_mesh.ply (once per scene)
	$(BIN)/python scripts/build_reference_mesh.py

run-lidar:        ## Phase 1: ARKit LiDAR depth -> mesh (the "iPhone Pro today" row)
	$(ROOMSCAN) run --config configs/depth/lidar.yaml

run-gt:           ## Phase 1: GT depth -> mesh (upper bound)
	$(ROOMSCAN) run --config configs/depth/gt.yaml

run-mono:         ## Phase 2: monocular + oracle scale
	$(ROOMSCAN) run --config configs/depth/mono_oracle.yaml

sweep-exp1:       ## Phase 3: depth source ablation
	$(ROOMSCAN) sweep configs/experiments/exp1_depth_source.yaml

sweep-exp2:
	$(ROOMSCAN) sweep configs/experiments/exp2_voxel_size.yaml

sweep-exp3:       ## gt control + mono oracle + mono per_scene at strides 1/5/10/20
	$(ROOMSCAN) sweep configs/experiments/exp3_frame_stride_gt.yaml --skip-existing
	$(ROOMSCAN) sweep configs/experiments/exp3_frame_stride_oracle.yaml --skip-existing
	$(ROOMSCAN) sweep configs/experiments/exp3_frame_stride.yaml --skip-existing

sweep-exp4:
	$(ROOMSCAN) sweep configs/experiments/exp4_model_size.yaml

sweep-exp5:       ## LiDAR confidence masking / weighting (ADR-012)
	$(ROOMSCAN) sweep configs/experiments/exp5_lidar_confidence.yaml --skip-existing

sweep-exp6:       ## ADR-013: the three ARKitScenes fine-tunes vs the pretrained metric model, RGB-only
	$(ROOMSCAN) sweep configs/experiments/exp6_finetune.yaml --skip-existing

figures:          ## scenes x runs top-down error grids for every experiment -> paper/figures/
	for e in exp1_depth_source exp2_voxel_size exp3_frame_stride_gt exp3_frame_stride_oracle exp3_frame_stride exp4_model_size; do \
	  $(BIN)/python scripts/make_figures.py experiments/results/$$e --out paper/figures; done

paper-figures:    ## analysis figures (scale drift, pipeline diagram) -> paper/figures/
	$(BIN)/python scripts/make_paper_figures.py

paper:            ## compile paper/latex/main.tex -> paper/latex/main.pdf (needs tectonic)
	cd paper/latex && tectonic main.tex

report:           ## aggregate experiments/results/**/metrics.json -> summary tables
	$(ROOMSCAN) report experiments/results

capture-zip:      ## export a scene as the app's capture zip for the web UI -> outputs/captures/<scene>.zip
	$(BIN)/python scripts/export_capture.py --scene $(or $(SCENE),42444474) --stride 5

web:              ## Phase 6: upload/queue/view API on http://localhost:8765 (needs .[web])
	$(BIN)/uvicorn --factory roomscan_web.app:create_app --host 0.0.0.0 --port 8765

train:            ## fine-tune DA-v2 metric (ADR-013): make train RUN=configs/training/r1_ft_faro.yaml
	$(BIN)/python -m roomscan.training.train --config $(RUN)

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check src tests scripts
