# Common entrypoints. Every target is one command so the paper's
# "reproduce" section can point here.
# All targets run inside ./.venv (Homebrew Python refuses global pip installs).
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
ROOMSCAN := $(BIN)/roomscan

.PHONY: venv setup setup-gt synthetic smoke figures-smoke figures sanity reference run-gt run-lidar run-mono sweep-exp1 sweep-exp2 sweep-exp3 sweep-exp4 report test lint

venv:
	test -d $(VENV) || $(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

setup: venv       ## full env (GT + monocular models + dev tools)
	$(BIN)/pip install -e ".[mono,dev]"
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

figures:          ## scenes x runs top-down error grids for every experiment -> paper/figures/
	for e in exp1_depth_source exp2_voxel_size exp3_frame_stride_gt exp3_frame_stride_oracle exp3_frame_stride exp4_model_size; do \
	  $(BIN)/python scripts/make_figures.py experiments/results/$$e --out paper/figures; done

report:           ## aggregate experiments/results/**/metrics.json -> summary tables
	$(ROOMSCAN) report experiments/results

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check src tests scripts
