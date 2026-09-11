# Common entrypoints. Every target is one command so the paper's
# "reproduce" section can point here.
.PHONY: setup setup-gt sanity reference run-gt run-lidar run-mono sweep-exp1 sweep-exp2 sweep-exp3 sweep-exp4 report test lint

setup:            ## full env (GT + monocular models + dev tools)
	pip install -e ".[mono,dev]"

setup-gt:         ## minimal env, enough for Phase 0-1 (no torch)
	pip install -e ".[dev]"

sanity:           ## Phase 0: one frame -> point cloud -> outputs/sanity_<scene>_<frame>_<source>.ply
	python scripts/sanity_check.py --frame 0 --source gt
	python scripts/sanity_check.py --frame 0 --source lidar

reference:        ## Phase 1: fuse Faro depth at 1cm -> data/.../reference_mesh.ply (once per scene)
	python scripts/build_reference_mesh.py

run-lidar:        ## Phase 1: ARKit LiDAR depth -> mesh (the "iPhone Pro today" row)
	roomscan run --config configs/depth/lidar.yaml

run-gt:           ## Phase 1: GT depth -> mesh (upper bound)
	roomscan run --config configs/depth/gt.yaml

run-mono:         ## Phase 2: monocular + oracle scale
	roomscan run --config configs/depth/mono_oracle.yaml

sweep-exp1:       ## Phase 3: depth source ablation
	roomscan sweep configs/experiments/exp1_depth_source.yaml

sweep-exp2:
	roomscan sweep configs/experiments/exp2_voxel_size.yaml

sweep-exp3:
	roomscan sweep configs/experiments/exp3_frame_stride.yaml

sweep-exp4:
	roomscan sweep configs/experiments/exp4_model_size.yaml

report:           ## aggregate experiments/results/**/metrics.json -> summary tables
	roomscan report experiments/results

test:
	pytest -q

lint:
	ruff check src tests scripts
