# roomscan — Monocular Depth → Voxel Grid → 3D Room Reconstruction

> **Research question:** ถ้าเปลี่ยนแค่ *แหล่งที่มาของ depth* (Faro laser / iPad LiDAR / โมเดล monocular)
> โดย pipeline สร้างห้อง 3D เหมือนเดิมทุกอย่าง ความถูกต้องเชิงเรขาคณิตต่างกันกี่เซนติเมตร?

System-development paper + backend ของ MVP ในชุดโค้ดเดียว
Dataset: [ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes) · 3D: Open3D · Depth: Depth Anything V2 / MiDaS (pretrained)

## Pipeline

```
video/frames ─▶ SceneDataset ─▶ DepthSource ─▶ ScaleAligner ─▶ TSDFFusion ─▶ mesh ─▶ metrics vs reference
                                 ▲ ตัวแปรเดียว    ▲ ตัวแปรเดียว     (คงที่)
```

อ่าน [docs/architecture/README.md](docs/architecture/README.md) ก่อนแตะโค้ด — ทุกการตัดสินใจมี ADR

## Quickstart

```bash
make setup            # creates ./.venv and installs everything into it
# make setup-gt         # lighter: no torch (Phase 0-1 only)
```

โหลดข้อมูล 3 ฉากตาม [scripts/README.md](scripts/README.md) แล้ว:

```bash
make sanity            # Phase 0 gate: one frame -> point cloud, open in MeshLab
make reference         # Phase 1: build reference_mesh.ply from Faro depth (once per scene)
make run-gt            # upper bound
make run-lidar         # iPad LiDAR row
make run-mono          # Phase 2
make sweep-exp1        # Phase 3 ablations ... sweep-exp4
make report            # -> experiments/results/summary.csv + per-experiment table.md
```

Any single run: `roomscan run --config configs/depth/gt.yaml --set fusion.voxel_size=0.02`

## Layout

```
configs/         base.yaml + depth presets + experiments/exp{1..4}_*.yaml  (1 table = 1 file)
src/roomscan/    the package — see docs/architecture/README.md §4
scripts/         sanity_check, build_reference_mesh, data download notes
tests/           pure-numpy tests (no data, no torch)
experiments/     results/<exp>/<scene>_<run>/{config.yaml, metrics.json}  (meshes gitignored)
docs/architecture/  overview + ADR-001..009
paper/           outline, figures
data/            NOT in git — see data/README.md
```

## No data yet? Run the whole thing on a synthetic room

```bash
make synthetic         # writes data/synthetic/ in ARKitScenes layout (~90 MB, 60 frames)
make smoke             # gt / lidar / mono_oracle / mono_scene sweep + report tables
make figures-smoke     # error heatmaps -> outputs/figures_smoke/
```

## Status (2026-09-12)

| Phase | State |
|---|---|
| 0 loader + sanity | ✅ implemented, tested on synthetic scene; **not yet run on real ARKitScenes** |
| 1 GT/LiDAR pipeline, metrics, reference builder | ✅ implemented; synthetic GT: accuracy 1.1 cm, precision@5cm 0.999 |
| 2 monocular + scale alignment + 2D metrics | ✅ implemented; DA-v2 Small verified end-to-end on MPS |
| 3 sweep / report / MiDaS | ✅ sweep+report verified (`exp0_synthetic_smoke`); MiDaS wrapper untested |
| 4 figures | ✅ `scripts/make_figures.py` (heatmap ply + top-down png) |
| Real data | ⏳ blocked on disk space + download — see `data/README.md` |
| 5 paper | ⏳ outline only |
