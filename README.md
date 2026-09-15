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

โหลดข้อมูลตาม [data/README.md](data/README.md) แล้ว:

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
docs/architecture/  overview + ADR-001..010
paper/           outline, figures
data/            NOT in git — see data/README.md
```

## No data yet? Run the whole thing on a synthetic room

```bash
make synthetic         # writes data/synthetic/ in ARKitScenes layout (~90 MB, 60 frames)
make smoke             # gt / lidar / mono_oracle / mono_scene sweep + report tables
make figures-smoke     # error heatmaps -> outputs/figures_smoke/
```

## Status (2026-09-15)

| Phase | State |
|---|---|
| 0 loader + sanity | ✅ on real scan 42444474 (poses interpolated between 10 Hz traj rows; 235/236 highres frames kept) |
| 1 GT/LiDAR pipeline, metrics, reference builder | ✅ 42444474 @ 4 cm: gt chamfer 1.78 cm / F@5cm 0.965, lidar 2.84 cm / 0.904 |
| 2 monocular + scale alignment + 2D metrics | ✅ DA-v2 Large: oracle 5.4 cm, per_scene 23.9 cm, metric-indoor 53.2 cm (ADR-010) |
| 3 sweep / report / MiDaS | ✅ Exp1–4 run on 42444474 (`experiments/results/*/table.md`); MiDaS small works via torch.hub |
| 4 figures | ✅ `paper/figures/` — heatmap ply + top-down png per run + Exp1 grid |
| Real data | ⏳ **1 scene only** (42444474: open-plan kitchen/dining, 9.8×5.2 m, sloped ceiling, glass table + balustrade). Need 2–5 more — `data/README.md`; disk is the limit (17 GB free, ~2–3 GB/scene) |
| 5 paper | ⏳ outline only |
