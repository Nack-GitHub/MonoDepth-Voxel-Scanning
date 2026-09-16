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

## Status (2026-09-17)

| Phase | State |
|---|---|
| 0 loader + sanity | ✅ 6 real scans (poses interpolated between 10 Hz traj rows) |
| 1 GT/LiDAR pipeline, metrics, reference builder | ✅ 6 scenes @ 4 cm: gt chamfer 2.06 ± 0.54 cm / F@5cm 0.97, lidar 3.04 ± 0.57 / 0.93 |
| 2 monocular + scale alignment + 2D metrics | ✅ DA-v2 Large: oracle 6.45 ± 2.42 cm, per_scene 22.7 ± 2.6, metric-indoor 49.0 ± 14.7 (ADR-010) |
| 3 sweep / report / MiDaS | ✅ Exp1–4 on all 6 scenes, 138 runs (`experiments/results/*/table.md` + `per_scene.md`); Exp3 = gt control / oracle / per_scene at 4 strides |
| 4 figures + analysis | ✅ `paper/figures/<exp>_topdown_grid.png` (scenes × runs); per-frame scale analysis on 6 scenes → `paper/analysis/scale_drift_summary.md` |
| Real data | ✅ **6 scenes**, 2,352 Faro frames, 13 GB (`data/README.md`; 42444474 was the development scene, the other 5 are held-out) |
| 5 paper | ✅ full Thai draft `paper/draft/00–09` from real numbers; open: English translation, BibTeX check, `sparse_points` aligner (§8.3) |
