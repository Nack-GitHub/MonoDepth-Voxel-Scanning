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
make sweep-exp1        # Phase 3 ablations ... sweep-exp5
make report            # -> experiments/results/summary.csv + per-experiment table.md
make figures paper-figures   # top-down error grids + scale-drift / pipeline figures -> paper/figures/
make paper             # paper/latex/main.tex -> main.pdf (tectonic)
make web               # Phase 6: http://localhost:8765 — upload a capture.zip, get a mesh in the browser
```

Any single run: `roomscan run --config configs/depth/gt.yaml --set fusion.voxel_size=0.02`

## Layout

```
configs/         base.yaml + depth presets + experiments/exp{1..4}_*.yaml  (1 table = 1 file)
src/roomscan/    the package — see docs/architecture/README.md §4
src/roomscan_web/  FastAPI upload/queue/status API + three.js viewer (imports roomscan, never the reverse)
scripts/         sanity_check, build_reference_mesh, data download notes
tests/           pure-numpy tests (no data, no torch)
experiments/     results/<exp>/<scene>_<run>/{config.yaml, metrics.json}  (meshes gitignored)
docs/architecture/  overview + ADR-001..012
paper/           outline, draft/ (Thai), latex/ (English IEEE manuscript + refs.bib), figures/, analysis/
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
| 3 sweep / report / MiDaS | ✅ Exp1–5 on all 6 scenes, 174 runs (`experiments/results/*/table.md` + `per_scene.md`); Exp3 = gt control / oracle / per_scene at 4 strides |
| 4 figures + analysis | ✅ `paper/figures/<exp>_topdown_grid.png` (scenes × runs), `scale_drift_timeline`, `scale_vs_depth`, `pipeline` (`make paper-figures`); `paper/analysis/scale_drift_summary.md` |
| Real data | ✅ **6 scenes**, 2,352 Faro frames, 13 GB (`data/README.md`; 42444474 was the development scene, the other 5 are held-out) |
| 5 sparse_points + Exp5 | ✅ `mono_sparse` row (200 LiDAR-sampled points/frame as VIO proxy, ADR-011): 7.2 ± 2.9 cm, F@5cm 0.69 — 0.8 cm from the oracle, 3.1× better than per-scene calibration; Exp5 LiDAR confidence weighting (ADR-012): no gain (3.04 → 3.07 cm) |
| 6 web / MVP loader | ✅ `src/roomscan_web/` (`make web`) + `dataio/custom.py` (app capture folder: rgb/ poses.json intrinsics.json [depth/ confidence/ sparse/]) |
| 7 paper | ✅ Thai draft `paper/draft/00–09`; English IEEE manuscript `paper/latex/` with `refs.bib` (15 refs); open: author/affiliation, venue-specific formatting |
