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
make setup-gt          # Phase 0-1: numpy/open3d/omegaconf (no torch)
# make setup           # + torch/transformers for monocular models
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

## Status

Phase 0 — skeleton + contracts committed; loader not yet implemented. See [docs/architecture/README.md §7](docs/architecture/README.md) for the phase → module map.
