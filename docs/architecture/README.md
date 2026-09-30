# Architecture — roomscan

> Monocular Depth → TSDF Voxel Grid → 3D Room Mesh, ออกแบบให้ **เปเปอร์กับ MVP เป็นโค้ดชุดเดียวกัน**

ตัวเลือกเชิงสถาปัตยกรรมทุกข้อมี ADR กำกับใน `docs/architecture/adr-*.md` — ถ้าจะเปลี่ยนอะไรที่นี่ ให้เขียน ADR ใหม่ที่ supersede อันเดิม ไม่แก้อันเก่า

## 1. Classification

| | |
|---|---|
| ประเภท | Research pipeline ที่ต้องกลายเป็น product backend ได้ (offline batch) |
| ทีม | 1 คน |
| Timeline | ~10 สัปดาห์ → paper, แล้วค่อย MVP |
| Scale | 3–5 ฉาก, ~1–5k เฟรม/ฉาก, รันบน GPU เครื่องเดียว/Colab |
| ⇒ | **Simple + Modular**: monolith package เดียว, ไม่มี service, ไม่มี DB, abstraction เฉพาะจุดที่เป็นแกนของการทดลอง |

## 2. Component view

```mermaid
flowchart LR
    subgraph Input
        DS[SceneDataset<br/>arkitscenes / scannet / custom]
    end
    subgraph Variable["ตัวแปรของการทดลอง"]
        DEPTH[DepthSource<br/>gt · lidar · mono · arcore]
        MODEL[DepthModel<br/>DA-v2 L/S, MiDaS, ...]
        ALIGN[ScaleAligner<br/>identity / oracle / per_scene / sparse]
        MODEL --> DEPTH
    end
    subgraph Fixed["คงที่ทุกการทดลอง"]
        TSDF[TSDFFusion<br/>Open3D ScalableTSDFVolume]
        POST[postprocess]
        EXP[export ply/obj/glb]
    end
    subgraph Eval
        M3[metrics_3d<br/>Chamfer, F@5cm, normals]
        M2[metrics_2d<br/>RMSE, AbsRel, δ]
        REP[report → tables]
    end
    DS -->|Frame| DEPTH -->|relative or metric depth| ALIGN -->|metric depth| TSDF --> POST --> EXP
    POST --> M3
    ALIGN --> M2
    DS -->|reference mesh| M3
    M3 --> REP
    M2 --> REP
    PIPE[[pipeline.ReconstructionPipeline]] -.orchestrates.- DS & DEPTH & ALIGN & TSDF & M3
    CLI[[cli: run / sweep / report]] --> PIPE
    CFG[(configs/*.yaml)] --> CLI
```

## 3. Data flow ต่อเฟรม

```
Frame{rgb, pose_c2w, gt_depth?, extra{lidar_depth, lidar_confidence}}   ← ARKitScenes mapping ใน ADR-009
   │
   ▼  DepthSource.get_depth(frame)             ← สลับได้ (Exp1, Exp4)
pred: (H,W) float32   [metric | relative]
   │
   ▼  ScaleAligner.align(pred, frame)          ← สลับได้ (Exp1)
depth_m: (H,W) float32 metres, 0/NaN = invalid
   │
   ▼  TSDFFusion.integrate(rgb, depth_m, pose) ← voxel_size (Exp2), stride มาจาก dataset.frames() (Exp3)
```

**Invariants** (บังคับใน `types.py` + `pipeline._check_consistency`):
1. หลัง aligner ค่า depth ต้องเป็นเมตรเสมอ
2. `pose_c2w` เป็น camera→world; การ invert เกิดที่ `TSDFFusion.integrate` **ที่เดียว**
3. Intrinsics ผูกกับ resolution — resize รูปแล้วต้อง `Intrinsics.scaled()`
4. DepthSource ที่ `is_metric=True` ต้องคู่กับ `identity` aligner; `is_metric=False` ต้องไม่ใช่ identity

## 4. Module map & dependency rule

```
src/roomscan/
├── types.py            Intrinsics, Frame, Timing            ← ไม่ import อะไรใน package
├── config.py           YAML + _base_ + dotlist overrides
├── _registry.py        string → class (lazy import, GT path ไม่ต้องมี torch)
├── dataio/             SceneDataset ABC + ARKitScenesScene, CustomCaptureScene (MVP), sparse_proxy (ADR-011) ← types
├── models/             DepthModel ABC + DA-v2, MiDaS         ← types
├── depth_sources/      DepthSource ABC + gt/lidar/mono/arcore ← models, types
├── geometry/           backproject, scale_align, tsdf_fusion, postprocess ← types
├── evaluation/         metrics_2d, metrics_3d, report
├── training/           fine-tune DA-v2 metric (ADR-013) ← dataio, evaluation.metrics_2d, types, config (ไม่ใช่ pipeline/geometry/open3d)
├── export.py
├── pipeline.py         orchestrator (ห้ามมี Open3D/torch call ตรง ๆ)
├── eval2d.py           2D-only evaluate / re-evaluate (ADR-014) — ใช้ `pipeline._to_grid` + `RunResult`, ไม่ fuse
└── cli.py              run / sweep / report / eval2d / reeval2d
src/roomscan_web/       Phase 6: FastAPI upload/queue + gallery ของ run ที่ sweep ไว้ + three.js viewer (เทียบ 3 ช่อง, สี error, Faro overlay)
                        — imports roomscan.pipeline / evaluation.metrics_3d / dataio, never the reverse (รายละเอียด §7.1)
```

กฎ: import ไหลลงล่างเท่านั้น (`cli → pipeline → stages → types`) ไม่มี stage ไหน import `pipeline`

## 5. Extension points → สิ่งที่ต้องแตะ

| อยากเพิ่ม | เพิ่มไฟล์ | แก้ registry | แก้ pipeline? |
|---|---|---|---|
| depth source ใหม่ (เช่น ARCore จากมือถือจริง) | `depth_sources/<x>.py` | `depth_sources/__init__.py` | ❌ |
| โมเดล depth ใหม่ (Exp4 / Project 1) | `models/<x>.py` | `models/__init__.py` | ❌ |
| วิธี align scale ใหม่ | class ใน `geometry/scale_align.py` | `build_aligner` | ❌ |
| dataset ใหม่ (ScanNet กลับมา, Replica, ห้องจริง) | `dataio/<x>.py` | `dataio/__init__.py` | ❌ |
| backend TSDF อื่น (GPU) | class ใน `geometry/tsdf_fusion.py` | — | ❌ (interface เดิม) |
| การทดลองใหม่ | `configs/experiments/<x>.yaml` | — | ❌ |

ถ้าการเพิ่มอะไรสักอย่างต้องแก้ `pipeline.py` แปลว่า abstraction รั่ว — ให้หยุดแล้วเขียน ADR

## 6. Experiment ↔ config mapping

| Exp | ตัวแปร | config key | file |
|---|---|---|---|
| 1 Depth source | gt / **lidar** / mono×aligner | `depth.source`, `depth.aligner`, `depth.model` | `exp1_depth_source.yaml` |
| 2 Voxel size | 2/4/8 cm | `fusion.voxel_size`, `fusion.sdf_trunc` | `exp2_voxel_size.yaml` |
| 3 Frame stride | 1/5/10/20 × {gt, mono+oracle, mono+per_scene} | `dataset.frame_stride` | `exp3_frame_stride_gt.yaml` (control: coverage อย่างเดียว) / `exp3_frame_stride_oracle.yaml` (+รูปทรงโมเดล) / `exp3_frame_stride.yaml` (+scale, deployable) — Faro มีแค่ ~2.7 fps คำถามจึงเป็น "ต้องเก็บกี่เฟรม/วินาที" ไม่ใช่ compute-vs-accuracy (ดู `paper/analysis/`) |
| 4 Model size | L / S / MobileViT | `depth.model` | `exp4_model_size.yaml` |
| 5 LiDAR confidence | all / mask ≥1 / mask =2 / weight [0,1,2] / [1,2,4] | `depth.lidar_min_confidence`, `fusion.confidence_weights` | `exp5_lidar_confidence.yaml` (ADR-012) — ผล: ไม่ต่างจาก baseline (3.04 → 3.07 cm) |
| 6 Fine-tune teacher | pretrained / ft_faro / ft_lidar / ft_lidar_all (RGB-only, `aligner=identity`) | `depth.model` | `exp6_finetune.yaml` (ADR-013) — DA-v2 metric ที่ fine-tune บน ARKitScenes ด้วยครู Faro เทียบครู LiDAR; 6 ฉากเทสต์ถูกกันออกจาก train/val ทั้งหมด |
| 7 Generalisation (2D) | pretrained / ft_lidar_all / lidar บน 20 ห้อง fold Validation | `depth.model` | `exp7_valfold_2d.yaml` (ADR-014) — `roomscan eval2d` เท่านั้น (ไม่มี reference mesh), เทียบ Faro depth ของ dataset |

ทุก run ทิ้ง `config.yaml` + `metrics.json` ไว้ที่ `experiments/results/<exp>/<scene>_<run>/` — `roomscan report` รวมเป็นตาราง

## 7. Phase → module (สถานะ 2026-09-23: 0–6 ผ่าน gate; Exp1–6 รันครบ 6 ฉาก — Exp1/3/4 รันใหม่หลังแก้การหมุนภาพ (PR #1) และ Exp6 = fine-tune ตาม ADR-013; ร่างไทย `paper/draft/` + ต้นฉบับอังกฤษ `paper/latex/`)

| Phase | ต้องทำให้ทำงาน | gate |
|---|---|---|
| 0 | `dataio/arkitscenes.py`, `scripts/sanity_check.py` + VERIFY list ใน ADR-009 | point cloud เฟรมเดียว (gt และ lidar) ดูออกว่าเป็นห้อง; 2 เฟรมซ้อนกันได้ |
| 1 | `geometry/tsdf_fusion.py`, `postprocess.py`, `export.py`, `evaluation/metrics_3d.py`, `scripts/build_reference_mesh.py` | `make reference` → `make run-gt` + `make run-lidar` ได้ mesh + ตัวเลข 2 แถวแรกของ Exp1 |
| 2 | `models/depth_anything.py`, `depth_sources/monocular.py`, `scale_align.fit_scale_shift`, `PerSceneAligner.fit`, `metrics_2d.py` | `make run-mono` |
| 3 | `models/midas.py`, `evaluation/report.py` | `make sweep-exp1..4` + `make report` |
| 4 | `per_point_error()` สำหรับ heatmap (ใน metrics_3d) | figures |
| 5 | `dataio/sparse_proxy.py` + `SparsePointsAligner` (ADR-011), `TSDFFusion.integrate(weights=)` (ADR-012), `dataio/custom.py` | แถว `mono_sparse` ใน Exp1 + Exp5 บน 6 ฉาก; `make test` ผ่านโดยไม่มีข้อมูลจริง |
| 6 | `roomscan_web/` (upload zip → job → mesh.ply → three.js viewer; ขยายเป็นเครื่องมือ demo เมื่อ 2026-09-30 ดู §7.1) | `make web` แล้ว POST capture.zip ได้ mesh ใน browser |

**ห้ามข้าม gate ของ Phase 1** — ถ้า GT depth ยังได้ mesh เละ ปัญหาอยู่ที่ pipeline ไม่ใช่โมเดล

### 7.1 Phase 6: `roomscan_web/` หลังขยายสำหรับ demo (SPEC.md, 2026-09-30)

เว็บยังเป็นชั้นบาง ๆ บน pipeline เหมือนเดิม คือไม่แก้ `pipeline.py` และไม่มี DB
สิ่งที่เพิ่มคือการแสดงผลสามอย่างที่เปเปอร์ต้องการให้เห็นด้วยตา ได้แก่ mesh หลายก้อนของห้องเดียวกันในกล้องเดียว สี error เทียบกับ Faro และ Faro wireframe ซ้อนทับ

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | routes เท่านั้น ตรวจ input แล้วเรียก module อื่น |
| `presets.py` | preset ของ dropdown (`lidar`, `ft_lidar_24`, `mono_metric`, `mono_sparse`) เป็นแหล่งเดียวของชื่อบนหน้าเว็บ ซึ่งต้องตรงกับชื่อในเปเปอร์ |
| `jobs.py` | คิวงาน worker เดียว เขียน `results/web/<id>/job.json` ทุกครั้งที่ status เปลี่ยน ตอนเปิด server จะอ่านกลับมา งานที่ค้าง `queued`/`running` กลายเป็น `failed: interrupted by server restart` และไม่ถูกรันซ้ำเอง อ่าน `meta.json` ของ capture (`scene`, `up`, `stride`, `sky_direction`) และส่ง `sky_direction` ต่อให้ `depth_sources/monocular.py` ผ่าน dataset ที่ inject เข้า pipeline เพราะ capture ที่ export จาก ARKitScenes เก็บภาพไม่ตั้งตรง |
| `gallery.py` | อ่าน `experiments/results/<exp>/<scene>_<run>/` แบบอ่านอย่างเดียว ตรวจ `exp` กับ allowlist และ `run` กับ regex ก่อนสร้าง path ทุกครั้ง ไม่แสดง `exp0_synthetic_smoke` |
| `errorcolor.py` | ระยะจาก vertex ของ mesh ไปยังจุดที่ใกล้ที่สุดบน reference (200k จุด, seed เดียวกับ `per_point_error`) แล้วระบายสี turbo 0–10 cm เท่ากับรูป error ในเปเปอร์ |
| `refmesh.py` | สำเนา reference สำหรับแสดงผล (vertex clustering 4 cm แล้ว decimate เหลือไม่เกิน 30k สามเหลี่ยม) เพราะ reference 1 cm ที่ใช้วัดผลหนักเกินไปสำหรับ browser (ห้อง 47429736: 1.6 ล้านสามเหลี่ยม) สี error คำนวณจาก reference ตัวเต็มเสมอ viewer วาดสำเนานี้เป็น shell โปร่งแสงชั้นเดียวทับ mesh |
| `static/` | `index.html` + `app.css` + ES modules (`js/api.js`, `viewer.js`, `panels.js`, `format.js`, `main.js`) ไม่มี build step และ three.js 0.160.0 อยู่ใน `vendor/three/` (ดู `VERSION`) หน้าเว็บจึงไม่โหลดอะไรจากอินเทอร์เน็ต |

| Endpoint | คืนอะไร |
|---|---|
| `GET /presets` | preset ตามลำดับใน dropdown |
| `POST /scans` (`file`, `preset`, `overrides`) | สร้างงาน |
| `GET /scans`, `GET /scans/{id}` | งานพร้อม `preset`, `label`, `scene`, `up`, `has_reference`, `created_at`/`started_at`/`finished_at`, `elapsed_s`, `eta_s` |
| `GET /scans/{id}/mesh.ply`, `reference.ply`, `error.ply` | mesh, สำเนา reference สำหรับแสดงผล, mesh ที่ระบายสี error |
| `GET /gallery` | run ที่มี `mesh.ply` + `metrics.json` พร้อม label ตามเปเปอร์ |
| `GET /gallery/{exp}/{run}/mesh.ply`, `reference.ply`, `error.ply` | เหมือนฝั่ง scans |
| `GET /static/...` | css / js / three.js ที่ vendor ไว้ |

ไฟล์ที่เว็บสร้างขึ้นเองมีสองที่เท่านั้น คือ `outputs/web/results/web/<id>/` (`job.json`, `reference_view.ply`, `error.ply` ข้างผลของ pipeline) และ `outputs/web/cache/<exp>/` สำหรับ gallery (`<scene>/reference.ply`, `<scene>/reference_view.ply`, `<scene>_<run>/error.ply`)
เว็บไม่เขียนอะไรลง `experiments/results/` และ cache จะถูกสร้างใหม่เมื่อ `mesh.ply` หรือ reference ใหม่กว่า
reference ของ gallery ได้จาก `build_dataset(cfg.dataset).gt_mesh()` ตาม `config.yaml` ของ run นั้น ถ้าเครื่องไม่มี dataset รายการจะมี `has_reference: false` และปุ่มสี error กับ overlay จะถูกปิด
`scripts/warm_web_cache.py` สร้างไฟล์เหล่านี้ล่วงหน้าด้วยฟังก์ชันชุดเดียวกับ endpoint

ตัวแปรแวดล้อม: `ROOMSCAN_WORK_DIR` (`outputs/web`), `ROOMSCAN_PRESET`, `ROOMSCAN_MAX_UPLOAD_MB`, `ROOMSCAN_GALLERY_ROOT` (`experiments/results`), `ROOMSCAN_GALLERY_EXPS` (`exp1_depth_source,exp6_finetune`)

ตัวเลขบนหน้าเว็บมีป้ายบอกที่มาเสมอ คือ "paper run" สำหรับ gallery และ "demo run (stride N, no GT mask)" สำหรับงานที่อัปโหลด เพราะงานบนเว็บรันด้วย capture ที่ลดเฟรมแล้วและไม่ได้ mask ด้วย GT จึงไม่ใช่ตัวเลขของเปเปอร์

## 8. ADR index

| ADR | เรื่อง | สถานะ |
|---|---|---|
| [001](adr-001-depth-source-strategy.md) | DepthSource เป็น strategy interface | Accepted |
| [002](adr-002-scale-alignment-separate-stage.md) | แยก ScaleAligner ออกจาก DepthSource | Accepted |
| [003](adr-003-scene-dataset-abstraction.md) | SceneDataset abstraction (ScanNet ไม่ hardcode) | Accepted |
| [004](adr-004-open3d-backend.md) | Open3D เป็น 3D backend | Accepted |
| [005](adr-005-config-driven-experiments.md) | YAML + OmegaConf, ไม่ใช้ Hydra | Accepted |
| [006](adr-006-results-layout.md) | ผลการทดลองเป็นไฟล์ flat, ไม่มี DB/tracker | Accepted |
| [007](adr-007-src-layout-single-package.md) | `src/` layout, package เดียว, ไม่แยก repo | Accepted |
| [008](adr-008-pretrained-inference-only.md) | ใช้โมเดล pretrained, ไม่เทรน | Superseded by 013 |
| [009](adr-009-dataset-arkitscenes.md) | **ARKitScenes** แทน ScanNet + วิธีสร้าง reference mesh | Accepted |
| [010](adr-010-scale-alignment-in-inverse-depth.md) | fit scale/shift ใน inverse-depth (ขยาย 002) — oracle 15 → 5.4 cm | Accepted |
| [011](adr-011-sparse-points-proxy.md) | ประเมิน `sparse_points` ด้วยจุด sparse จำลองจาก LiDAR (VIO proxy, 200 จุด/เฟรม) | Accepted |
| [012](adr-012-confidence-weighted-fusion.md) | น้ำหนักต่อ pixel ใน TSDF จาก confidence (integrate ซ้ำตามระดับ) — แก้ `pipeline.py` 2 บรรทัด | Accepted |
| [013](adr-013-finetune-with-lidar-teacher.md) | Fine-tune DA-v2 metric ด้วยครู LiDAR/Faro ใน `training/` (freeze encoder), split scene-disjoint ตาม `visit_id` | Accepted |
| [014](adr-014-2d-protocol-and-valfold-test.md) | 2D metrics protocol 2 (mask ตาม GT, clip pred) + `eval2d`/`reeval2d` + Exp7 เทสต์ 2D 20 ห้องจาก fold Validation | Accepted |

## 9. สิ่งที่ตั้งใจ *ไม่* ทำตอนนี้

- ✅ ~~Web API / queue / three.js viewer~~ — ทำแล้วเป็น `src/roomscan_web/` (thread queue + FastAPI, ไม่มี broker; งานเก็บเป็นไฟล์ `job.json`); ยังไม่มี auth/multi-worker และยังไม่มีความคืบหน้ารายเฟรม (ต้องแก้ `pipeline.py` จึงต้องเขียน ADR ก่อน)
- ❌ Experiment tracker (W&B/MLflow) — 6 ฉาก × ~30 run = JSON ก็พอ
- ❌ GPU TSDF — เครื่องพัฒนาเป็น Apple Silicon (Open3D CUDA ใช้ไม่ได้) และ fusion < 2 s/ฉาก ไม่ใช่คอขวด; ใส่ได้ทีหลังหลัง interface `TSDFFusion` เดิม
- ❌ Pose estimation (COLMAP/ARKit) — ARKitScenes ให้ pose (VIO) มาแล้ว; `dataio/custom.py` รับ pose จากแอป (ARKit/ARCore) ใน `poses.json`
- ✅ ~~confidence-weighted fusion~~ — ทำเป็น Exp5 (ADR-012): ไม่ช่วย (3.04 → 3.07 cm) — depth upsampling ยังไม่ทำ
