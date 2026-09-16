# scripts/

One-off helpers ที่ *ไม่ใช่* ส่วนของ pipeline

| script | phase | what |
|---|---|---|
| `sanity_check.py` | 0 | เฟรมเดียว → point cloud → `.ply` เปิดดูใน MeshLab; gate ของ Phase 0 |
| `build_reference_mesh.py` | 1 | fuse Faro `highres_depth` ทุกเฟรมที่ voxel 1 cm → `reference_mesh.ply` (ADR-009) |
| `make_figures.py` | 4 | grid ฉาก × run ของ top-down error ต่อ experiment → `paper/figures/<exp>_topdown_grid.png` (`--per-run` เพิ่ม heatmap `.ply` + `.png` ต่อ run ใน sub-folder) |
| `analyze_scale_drift.py` | 4 | per-frame oracle (s, t) / abs_rel ของ oracle vs per_scene vs scale-only → ตอบว่า gap มาจากโมเดลหรือจาก scale ต่อมุมมอง (ผลใน `paper/analysis/`) |
| `scene_info.py` | — | ตาราง 4.1: ขนาดห้อง (จาก reference mesh), ความยาว scan, จำนวนเฟรม Faro/VGA ของทุกฉากที่โหลดแล้ว (`--md`) |
| `summarize_scale_drift.py` | 4 | รวม JSON ของ `analyze_scale_drift.py` ทุกฉาก → ตาราง markdown ต่อโมเดล (บท 6) |
| `screen_scenes.py` | — | HEAD ขนาด asset + traj + 3DOD labels ของฉากใน metadata โดยไม่โหลด → เลือกฉากก่อนใช้ดิสก์ (ผลอยู่ใน data/README.md) |
| `make_synthetic_scene.py` | — | ห้องสังเคราะห์ใน layout ARKitScenes สำหรับ `make test` / `make smoke` |

## โหลด ARKitScenes

คำสั่งโหลด, asset ที่ pipeline ใช้จริง, ขนาดต่อฉาก และเกณฑ์เลือกฉาก อยู่ที่ [data/README.md](../data/README.md)
(ตรวจกับ source ของ Apple แล้ว — `raw` ไม่มี asset ชื่อ `wide`; RGB ความละเอียดสูงอยู่ใน dataset `upsampling`)
