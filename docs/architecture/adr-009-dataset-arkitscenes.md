# ADR-009: ใช้ ARKitScenes เป็น dataset หลัก (แทน ScanNet)

## Status
Accepted (2026-09-11) — supersedes ข้อ "Dataset = ScanNet" ใน development-workflow doc

## Context
Workflow doc เลือก ScanNet เพราะมี RGB/depth/pose/GT mesh ครบ แต่ต้องรอ approval และไฟล์เป็น `.sens` ที่ต้อง export
ผู้ใช้เลือก [ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes) แทน

## เปรียบเทียบ

| | ScanNet | **ARKitScenes** |
|---|---|---|
| การเข้าถึง | กรอกฟอร์ม รอ approve (ความเสี่ยงสูงในตาราง risk) | click-through license, `download_data.py` ได้ทันที |
| อุปกรณ์ที่เก็บ | Structure sensor + iPad (2017) | **iPad Pro 2020 มี LiDAR** — ตรงกับ product target |
| Depth ที่ให้ | sensor depth 640×480 (ถือเป็น GT) | `lowres_depth` 256×192 จาก ARKit (LiDAR-fused, **consumer-grade**) + `highres_depth` จาก Faro laser scanner (GT แท้, **เฉพาะบางฉาก**) |
| Pose | BundleFusion (offline, optimized) | ARKit VIO (online) — มี drift มากกว่า |
| GT mesh | `_vh_clean_2.ply` (จาก sensor depth) | ❌ ไม่มี laser mesh; `mesh` asset = mesh ที่ ARKit สร้าง (จาก LiDAR เดียวกับ lowres_depth) |
| RGB | 1296×968 | `wide` 1920×1440 + `lowres_wide` 256×192 |
| ขนาด/ฉาก | ~1–2 GB (.sens) | เลือกโหลดเฉพาะ asset ที่ต้องการได้ |

## Decision
1. `dataio/arkitscenes.py::ARKitScenesScene` เป็น dataset default; `ScanNetScene` คงไว้ใน registry เป็น fallback (stub)
2. Mapping ลง `Frame`:
   - `rgb` = `wide` (resize ลงเท่ากับ depth resolution ที่ใช้ fuse) — **VERIFY** ว่า wide กับ lowres_depth ใช้กล้องเดียวกัน (ใช่: ทั้งคู่คือ wide camera, ต่างแค่ resolution → ใช้ `Intrinsics.scaled()`)
   - `gt_depth` = `highres_depth` (Faro) เมื่อมี; None เมื่อไม่มี
   - `extra["lidar_depth"]` = `lowres_depth` (ARKit) → ป้อน `LiDARDepth` source
   - `extra["lidar_confidence"]` = `confidence` (0/1/2)
   - `pose_c2w` จาก `.traj` (timestamp, axis-angle rx ry rz, tx ty tz) — **VERIFY** ทิศทาง: helper ของ Apple (`TrajStringToMatrix`) invert หลังสร้าง Rt → ไฟล์เก็บ world-to-camera; sanity_check.py จะบอกทันทีถ้าผิด
   - intrinsics จาก `.pincam` (`w h fx fy cx cy`) ต่อเฟรม
   - จับคู่เฟรมด้วย timestamp ในชื่อไฟล์ `<video_id>_<ts>.png` กับบรรทัดใน `.traj` (tolerance ~1 ms; เฟรมที่ไม่มี pose → skip)
3. **เลือกเฉพาะฉากที่มี `highres_depth`** (3–5 ฉาก) — ไม่งั้นไม่มี GT ให้ eval

## Reference geometry (ประเด็นที่ต้องตัดสินใจเพราะไม่มี laser mesh)

| ตัวเลือก `eval.reference` | คืออะไร | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **`faro_fused`** (default) | TSDF-fuse `highres_depth` + GT pose ทุกเฟรม ที่ voxel ละเอียด (1 cm) แล้ว cache เป็น `reference_mesh.ply` | ใกล้ GT แท้ที่สุด; อิสระจาก LiDAR ของ iPad | GT row ที่ voxel/stride เดียวกับ reference จะได้ Chamfer≈0 (tautology) → **GT row ต้องรันที่ voxel/stride ของการทดลอง** เพื่อวัด "pipeline loss" ไม่ใช่ 0 |
| `arkit_mesh` | `<id>_3dod_mesh.ply` ที่ ARKit สร้าง | มีให้เลย ทุกฉาก | bias เข้าหา `lidar` row (สร้างจาก sensor เดียวกัน) — ใช้เป็น secondary เท่านั้น |

**Decision:** `faro_fused` เป็น primary, รายงาน `arkit_mesh` เป็น sanity column ถ้ามีเวลา
สร้าง reference ด้วย `scripts/build_reference_mesh.py` (Phase 1) แล้ว `ARKitScenesScene.gt_mesh()` อ่านจาก cache

## ผลต่อ Exp1
ตาราง depth source ได้ row ใหม่ที่มีค่ามาก:

| row | source | aligner | ความหมาย |
|---|---|---|---|
| gt | Faro highres | identity | upper bound / pipeline loss |
| **lidar** | ARKit lowres (iPad LiDAR) | identity | **สิ่งที่ iPhone Pro ทำได้จริงวันนี้** — คู่แข่งตรงของ mono |
| mono_oracle / mono_scene / mono_metric | โมเดล | ... | สิ่งที่ Android ทำได้ |

ช่องว่าง `lidar` vs `mono_*` คือคำตอบเชิงธุรกิจโดยตรง ("Android ห่างจาก iPhone Pro กี่ซม.")

## Trade-offs ที่ยอมรับ
- Pose จาก VIO มี drift → error นี้ปนอยู่ใน *ทุก* row เท่ากัน ยังเป็น controlled experiment; แต่ตัวเลข absolute จะสูงกว่า ScanNet — เขียนใน Limitations
- Depth 256×192 ต่ำ → fuse ที่ resolution นี้ทุก row (mono ก็ downsample output ลงมา) เพื่อความแฟร์; ทดลอง fuse mono ที่ res สูงกว่าได้เป็น extra
- `highres_depth` ไม่ครบทุกเฟรม (ถ้าเป็นเช่นนั้น) → 2D metrics คิดเฉพาะเฟรมที่มี; reference mesh ใช้เฟรมที่มี

## สิ่งที่ต้อง VERIFY ตอนโหลดฉากแรก (Phase 0 checklist)
- [ ] `lowres_depth` เป็น uint16 มิลลิเมตร? (`np.unique` ดูช่วงค่า)
- [ ] `highres_depth` มี resolution เท่าไร และมีกี่ % ของเฟรม
- [ ] `.traj` เป็น c2w หรือ w2c — ดู point cloud จาก 2 เฟรมซ้อนกันไหม
- [ ] `.pincam` ของ wide กับ lowres_wide สัมพันธ์กันด้วย scale ล้วน ๆ (cx/cy สเกลตาม)
- [ ] timestamp ชื่อไฟล์ vs traj: offset/tolerance เท่าไร

## Revisit trigger
ถ้าไม่มีฉากไหนที่ `highres_depth` ครอบคลุมพอสร้าง reference ได้ → กลับไป ScanNet (loader interface เดิม, ADR-003) หรือใช้ Replica (synthetic, GT สมบูรณ์)
