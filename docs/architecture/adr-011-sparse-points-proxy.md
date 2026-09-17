# ADR-011: ประเมิน `sparse_points` aligner ด้วยจุด sparse จำลองจาก LiDAR (VIO proxy)

## Status
Accepted (2026-09-17) — ขยาย ADR-002 (aligner ตัวที่ 4 ที่มี interface อยู่แล้ว ได้ข้อมูลเข้าเป็นครั้งแรก)

## Context
Exp1 มีช่องว่างใหญ่ระหว่าง `oracle_frame` (6.5 cm) กับ `per_scene` (22.7 cm) และ §6 สรุปว่าสาเหตุคือ
scale ของโมเดล**ต่อมุมมอง** ไม่ใช่ drift ตามเวลา ⇒ ทางแก้ที่ deploy ได้คือ fit `(s, t)` **ต่อเฟรม**
จากข้อมูล metric ที่เครื่องมีอยู่แล้ว — จุด feature 3D ของ VIO (ARKit `rawFeaturePoints`, ARCore point cloud)
`SparsePointsAligner` มี interface นี้ตั้งแต่ ADR-002 แต่ไม่เคยถูกประเมิน เพราะ ARKitScenes
ไม่ได้แจก feature points ของ ARKit มาด้วย

## Options

| Option | Pros | Cons |
|---|---|---|
| A. ไม่ประเมิน ทิ้งไว้ future work (เดิม) | ไม่มีข้อมูลสังเคราะห์ในเปเปอร์ | คำถาม "ถ้ามี VIO points จะปิดช่องว่างได้ไหม" ไม่มีคำตอบเชิงตัวเลข |
| B. **sample จุดจาก LiDAR frame เป็น proxy** (`n` pixel, confidence 2, seed ต่อเฟรม) | ใช้ข้อมูลจริงของฉากเดียวกัน; ควบคุมจำนวน/noise ได้; ไม่แตะ pipeline | มองโลกในแง่ดี: จุด LiDAR สะอาดกว่าจุด VIO ที่ triangulate; กระจายสม่ำเสมอไม่ใช่ตาม texture |
| C. รัน visual SLAM (เช่น ORB-SLAM) บน RGB เพื่อได้จุดจริง | ใกล้ความจริงที่สุด | ต้องเพิ่ม dependency ใหญ่ + scale ของ SLAM แบบ mono ก็ไม่ metric อยู่ดี ต้อง align กับ traj อีกชั้น |

## Decision
**B.** loader ของ ARKitScenes ได้ option `sparse_points: N` (default 0 = ปิด) — ถ้าเปิด จะ sample N pixel
จาก `lowres_depth` ที่ `confidence >= sparse_min_confidence` (default 2) ด้วย `seed = frame idx`
ใส่ใน `frame.extra["sparse_depth"]` (`dataio/sparse_proxy.py`) — `pipeline.py` ไม่เปลี่ยน

`SparsePointsAligner` fit ต่อเฟรมใน inverse-depth (ADR-010) ด้วย trimmed LS 2 รอบ ตัด 10 %
ถ้าเฟรมมีจุดใช้ได้ < `depth.sparse_min_points` (20) ให้ใช้ `(s, t)` ของเฟรมก่อนหน้า (นับเป็น `n_fallback`)
ซึ่งเป็นสิ่งที่ระบบจริงทำเมื่อ tracker หลุดชั่วคราว

ค่า default ในเปเปอร์: **N = 200** — อยู่ในช่วงจำนวน feature points ต่อเฟรมของ ARKit ในห้องปกติ (ราวร้อยถึงหลายร้อย)
`sparse_noise` (relative std) มีไว้ทำ sensitivity ถ้าต้องการ; ผลหลักใช้ 0

เพิ่มแถว `mono_sparse` ใน Exp1 (`configs/experiments/exp1_depth_source.yaml`) และ preset `configs/depth/mono_sparse.yaml`

## Consequences
- ✅ ตอบคำถาม "ราคาของการไม่รู้ scale" ครบทั้งสามระดับ: รู้ทุก pixel (oracle) / รู้ ~200 จุดต่อเฟรม (sparse) / รู้ครั้งเดียวต่อฉาก (per_scene)
- ⚠️ ตัวเลข `mono_sparse` เป็น **upper bound ของ VIO จริง** ต้องเขียนกำกับทุกครั้งที่อ้าง (§4, §5, §8)
- ⚠️ ห้ามใช้ proxy กับแถว `lidar`/`gt` — จุดมาจาก LiDAR เอง ไม่มีความหมาย
- `dataio/custom.py` อ่าน `sparse/<stem>.png` เป็นจุดจริงจากแอป ⇒ ตอน MVP เปลี่ยนแค่ dataset ไม่ต้องแก้ aligner
