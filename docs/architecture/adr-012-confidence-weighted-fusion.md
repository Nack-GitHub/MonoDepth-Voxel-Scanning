# ADR-012: น้ำหนักต่อ pixel ใน TSDF จาก confidence ของเซ็นเซอร์

## Status
Accepted (2026-09-17) — แก้ `pipeline.py` 1 จุด (ส่ง `weights` ให้ `fusion.integrate`) จึงต้องมี ADR ตามกฎ §4

## Context
ARKit ให้ confidence map {0,1,2} คู่กับ LiDAR depth ทุกเฟรม Exp1–4 ตั้งใจ**ไม่ใช้** เพื่อให้ตารางวัด
depth source ล้วน ๆ (ทุก pixel น้ำหนัก 1) README §9 ระบุว่า confidence-weighted fusion เป็น "อีกเปเปอร์"
แต่คำถามเชิงผลิตภัณฑ์ว่า "LiDAR 3.0 cm ลดได้อีกไหมด้วยข้อมูลที่เครื่องมีอยู่แล้ว" คุ้มที่จะตอบด้วยการทดลองเล็ก ๆ

Open3D `ScalableTSDFVolume` (legacy) ไม่รับ weight ต่อ pixel — ทุก `integrate()` บวก weight 1

## Options

| Option | Pros | Cons |
|---|---|---|
| A. mask อย่างเดียว (`depth.lidar_min_confidence`, มีอยู่แล้ว) | ไม่แตะ fusion | ทิ้งข้อมูล ไม่ใช่ถ่วงน้ำหนัก |
| B. **integrate ซ้ำตามระดับ weight**: pixel ที่ weight ≥ ℓ ถูก integrate 1 ครั้งต่อ ℓ | ได้ weighted average ที่ถูกต้องตามนิยาม TSDF สำหรับ weight จำนวนเต็ม; ไม่เปลี่ยน backend | ต้นทุน `max(w)` เท่า (LiDAR fusion ~1 s/ฉาก จึงไม่มีนัย) |
| C. ย้ายไป `o3d.t.geometry.VoxelBlockGrid` แล้ว implement weight เอง | ยืดหยุ่น, ทางไป GPU | เขียน kernel เอง; เปลี่ยน backend กลางทาง ทำให้ Exp1–4 ต้องรันใหม่เพื่อเทียบ |

## Decision
**B.** `TSDFFusion.integrate(..., weights: (H,W) int | None)` — `None` = พฤติกรรมเดิมทุกประการ
`confidence_weights(conf, table)` แปลง confidence → weight ด้วยตาราง `fusion.confidence_weights: [w0, w1, w2]`
(default `null` = ปิด; ผล Exp1–4 ไม่เปลี่ยน) `pipeline.py` เรียกสองบรรทัดนี้แล้วส่งต่อ ไม่มี Open3D

Exp5 (`configs/experiments/exp5_lidar_confidence.yaml`): `lidar` / mask ≥1 / mask =2 / weight [0,1,2] / weight [1,2,4]
บน 6 ฉาก — แถว `lidar` ต้องเท่ากับ Exp1 ทุกหลัก (regression check ของการเปลี่ยนแปลงนี้)

## Consequences
- ✅ ผลิตภัณฑ์ได้ knob เดียว (`fusion.confidence_weights`) โดยไม่แตะ backend
- ✅ `custom.py` ส่ง `confidence/<stem>.png` เข้าช่องเดียวกัน ⇒ ใช้กับ capture จริงได้ทันที
- ⚠️ weight เป็นจำนวนเต็มเท่านั้น และ pixel weight 0 = ทิ้ง
- ⚠️ ใช้ได้กับ source ที่มี `lidar_confidence` ใน `frame.extra` — กับ `gt`/`mono` จะเป็น None (ไม่ถ่วง) โดยอัตโนมัติ
- ❌ GPU TSDF ยังไม่ทำ: เครื่องพัฒนาเป็น Apple Silicon (Open3D CUDA ใช้ไม่ได้) และ Exp ทั้งหมด fusion < 2 s/ฉาก จึงไม่ใช่คอขวด
