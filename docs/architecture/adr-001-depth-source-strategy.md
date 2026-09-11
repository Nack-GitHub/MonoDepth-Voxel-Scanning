# ADR-001: DepthSource เป็น Strategy interface ที่สลับได้จาก config

## Status
Accepted (2026-09-11)

## Context
คำถามวิจัยคือ "เปลี่ยนแหล่ง depth โดย pipeline เหมือนเดิมทุกอย่าง ผลต่างแค่ไหน"
และ MVP ในอนาคตต้องรองรับ LiDAR/ARCore/mono สลับกันตามเครื่องลูกค้า
ถ้าเขียน pipeline ผูกกับ GT depth ก่อนแล้วค่อย "แปะ" โมเดลทีหลัง จะได้โค้ดสองชุดที่เทียบกันไม่แฟร์

## Options

| Option | Pros | Cons | Complexity |
|---|---|---|---|
| A. `if source == "gt": ... elif "mono": ...` ใน pipeline | เขียนเร็ว | pipeline รู้ทุก source, เพิ่ม source = แก้ orchestrator, ablation ไม่สะอาด | Low now, High later |
| B. **`DepthSource` ABC + registry, pipeline เห็นแค่ `get_depth(frame)`** | ตัวแปรเดียวเปลี่ยน, เพิ่ม source = ไฟล์ใหม่ 1 ไฟล์, code เปเปอร์ = code product | ต้องคิด interface ให้ครบตั้งแต่แรก | Low |
| C. Plugin system แบบ entry_points | ยืดหยุ่นสุด | over-engineering สำหรับ 4 source | Medium |

## Decision
**B.** `roomscan.depth_sources.base.DepthSource` มี `name`, `is_metric`, `get_depth(frame: Frame) -> np.ndarray` (metres, float32, 0/NaN = invalid)

รับ `Frame` ทั้งก้อน (ไม่ใช่ `frame_idx, rgb` อย่างในร่างแรก) เพราะ source ต่างชนิดต้องการ input ต่างกัน: GT อ่าน `gt_depth`, mono อ่าน `rgb`, LiDAR/ARCore อ่าน `extra[...]` — ถ้า signature ระบุ input ตายตัว จะต้องแก้ทุกครั้งที่เพิ่ม source

## Trade-offs
- `Frame` โหลดทุก field แม้ source ไม่ใช้ (GT depth ถูกอ่านทั้งที่รัน mono) — ยอมรับได้ เพราะต้องใช้ทำ 2D metrics อยู่ดี
- `is_metric` เป็น property ที่ source ต้อง "รู้ตัวเอง" — ถ้าใส่ผิด pipeline จะ align ผิด → มี `_check_consistency` ดักไว้

## Consequences
- ✅ Exp1 คือ loop เดียวเปลี่ยน config
- ✅ วันที่ LiDAR แพร่หลาย เปลี่ยน `depth.source=lidar` จบ
- ⚠️ ห้ามให้ DepthSource ทำ scale alignment เอง (ดู ADR-002)

## Revisit trigger
ถ้ามี source ที่ต้องการ *หลายเฟรม* พร้อมกัน (multi-view stereo, video depth) — interface `get_depth(frame)` จะไม่พอ ต้องเพิ่ม `get_depth_batch(frames)`
