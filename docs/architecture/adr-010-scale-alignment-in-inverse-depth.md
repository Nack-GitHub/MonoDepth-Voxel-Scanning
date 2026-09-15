# ADR-010: Fit scale/shift ในสเปซ inverse-depth (disparity) ไม่ใช่ depth

## Status
Accepted (2026-09-14) — ขยาย ADR-002 (interface `ScaleAligner` ไม่เปลี่ยน เปลี่ยนเฉพาะสเปซที่ fit)

## Context
ADR-002 กำหนดว่า aligner หา `(s, t)` ให้ `metric ≈ s·pred + t` และ `MonocularDepth` แปลง disparity → `1/x`
ก่อนส่งให้ aligner เพื่อให้ทุก aligner เห็น "depth-like map" เหมือนกัน

พอรันกับฉากจริง (42444474, ADR-009) ผล mono ออกมาแย่ผิดคาด:
mono + oracle_frame chamfer **15.2 cm** / F@5cm 0.34 ทั้งที่ oracle คือเพดานของโมเดล
สาเหตุ: Depth Anything V2 (และ MiDaS) เป็นโมเดล **affine-invariant ใน disparity** —
สิ่งที่โมเดลรับประกันคือ `disp_pred ≈ a·(1/depth) + b` ไม่ใช่ `depth_pred ≈ s·depth + t`
การ invert ก่อนแล้วค่อย fit affine ในสเปซ depth จึงเป็นโมเดลผิด: shift `b` ใน disparity
กลายเป็น distortion ที่ไม่ใช่ affine ใน depth และ least-squares ก็ดึงไปตามจุดไกล ๆ

## Options

| Option | Pros | Cons |
|---|---|---|
| A. fit ใน depth space (เดิม) | เรียบง่าย, ตรงกับ ADR-002 ตามตัวอักษร | โมเดลผิดสำหรับ DA-v2 / MiDaS; oracle 15 cm |
| B. **fit ใน inverse-depth**: `gt ≈ 1 / (s·(1/pred) + t)` | ตรงกับสิ่งที่โมเดล train มา; ถูกต้องสำหรับทั้ง disparity model และ metric model (s≈1, t≈0) | ต้อง invert 2 ครั้ง; ค่า `(s, t)` ตีความเป็นเมตรตรง ๆ ไม่ได้ |
| C. ให้ `MonocularDepth` ส่ง disparity ดิบออกมา แล้ว aligner รับ disparity | invert ครั้งเดียว | aligner ต้องรู้ว่า input เป็น depth หรือ disparity → ผูก source กับ aligner ซึ่งเป็นสิ่งที่ ADR-002 ตั้งใจเลี่ยง; GT/LiDAR ก็ต้องแปลงตาม |

## Decision
**B.** `_AffineAligner(space="inverse")` ใน `geometry/scale_align.py` — ทุก aligner ที่ไม่ใช่ identity
(`oracle_frame`, `per_scene`, `sparse_points`) สืบทอดจากตัวนี้ ค่า default คือ `inverse`;
`depth.align_space: depth` ยังเลือกได้สำหรับ ablation

`MonocularDepth` ยังคง invert disparity → depth-like ก่อนส่งออกตาม ADR-002 (aligner จึงไม่ต้องรู้จัก source)
aligner invert กลับเองภายในถ้า `space == "inverse"` — ต้นทุนคือการ invert ซ้ำ ซึ่งไม่มีนัยเทียบกับ inference

พ่วงมาด้วย: `PerSceneAligner` เลือก fit frames **กระจายทั่ว scan** (pipeline เป็นคนเลือก) แทนเอา N เฟรมแรก
เพราะ N เฟรมแรกมักมองมุมเดียว ทำให้ `(s, t)` overfit กับ depth range แคบ ๆ

## ผลบน 42444474 @ voxel 4 cm (Exp1)

| row | depth space (เดิม) | inverse (ใหม่) |
|---|---|---|
| mono_oracle (DA-v2 L) | 15.2 cm / F@5cm 0.34 | **5.4 cm / 0.72** |
| mono_scene (DA-v2 L, 10 fit frames) | 62.6 cm | **23.9 cm** |
| mono_metric (DA-v2 metric indoor, identity) | 53.2 cm | 53.2 cm (ไม่ผ่าน aligner) |

## Consequences
- ✅ ตัวเลข mono ในเปเปอร์สะท้อนความสามารถของโมเดล ไม่ใช่ความผิดพลาดของวิธี fit
- ✅ metric model ที่ผ่าน `oracle_frame` ก็ยังถูก (affine ใน inverse ครอบคลุม s≈1, t≈0)
- ⚠️ `(s, t)` ที่ log ไว้อยู่ในหน่วย 1/m — ห้ามอ่านเป็น "scale เป็นเมตร" ในเปเปอร์
- ⚠️ ผลใน `experiments/results/` ก่อน commit `4e867d8` fit ในสเปซ depth — ถูกรันทับหมดแล้ว ไม่มีค้าง
- ADR-002 ข้อ "aligner เห็นแค่ depth-like map" ยังจริงในเชิง interface; ADR นี้บอกว่าข้างในจะทำอะไรกับมัน
