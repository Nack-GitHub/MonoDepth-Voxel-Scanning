# ADR-002: แยก ScaleAligner ออกจาก DepthSource

## Status
Accepted (2026-09-11)

## Context
โมเดล mono ส่วนใหญ่ให้ relative depth ต้องหา `s, t` ให้ `metric ≈ s·pred + t`
มีอย่างน้อย 3 วิธี (oracle ต่อเฟรม / ครั้งเดียวต่อฉาก / จาก sparse SLAM points) และแต่ละวิธีคือ row ในตาราง Exp1
ช่องว่างระหว่าง oracle กับ per-scene คือ "ราคาของการไม่รู้สเกล" — เป็นผลการทดลองที่มีค่า

## Options

| Option | Pros | Cons |
|---|---|---|
| A. `MonocularDepth(align="oracle")` — align อยู่ใน source | ไฟล์น้อย | source × aligner ถูกผูกกัน; ทดสอบ aligner แยกไม่ได้; LiDAR ที่มี drift ก็ align ไม่ได้ |
| B. **`ScaleAligner` เป็น stage แยก ระหว่าง source กับ fusion** | grid `{source} × {aligner}` สะอาด; unit-test ได้ด้วย array สังเคราะห์ | เพิ่ม abstraction 1 ชั้น |

## Decision
**B.** `geometry/scale_align.py`: `ScaleAligner.fit(frames, preds)` (optional, one-off) + `align(pred, frame)`
DepthSource คืนค่าดิบจากโมเดล **ห้าม** align เอง

Pipeline บังคับความสอดคล้อง: `is_metric=True ⇒ identity`, `is_metric=False ⇒ ไม่ใช่ identity` (fail fast ตอน construct)

## Trade-offs
- `PerSceneAligner.fit()` ต้อง materialize เฟรมชุดแรกก่อน loop หลัก → pipeline โหลด `list(frames)` ล่วงหน้า; ที่ 1–5k เฟรม/ฉาก ยังไหว ถ้าไม่ไหวค่อยเปลี่ยนเป็น two-pass
- Disparity→depth inversion: อยู่ใน `MonocularDepth.get_depth` (source รู้ว่าตัวเองคืน disparity) ไม่ใช่ใน aligner — aligner เห็นแค่ "depth-like map"

## Consequences
- ✅ "Mono + oracle" กับ "Mono + per-scene" ต่างกันแค่ `depth.aligner`
- ✅ `sparse_points` aligner = ทางไป MVP โดยไม่แตะ source
