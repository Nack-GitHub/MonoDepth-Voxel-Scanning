# ADR-014: 2D metrics protocol 2 (mask ตาม GT, clip prediction) + ชุดเทสต์ 2D จาก Validation fold

## Status
Accepted (2026-09-25) — แก้นิยามใน `evaluation/metrics_2d.py` (เดิมไม่มี ADR); เพิ่ม `eval2d.py` + Exp7;
ไม่แตะ `pipeline.py`

## Context
1. **นิยาม 2D เดิม (protocol 1) ตัด pixel ตามค่า *prediction*** — `depth_metrics` คัด pixel ที่ทั้ง GT **และ pred**
   อยู่ใน `[depth_min, depth_trunc]` (0.1–5 m) โมเดลที่ทายไกลเกิน (เช่น DA-v2 Metric pretrained ที่ทายไกลเกิน 1.2–1.45× ต่อห้อง และ Depth Pro)
   จึงถูกตัด pixel ที่ผิดหนักที่สุดทิ้ง และแต่ละแถวถูกวัดบนชุด pixel ไม่เท่ากัน
   นิยามมาตรฐาน (NYU/KITTI, Eigen et al.) คือคัดตาม GT แล้ว **clip** prediction เข้าช่วง
2. **ข้อสรุปหลักของเปเปอร์ย้ายมาอยู่ที่ "fine-tune ด้วยครู LiDAR ดีกว่า pretrained จริงไหม"** (R3 vs pretrained,
   วัดกับ Faro ทั้งคู่) แต่มีห้องเทสต์ 6 ห้องเท่านั้น (ห้องเดียวที่มี reference mesh) ซึ่ง ADR-013 เขียนไว้เป็น limitation
   การวัด 2D ใช้แค่ Faro depth (`highres_depth`) ของ dataset เอง ไม่ต้องมี mesh จึงขยายได้ถูก

## Options

| Option | Pros | Cons |
|---|---|---|
| A. คง protocol 1 | ตัวเลขเดิมไม่ขยับ | ลำเอียงเข้าข้างโมเดลที่ทายไกลเกิน; ไม่ใช่นิยามที่ผู้อ่านคาด |
| B. **mask ตาม GT + pred ที่ "มีค่า" (> 0, finite), แล้ว clip pred** | ตรงมาตรฐาน; ทุกแถวที่ dense ได้ชุด pixel เดียวกัน; pixel ที่ source ไม่ให้ค่า (LiDAR ที่ถูก mask ด้วย confidence) ไม่ถูกลงโทษเป็น 0.1 m | ต้องคำนวณ 2D ของทุก run ใหม่ |
| C. mask ตาม GT อย่างเดียว, pred = 0 นับเป็น 0.1 m | "ลงโทษ" แถวที่ไม่ครอบคลุม | แถว LiDAR/confidence กลายเป็นวัด coverage แทนความแม่น — ซ้ำกับ completeness ของ 3D |

| Option (เทสต์เพิ่ม) | Pros | Cons |
|---|---|---|
| D. ไม่เพิ่ม | ไม่มีงาน | ข้อสรุปหลักยืนบน 6 ห้อง |
| E. **2D-only บนห้องจาก fold Validation** ที่ไม่ใช่ V และไม่ซ้ำ visit กับชุดใดใน `splits.yaml` | ห้องมากขึ้น 3× โดยไม่ต้องมี mesh; ใช้ GT ดิบของ dataset | ไม่มีตัวเลข 3D บนห้องเหล่านี้ |
| F. สร้าง reference mesh ให้ห้องใหม่ด้วย | ได้ 3D ด้วย | ต้องโหลด Faro ทุกเฟรม + mesh ~1–2 GB/ห้อง; ดิสก์ Mac ไม่พอ |

## Decision
**B + E.**

- `depth_metrics`: `m = isfinite(gt) & min ≤ gt ≤ max & isfinite(pred) & pred > 0`; `p = clip(pred[m], min, max)`
  `Metrics2D.protocol_2d = 2` ถูกเขียนลง `metrics.json` ทุกไฟล์ ⇒ run เก่าที่ยังไม่ re-eval เห็นได้จากคอลัมน์ว่าง
- `roomscan reeval2d <run dirs>` คำนวณ `metrics_2d` ใหม่ในที่เดิม (3D ไม่แตะ) ด้วยเฟรม/aligner fit/grid/mask
  ชุดเดียวกับ `ReconstructionPipeline.run()` และแชร์ forward pass ระหว่าง run ที่ใช้โมเดลเดียวกันบนห้องเดียวกัน
  (`src/roomscan/eval2d.py`; `tests/test_eval2d.py` ยืนยันว่าได้ตัวเลขเท่ากับ pipeline เต็ม)
- `roomscan eval2d <experiment>` = `sweep` ที่ไม่ fuse ไม่ต้องมี mesh — ใช้กับ **Exp7** (`exp7_valfold_2d.yaml`):
  - 20 ห้องจาก fold Validation (`configs/eval/valfold_2d.yaml`, `scripts/select_valfold_2d.py`, seed 0):
    scan 30–150 s, `highres_depth` ≥ 40 MB, หนึ่ง video ต่อ visit; ตัด visit ที่อยู่ใน `splits.yaml` (test/val/train)
    และ visit ที่โผล่ใน fold Training ออก
  - โหลดด้วย `scripts/download_valfold_2d.py`: Faro ทุก 2 เฟรม (≈ 2 fps) และภาพ `vga_wide`/LiDAR เฉพาะเฟรมที่ใกล้
    Faro ที่สุด (~250 MB/ห้อง แทน ~1.5 GB)
  - แถว: `mono_metric` (pretrained), `mono_ft_lidar_all` (R3), `lidar` (อ้างอิง: ครูที่ R3 เห็น)
- ห้องใน Exp7 ไม่ถูกใช้เลือกอะไรเลย (checkpoint ถูกเลือกไปแล้วบน V ตาม ADR-013)

## Consequences
- ✅ ตัวเลข 2D ทุกตารางในเปเปอร์นิยามเดียว (protocol 2); `summary.csv` มีคอลัมน์ `protocol_2d`
- 📊 ผลจริงของการเปลี่ยน (re-eval 213 run, 2026-09-26): ทุกแถวของ Exp1–6 ขยับ AbsRel < 0.001 (pretrained 0.382 → 0.382, R3 0.134 → 0.134)
  ยกเว้น **Depth Pro 1.043 → 2.007** (ทายไกลเกิน 5 m บ่อย = กรณีที่ protocol 1 ซ่อน); การวิเคราะห์ scale ต่อเฟรม (§6): scale-only ในสเปซ depth ของ
  DA-v2 L 0.289 → 0.331; run พัฒนาเก่า (`phase2_mono`, `exp0`) ขยับมากกว่าเพราะถูกคำนวณด้วยโค้ดก่อนแก้การหมุนภาพ — ไม่อยู่ในเปเปอร์
- 📊 Exp7 (20 ห้อง, 2,090 เฟรม): AbsRel pretrained 0.410 → R3 0.130, ดีขึ้น 19/20 ห้อง (Wilcoxon p = 3.8×10⁻⁶); ห้องที่ไม่ดีขึ้น 48018730 (0.275 → 0.281)
- ✅ ข้อสรุป "fine-tune ด้วยครู LiDAR ดีกว่า pretrained" มีหลักฐาน 2D บน 26 ห้อง (6 + 20) แทน 6
- ⚠️ `experiments/training/RESULTS.md` (val ตอนเทรน) ยังเป็น protocol 1 — เป็นตัวเลขเลือก checkpoint ไม่ใช่ผล และ
  ฉาก V ไม่อยู่บน Mac; ไม่ re-run
- ⚠️ Exp7 เป็น 2D อย่างเดียว — ไม่บอกคุณภาพ mesh บนห้องเหล่านั้น
