# ADR-005: การทดลองขับด้วย YAML + OmegaConf (ไม่ใช้ Hydra, ไม่ hardcode ใน script)

## Status
Accepted (2026-09-11)

## Context
มี 4 ชุดการทดลอง × 3 ฉาก × 3–5 run ≈ 40–60 runs ต้อง reproducible และตารางในเปเปอร์ต้องชี้กลับมาที่ config ได้

## Options

| Option | Pros | Cons |
|---|---|---|
| argparse ล้วน | ไม่มี dep | 20+ flags; sweep ต้องเขียน bash |
| **OmegaConf: `base.yaml` + `_base_` inheritance + dotlist override** | merge/override/serialize ครบใน dep เดียว; sweep file เป็น YAML อ่านง่าย | ไม่มี launcher (ไม่ต้องการ) |
| Hydra | launcher, multirun, logging | เปลี่ยน cwd, สร้าง `outputs/` เอง, learning curve — เกินความจำเป็นสำหรับ 1 คน 3 ฉาก |

## Decision
- `configs/base.yaml` เป็นค่าตั้งต้นเดียว; ทุก axis ของการทดลอง = 1 key
- `configs/depth/*.yaml` = preset สำหรับรัน 1 ครั้ง (`_base_: configs/base.yaml`)
- `configs/experiments/*.yaml` = sweep: `scenes × runs[overrides]`
- `roomscan run --set k=v` override จาก CLI; `config.yaml` ที่ resolve แล้วถูกเขียนลง run dir เสมอ

## Consequences
- ✅ ตารางในเปเปอร์ทุกตาราง = 1 ไฟล์ใน `configs/experiments/`
- ⚠️ `_base_` เป็น path จาก repo root — รันจาก root เสมอ (Makefile บังคับอยู่แล้ว)
