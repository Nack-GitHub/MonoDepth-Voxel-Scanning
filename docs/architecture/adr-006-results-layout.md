# ADR-006: ผลการทดลองเป็นไฟล์ flat ใน git — ไม่มี DB, ไม่มี experiment tracker

## Status
Accepted (2026-09-11)

## Decision
```
experiments/results/<experiment>/<scene>_<run>/
├── config.yaml     ← resolved config (commit)
├── metrics.json    ← ตัวเลขทั้งหมด + timing (commit)
└── mesh.ply        ← gitignored
```
`roomscan report` เดินไฟล์ `metrics.json` → `summary.csv` + `table.md` ต่อ experiment

## Rationale
- ~60 runs, ตัวเลขไม่กี่สิบตัวต่อ run — `git log` คือ history, `git diff metrics.json` คือ regression check
- W&B/MLflow เพิ่ม account/service/dep โดยไม่ได้อะไรที่ `pandas.read_json` ให้ไม่ได้

## Trade-offs
- ไม่มี UI เปรียบเทียบ run — ใช้ notebook ใน `notebooks/` แทน
- mesh ไม่อยู่ใน git → reproduce ต้องรันใหม่ (ยอมรับ; config อยู่ครบ)

## Revisit trigger
runs > ~300 หรือมีคนร่วมงานมากกว่า 1 คน
