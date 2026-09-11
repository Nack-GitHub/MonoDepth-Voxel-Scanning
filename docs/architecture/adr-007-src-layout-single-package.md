# ADR-007: `src/roomscan` package เดียว, `pip install -e .`, ไม่แยก repo

## Status
Accepted (2026-09-11)

## Options

| Option | Pros | Cons |
|---|---|---|
| ไฟล์ `.py` ใต้ `src/` ไม่มี package (ร่างแรก) | ง่าย | `sys.path` hack ใน notebook/script; import ชนกัน |
| **`src/roomscan/` + pyproject, editable install** | `import roomscan` ได้ทุกที่ (tests, notebooks, Colab); entrypoint `roomscan` CLI | ต้อง `pip install -e .` ครั้งเดียว |
| แยก repo `roomscan-core` / `roomscan-experiments` | clean product boundary | เร็วไปสำหรับสิ่งที่ยังไม่มี user |

## Decision
Package เดียว optional deps แยก: `[mono]` (torch/transformers) ติดตั้งเฉพาะเมื่อรัน mono — GT pipeline (Phase 0–1) และ tests ไม่ต้องมี torch
Registry ใช้ lazy import (`_registry.py`) เพื่อรักษาคุณสมบัตินี้

## Consequences
- ✅ Colab: `!pip install -e ".[mono]"` แล้วใช้ CLI ได้เลย
- ✅ MVP web layer ในอนาคต = package `roomscan_web/` ที่ `import roomscan.pipeline` — ไม่ต้องย้ายอะไร
