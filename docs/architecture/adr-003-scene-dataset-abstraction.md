# ADR-003: SceneDataset abstraction — ไม่ hardcode dataset เดียว

## Status
Accepted (2026-09-11)

## Context
เดิมวางแผนใช้ ScanNet, ตอนนี้เปลี่ยนเป็น ARKitScenes (ADR-009) — การเปลี่ยนนี้เกิดขึ้น *ก่อนเขียน loader บรรทัดแรก* ด้วยซ้ำ
และ MVP ต้องกินวิดีโอห้องจริง + pose จาก COLMAP/ARKit ซึ่งไม่ใช่ทั้งสองอย่าง

## Decision
`dataio/base.py::SceneDataset` — `intrinsics`, `__len__`, `frame(idx) -> Frame`, `frames(stride, max_frames)` (ข้าม pose ที่ไม่ finite ให้), `gt_mesh()`
Pipeline ไม่รู้จักชื่อ dataset; `build_dataset(cfg)` map ผ่าน registry

`Frame.extra: dict` เป็นช่องทางส่ง sensor side-channel (ARKit depth, confidence, sparse points) โดยไม่ต้องแก้ `Frame` ทุกครั้ง

## Trade-offs
- `extra` เป็น dict ไม่ typed — แลกกับการไม่ต้องแก้ core type เมื่อเพิ่ม sensor; ถ้า key ชนกันให้ document ใน loader
- `gt_mesh()` อาจคืน "reference mesh ที่สร้างขึ้น" ไม่ใช่ GT แท้ (ARKitScenes) — ชื่อเมธอดคงไว้ แต่ความหมายอยู่ที่ ADR-009

## Consequences
- ✅ เปลี่ยน ScanNet→ARKitScenes กระทบแค่ `dataio/` + config
- ✅ ห้องจริงตอน MVP = `dataio/custom.py` ไฟล์เดียว
