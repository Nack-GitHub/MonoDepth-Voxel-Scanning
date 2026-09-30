# TODO: roomscan_web demo (SPEC.md, tasks/plan.md)

ทุก task: `make test` + `make lint` ผ่าน → commit แยก (ไม่ push)

## Phase 1: Foundation
- [x] **T1** FT-LiDAR-24 preset จาก server ถึง dropdown — `presets.py`, `GET /presets`, `POST /scans preset=` · S–M
  - Verify: `test_presets*`, dropdown บน `make web`
- [x] **T2** งานไม่หายเมื่อ restart + timestamps + ตัวจับเวลา — `job.json`, งานที่ค้างกลายเป็น failed · S–M · deps T1
  - Verify: `test_persist_across_restart`, `test_interrupted_job`, Ctrl-C แล้วเปิดใหม่ งานยังอยู่
- [ ] **T3** three.js 0.160.0 ในเครื่อง + แตก index.html เป็น ES modules (ไม่เปลี่ยนพฤติกรรม) · M · deps T1,T2 · ⚠️ ถามก่อนดาวน์โหลด
  - Verify: `test_static_vendor_served`, `test_no_cdn_in_index`, network log ไม่มี request ออกนอก localhost
  - [x] T3a แตก `index.html` เป็น `app.css` + `js/{api,viewer,panels,format,main}.js`, mount `/static` (importmap ยังชี้ CDN)
  - [ ] T3b vendor three.js 0.160.0 ลง `static/vendor/three/` + importmap ในเครื่อง + test 2 ตัวข้างบน — **รออนุญาตดาวน์โหลด**

### ☐ Checkpoint 1: synthetic upload → mesh → restart → ยังอยู่ (offline) · review กับผู้ใช้

## Phase 2: Core demo story
- [x] **T4** compare 1–3 ช่อง ใช้ renderer เดียว + scissor + กล้องร่วมกัน + ป้ายใต้ช่อง + เตือนเมื่อคนละห้อง · M · deps T3 · risk สูงสุด
  - Verify: screenshot 3 ช่องที่หมุนพร้อมกัน, resize/Retina ไม่เพี้ยน
- [ ] **T5** `meta.json` (export_capture) → `up`/`scene` ใน public() → Z-up อัตโนมัติ · S · deps T4,T2
  - Verify: `test_meta_json_up_and_scene`, `make capture-zip SCENE=47429736` แล้วใน zip มี `meta.json`
- [ ] **T6** `GET /scans/{id}/reference.ply` + Faro wireframe overlay + framing จาก reference · M · deps T4
  - Verify: `test_reference_endpoint`, screenshot overlay
- [ ] **T7** `errorcolor.py` (turbo 0–10 cm, cache) + `GET /scans/{id}/error.ply` + toggle + legend · M · deps T6
  - Verify: `test_errorcolor_*` (0 cm → turbo(0), 5 cm → turbo(0.5)), `test_error_endpoint` (409/404/cache)
- [ ] **T8** แผง metric 8 ค่า + badge Overview/Furniture/Renovation · S · deps T4
  - Verify: 0.584 → ✗, 0.8 → ~, 0.91 → ✓; ค่าตรงกับ metrics.json

### ☐ Checkpoint 2: demo ช่วง A–D ครบบน synthetic · review screenshot · ตัดสินใจว่าจะทำ Phase 3 ไหม

## Phase 3: Gallery
- [ ] **T9** `gallery.py` + `GET /gallery` + `/gallery/{exp}/{run}/{mesh,reference,error}.ply` (อ่านอย่างเดียว, allowlist, cache ใน outputs/) · M · deps T7
  - Verify: `test_gallery_*` (list, 404 traversal, ไม่มีไฟล์ถูกเขียนใต้ gallery root)
- [ ] **T10** แท็บ Gallery (ตารางห้อง × source) + ผสมกับ web jobs ใน compare + ป้าย "paper run" · M · deps T4,T7,T8,T9
  - Verify: 47429736 ได้ 54.1 / 12.4 / 3.1 cm ใต้ช่อง; 1280×720 ไม่ล้น
- [ ] **T11** `scripts/warm_web_cache.py` · S · deps T9
  - Verify: error เปิดได้ภายใน 3 วินาทีหลัง warm, รันซ้ำแล้วข้าม

### ☐ Checkpoint 3: SPEC §8 ข้อ 4–7 ผ่านบนข้อมูลจริง

## Phase 4: Demo readiness
- [ ] **T12** อัปเดต docs (architecture README, demo plan §1.3/§2.3) + ซ้อม offline 1280×720 + เช็คว่าตัวเลขของ web ต่างจาก gallery ไม่เกิน ±20% · S · deps ทั้งหมด · ⚠️ ถามก่อนรัน mono บนห้องจริง
  - Verify: SPEC §8 ครบ 1–10

### ☐ Complete: SPEC §8 ครบ · commit แยก ไม่ push · ผู้ใช้ review
