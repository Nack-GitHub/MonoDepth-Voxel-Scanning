# Implementation Plan: roomscan_web สำหรับ demo วันนำเสนอ IS

อ้างอิง: [`SPEC.md`](../SPEC.md) (2026-09-30), `docs/notes/demo_presentation_plan.md` §1–2
สถานะ: **ทำแล้วบน branch `feat/web-demo` (2026-09-30)** ยกเว้นการซ้อมกับข้อมูลจริงใน Checkpoint 3 และ T12 · ความคืบหน้ารายข้ออยู่ใน [`todo.md`](todo.md) และผลการตรวจอยู่ใน `SPEC.md` §10
ช่อง `[ ]` ใต้แต่ละ task ด้านล่างคงไว้ตามแผนเดิม ไม่ได้ติ๊กทีละข้อ

## Overview

ขยาย `src/roomscan_web/` จาก viewer ที่แสดง mesh ได้ทีละก้อน ให้เป็นเครื่องมือ demo ที่เล่าเรื่องของเปเปอร์ได้ครบ ได้แก่
- preset FT-LiDAR-24
- งานไม่หายเมื่อ restart server
- เปรียบเทียบ 3 ช่องที่ใช้กล้องร่วมกัน
- สี error เทียบกับ Faro
- Faro overlay
- แผง metric พร้อม badge use case
- gallery ของผลจาก Exp1/Exp6
- ทำงานได้แบบ offline

งานทั้งหมดอยู่ใน `roomscan_web/`, `scripts/` และ `tests/test_web.py` ไม่แตะ `pipeline.py`

## Architecture Decisions

- **Presets อยู่ฝั่ง server** (`presets.py` + `GET /presets`) ให้ label มีที่เดียว, test ได้, และ `job.json` เก็บ key ไว้ติดป้ายหลัง restart
- **Persist ด้วยไฟล์** `results/web/<id>/job.json` ไม่ใช้ DB (ADR-006) งานที่ค้างตอน restart → `failed` ไม่ requeue (ป้องกันงาน mono 7 นาทีเริ่มเองตอนเปิดเครื่องบนเวที)
- **Item abstraction ฝั่ง UI:** web job กับ gallery run เป็น "item" รูปเดียวกัน `{key, origin, label, scene, up, metrics, urls:{mesh, reference, error}}` → viewer/compare/panel ไม่ต้องรู้ว่ามาจากไหน ฝั่ง server ก็ใช้ helper เดียวกันเสิร์ฟ `mesh/reference/error.ply`
- **errorcolor ใช้ `_sample`/`_nn` ของ `roomscan.evaluation.metrics_3d`** (seed/จำนวนจุดเดียวกับ `per_point_error`) แต่คำนวณที่ vertex ของ mesh แทนจุด sample เพื่อระบายสี mesh ได้ตรง ๆ; turbo LUT ทำเองด้วย numpy (ไม่ดึง matplotlib เข้า `.[web]`)
- **Viewer: renderer เดียว + scissor/viewport**, camera/OrbitControls ตัวเดียว, framing ครั้งเดียวจาก reference → ความ "พอง" ของ pretrained ยังเห็นได้
- **Gallery reference** มาจาก `build_dataset(cfg.dataset).gt_mesh()` ตาม `config.yaml` ของ run (แบบเดียวกับ `make_figures.py`) แล้ว cache เป็น `outputs/web/cache/<exp>/<scene>/reference.ply` → test ได้ด้วย synthetic, ไม่ต้องเดา path ของ ARKitScenes
- **Refactor `index.html` → ES modules เป็น task แยกที่ไม่เปลี่ยนพฤติกรรม** ก่อนเพิ่มฟีเจอร์ UI เพื่อให้ task ถัด ๆ ไปแก้ไฟล์เล็กและ review ง่าย

## Dependency Graph

```
T1 presets ─────────────┐
T2 persist + timestamps ┤
T3 vendor + ES modules ─┼─► T4 compare 3 ช่อง ─► T5 meta.json/Z-up
                        │         │
                        │         ├─► T6 reference.ply + overlay ─► T7 error.ply + toggle
                        │         └─► T8 metric panel + badges
                        │
                        └─► T9 gallery API ─► T10 gallery UI (ต้องมี T4,T7,T8) ─► T11 warm cache
                                                                                   │
                                                                     T12 docs + demo readiness
```

ทำพร้อมกันได้: T1 ∥ T2 ∥ T3 (คนละไฟล์หลัก ยกเว้น `app.py` ที่แก้คนละส่วน) · T9 ∥ T5–T8 (backend ล้วน)
ต้องเรียง: T3 ก่อนงาน UI ทุกข้อ, T6 ก่อน T7 (error ต้องมี reference), T4 ก่อน T10

## Task List

### Phase 1: Foundation

#### Task 1: FT-LiDAR-24 preset จาก server ถึง dropdown
**Description:** สร้าง `presets.py` (4 preset ตาม SPEC §4.1, `mono_sparse` = Large + ป้าย upper bound), `GET /presets`, `POST /scans` รับ field `preset` (merge overrides, เก็บ key ใน Job) และให้ dropdown ใน `index.html` สร้างจาก `/presets` แทน `PRESETS` ที่ hardcode
**Acceptance:**
- [ ] `/presets` คืน 4 key ตามลำดับ `lidar, ft_lidar_24, mono_metric, mono_sparse` พร้อม label ตามเปเปอร์
- [ ] preset ที่ไม่รู้จัก → 400; overrides ของทุก preset ⊆ `ALLOWED_OVERRIDES`
- [ ] รายการงานแสดง label ของ preset แทน job id อย่างเดียว
**Verify:** `make test` (test ใหม่ `test_presets*`), `make lint`, เปิด `make web` ดู dropdown
**Dependencies:** none · **Files:** `presets.py`(ใหม่), `app.py`, `jobs.py`, `static/index.html`, `tests/test_web.py` · **Scope:** M

#### Task 2: งานไม่หายเมื่อ restart + timestamps + ตัวจับเวลา
**Description:** `jobs.py` เขียน `job.json` ทุกครั้งที่ status เปลี่ยน และมี `created_at/started_at/finished_at` `JobRunner.__init__` จะโหลดงานกลับมา โดยงาน `queued/running` จะกลายเป็น `failed: interrupted by server restart` ฝั่ง UI แสดง `running mm:ss` นับจาก `started_at` และแสดง ETA = `eta_s_per_frame × n_frames` จาก preset
**Acceptance:**
- [ ] สร้าง `create_app` ใหม่บน work dir เดิมแล้วงาน `done` กลับมาครบ พร้อม result, mesh และ preset
- [ ] งานที่ค้างอยู่กลายเป็น `failed` และไม่ถูกรันซ้ำ
- [ ] reload หน้าระหว่างที่งานรันอยู่ ตัวจับเวลายังนับต่อจากเวลาจริง
**Verify:** `make test` (`test_persist_across_restart`, `test_interrupted_job`), manual: รัน LiDAR job → Ctrl-C server → `make web` → งานยังอยู่
**Dependencies:** T1 (เก็บ preset key) · **Files:** `jobs.py`, `static/index.html`, `tests/test_web.py` · **Scope:** S–M

#### Task 3: three.js ในเครื่อง + แตก index.html เป็น ES modules (ไม่เปลี่ยนพฤติกรรม)
**Description:** ⚠️ **ถามก่อนดาวน์โหลด** `three.module.js`, `OrbitControls.js`, `PLYLoader.js` เวอร์ชัน 0.160.0 จาก jsdelivr ไปไว้ที่ `static/vendor/three/` + ไฟล์ `VERSION` จากนั้น mount `StaticFiles` ที่ `/static` แก้ importmap และย้าย CSS/JS ออกจาก `index.html` ไปเป็น `app.css`, `js/{api,viewer,panels,format,main}.js`
**Acceptance:**
- [ ] หน้าเว็บทำงานเหมือนเดิมทุกอย่าง (อัปโหลด, รายการ, ดู mesh, Z-up)
- [ ] ไม่มี URL ภายนอกใน `index.html` และ network log ตอน offline ไม่มี request ออกนอก localhost
**Verify:** `make test` (`test_static_vendor_served`, `test_no_cdn_in_index`), browser pane: เปิดหน้า → ดู mesh ของงานเดิม → `read_network_requests`
**Dependencies:** T1, T2 (แก้ `index.html` เดียวกัน ให้ทำหลังเพื่อไม่ต้อง merge) · **Files:** `app.py`, `static/index.html`, `static/app.css`, `static/js/*.js`(ใหม่), `static/vendor/three/*`, `tests/test_web.py` · **Scope:** M (ไฟล์เยอะแต่เป็นการย้ายโค้ด)

### Checkpoint 1: Foundation
- [ ] `make test` + `make lint` ผ่าน
- [ ] อัปโหลด synthetic capture ด้วย preset `lidar` ได้ → mesh ขึ้น → restart → ยังอยู่ (offline)
- [ ] commit แยก 3 ก้อน · **review กับผู้ใช้ก่อนเริ่ม Phase 2**

### Phase 2: Core demo story (demo ช่วง A–D บน web jobs)

#### Task 4: โหมดเปรียบเทียบ 1–3 ช่องที่ใช้กล้องร่วมกัน
**Description:** เพิ่ม checkbox "compare" ในรายการ (เลือกได้สูงสุด 3 งาน) ให้ `viewer.js` ใช้ renderer เดียววาดหลาย viewport ด้วย `setScissor/setViewport` และให้ทุกช่องใช้ camera กับ OrbitControls ร่วมกัน framing ตั้งครั้งเดียวจากช่องที่เป็น LiDAR ถ้าไม่มีก็ใช้ช่องแรก (reference ยังไม่มีจนกว่าจะถึง T6) ใต้แต่ละช่องมีป้าย label + Chamfer + "demo run" และถ้า `scene` ต่างกันให้ขึ้นแถบเตือน ส่วนนี้ใช้ item abstraction เดียวกับที่ T10 จะใช้ต่อ
**Acceptance:**
- [ ] เลือก 3 งานแล้วได้ 3 ช่องที่หมุนและซูมพร้อมกัน ขนาดจริงของ mesh ไม่ถูก normalize
- [ ] เลือกงานที่ 4 ไม่ได้ และถ้ายกเลิกให้เหลือ 1 งาน ต้องกลับเป็นช่องเดียว
- [ ] resize หน้าต่างแล้ว viewport ต้องไม่เพี้ยน รวมถึงบนจอ Retina (อย่าลืม comment เรื่อง `updateStyle` ในโค้ดเดิม)
**Verify:** browser pane: รัน synthetic 2–3 งานด้วย voxel ต่างกัน → compare → screenshot; `make test` ยังผ่าน
**Dependencies:** T3 · **Files:** `js/viewer.js`, `js/panels.js`, `js/main.js`, `app.css` · **Scope:** M (**risk สูงสุดฝั่ง UI — ทำก่อน**)

#### Task 5: meta.json → ตั้ง Z-up และรู้ว่าเป็นห้องไหนโดยอัตโนมัติ
**Description:** ให้ `export_capture.py` เขียน `meta.json` (`source, scene, up, stride`) และให้ `jobs.py` อ่านไฟล์นี้ใส่ `up`/`scene` ลงใน `public()` (ถ้าไม่มีไฟล์ใช้ `up:"y"`, `scene:null`) ฝั่ง viewer ใช้ `up` เป็นค่าตั้งต้นของ Z-up แต่ละ item แต่ยังติ๊กเปลี่ยนเองได้ ส่วน compare warning (T4) ใช้ `scene` นี้
**Acceptance:**
- [ ] capture ที่มี `meta.json` → `public()` ได้ `up:"z"` กับ scene ที่ถูกต้อง และ pipeline ยังรันผ่าน
- [ ] capture ที่ไม่มี `meta.json` → พฤติกรรมเหมือนเดิม
**Verify:** `make test` (`test_meta_json_up_and_scene`), `make capture-zip SCENE=47429736` แล้วดูว่ามี `meta.json` ใน zip
**Dependencies:** T4 (warning), T2 · **Files:** `scripts/export_capture.py`, `jobs.py`, `js/main.js`, `tests/test_web.py` · **Scope:** S

#### Task 6: `reference.ply` endpoint + Faro overlay
**Description:** เพิ่ม `GET /scans/{id}/reference.ply` ที่อ่านจาก `scene_dir/reference.ply` และส่ง `has_reference` ใน `public()` ฝั่ง toolbar มีปุ่ม "Faro overlay" วาด reference เป็น wireframe สีเทาโปร่งใสในทุกช่อง และเปลี่ยน framing ของ T4 ให้ใช้ reference ก่อนเสมอ ถ้าไม่มี reference ให้ปิดปุ่มนี้และมี tooltip บอก
**Acceptance:**
- [ ] 200 + `ply` เมื่อมีไฟล์, 404 เมื่อไม่มี, 404 เมื่อ id ไม่มีอยู่
- [ ] overlay ซ้อนตรงกับ mesh ของ LiDAR (เพราะใช้ frame เดียวกัน) และถูกหมุนตาม Z-up ด้วย
**Verify:** `make test` (`test_reference_endpoint`), browser pane: เปิด overlay บน synthetic job แล้ว screenshot
**Dependencies:** T4 · **Files:** `app.py`, `jobs.py`, `js/viewer.js`, `js/panels.js`, `tests/test_web.py` · **Scope:** M

#### Task 7: สี error เทียบ Faro (`error.ply`) + toggle + legend
**Description:** สร้าง `errorcolor.py` โดยให้
- `vertex_error(pred, ref)` ใช้ `_sample(ref, 200k, seed=1)` แล้ว `_nn` จาก vertex ของ pred
- `turbo(x)` เป็น LUT ที่ทำด้วย numpy
- `write_error_ply(mesh_path, ref_path, out_path)` ต้องทิ้ง cache เก่าเมื่อ mtime ของไฟล์ต้นทางใหม่กว่า

endpoint `GET /scans/{id}/error.ply` ตอบ 409 ถ้างานยังไม่ `done` และ 404 ถ้าไม่มี reference cache เก็บไว้ที่ `results/web/<id>/error.ply` ฝั่ง UI มีปุ่ม "Color: real / error" ที่มีผลกับทุกช่อง พร้อม legend turbo 0–10 cm
**Acceptance:**
- [ ] ถ้า mesh = reference ต้องได้ error ≈ 0 (สี ≈ turbo(0)) และถ้าเลื่อนไป 5 cm ต้องได้สี ≈ turbo(0.5)
- [ ] จำนวน vertex ต้องเท่ากับ `mesh.ply` และเรียกครั้งที่สองต้องไม่คำนวณใหม่ (mtime ของ cache ไม่เปลี่ยน)
- [ ] legend อ่านออกบนพื้นทั้ง light และ dark
**Verify:** `make test` (`test_errorcolor_*`, `test_error_endpoint`), browser pane: สลับสีบน synthetic job แล้ว screenshot
**Dependencies:** T6 · **Files:** `errorcolor.py`(ใหม่), `app.py`, `js/viewer.js`, `js/panels.js`, `tests/test_web.py` · **Scope:** M

#### Task 8: แผง metric แบบเต็ม + badge use case
**Description:** แผงของ item ที่ focus อยู่ (item ที่คลิก หรือช่องที่ hover) แสดง Chamfer, Accuracy, Completeness, F@5, R@2/5/10 และ `timing.depth/fusion/total` มี badge Overview (R@10), Furniture (R@5), Renovation (R@2) ใช้เกณฑ์ ≥0.9 ✓ / 0.75–0.9 ~ / <0.75 ✗ พร้อมหมายเหตุว่า "thresholds are illustrative" ส่วน item ที่ไม่มี `metrics_3d` ให้ขึ้นว่า "no Faro reference — no 3D metrics"
**Acceptance:**
- [ ] ค่าที่แสดงตรงกับ `metrics.json` ถึงทศนิยมหนึ่งตำแหน่ง (หน่วยเป็น cm)
- [ ] ถ้า R = 0.584 ต้องได้ ✗, 0.8 ได้ ~, 0.91 ได้ ✓ (เช็คใน `format.js`)
**Verify:** browser pane กับ synthetic job, เช็คเกณฑ์ badge ด้วย `javascript_tool` import `format.js`
**Dependencies:** T4 · **Files:** `js/format.js`, `js/panels.js`, `app.css` · **Scope:** S

### Checkpoint 2: Core story บน web jobs
- [ ] `make test` + `make lint` ผ่าน
- [ ] ใช้ synthetic ทำ demo ช่วง A–D ได้ครบ: อัปโหลด → compare → สี error → overlay → metric panel
- [ ] ยังไม่มี request ออกนอก localhost
- [ ] **review กับผู้ใช้**: ดู screenshot และตัดสินว่าจะทำ Phase 3 หรือจะหยุดแค่นี้แล้วใช้ web jobs จริงแทน gallery

### Phase 3: Gallery (ผลจากเปเปอร์ เปิดได้ครบ 6 ห้อง)

#### Task 9: Gallery API (อ่านอย่างเดียว)
**Description:** สร้าง `gallery.py` ให้มี
- `list_runs(root, exps)` ที่ไล่หา `<exp>/<scene>_<run>/` ที่มี `mesh.ply` และ `metrics.json` แล้วใส่ label ตาม `RUN_LABELS` ถ้า run ซ้ำให้ใช้ของ exp6 ก่อน
- `resolve_run()` ตาม SPEC §5
- `reference_for(run_dir, cache_dir)` ที่สร้าง reference ด้วย `build_dataset(cfg.dataset).gt_mesh()` แล้ว cache ต่อห้อง

endpoint ที่ต้องมีคือ `GET /gallery` และ `GET /gallery/{exp}/{run}/{mesh,reference,error}.ply` โดย error ให้ reuse `errorcolor` และ cache ไว้ใต้ `outputs/web/cache/` ตั้งค่าได้ด้วย env `ROOMSCAN_GALLERY_ROOT` และ `ROOMSCAN_GALLERY_EXPS`
**Acceptance:**
- [ ] ใน tmp gallery root ที่มี synthetic 2 run ต้องได้ 2 รายการที่มี label, metrics และ `origin:"paper"`
- [ ] `..`, exp ที่อยู่นอก allowlist และ run ที่ไม่ตรง regex ต้องได้ 404
- [ ] หลังเรียกทุก endpoint แล้ว ต้องไม่มีไฟล์ใหม่ถูกเขียนใต้ gallery root
**Verify:** `make test` (`test_gallery_*`), `curl localhost:8765/gallery | head` บนเครื่องจริงต้องเห็น 47429736 ครบทุก source
**Dependencies:** T7 (errorcolor) · ทำขนานกับ T8 ได้ · **Files:** `gallery.py`(ใหม่), `app.py`, `tests/test_web.py` · **Scope:** M

#### Task 10: แท็บ Gallery + ผสมกับ web jobs ในโหมดเปรียบเทียบ
**Description:** sidebar มี 2 แท็บคือ Scans และ Gallery แท็บ Gallery เป็นตารางห้อง × source ที่แต่ละช่องแสดง Chamfer ในหน่วย cm และมี checkbox "compare" ที่นับรวมกับ web jobs (สูงสุด 3 รายการ) ป้ายใต้ช่องแสดง "paper run" ส่วน error/overlay/metric panel ใช้กับ item ของ gallery ได้เหมือนกันผ่าน item abstraction
**Acceptance:**
- [ ] เลือก 47429736 pretrained + FT-LiDAR-24 + LiDAR แล้วต้องได้ Chamfer 54.1 / 12.4 / 3.1 cm ใต้แต่ละช่อง
- [ ] ผสม LiDAR web job ของห้องเดียวกันกับ FT จาก gallery ได้ และต้องไม่มีคำเตือนเรื่องคนละห้อง
- [ ] ที่ 1280×720 ตารางต้องไม่ล้นแนวนอน
**Verify:** browser pane บนเครื่องที่มี `experiments/results` จริง → screenshot compare + error
**Dependencies:** T4, T7, T8, T9 · **Files:** `js/panels.js`, `js/main.js`, `js/api.js`, `app.css` · **Scope:** M

#### Task 11: `warm_web_cache.py`
**Description:** สคริปต์สร้าง `reference.ply` และ `error.ply` ของ gallery ไว้ล่วงหน้าสำหรับห้องที่ระบุ รวมถึง web jobs ที่ `done` ด้วย โดยเรียกฟังก์ชันเดียวกับ endpoint ไม่ต้องผ่าน HTTP
**Acceptance:**
- [ ] หลังรัน `--scenes 47429736 47333774` แล้ว การเปิด error ของแต่ละ mesh บนเว็บต้องใช้เวลาไม่เกิน 3 วินาที
- [ ] รันซ้ำแล้วต้องไม่คำนวณใหม่ (ข้ามรายการที่ cache ยังใหม่อยู่)
**Verify:** รันจริง 2 ครั้ง, จับเวลา `curl -o /dev/null -w '%{time_total}'`
**Dependencies:** T9 · **Files:** `scripts/warm_web_cache.py`(ใหม่) · **Scope:** S

### Checkpoint 3: Gallery
- [ ] `make test` + `make lint` ผ่าน
- [ ] ข้อ 4–7 ของ SPEC §8 ผ่านบนข้อมูลจริง

### Phase 4: Demo readiness

#### Task 12: Docs + ซ้อม demo แบบ offline
**Description:**
- อัปเดตบรรทัด Phase 6 ใน `docs/architecture/README.md` (endpoint ใหม่, `job.json`, gallery)
- แก้ `demo_presentation_plan.md` §1.3/§2.3 ให้ตรงกับที่ทำจริง (ตัวจับเวลาใช้เวลาจาก server, `warm_web_cache.py`, `meta.json`)
- ซ้อม demo ช่วง A–D ด้วยข้อมูลจริง โดยปิด Wi-Fi ตั้ง `HF_HUB_OFFLINE=1` และใช้ viewport 1280×720
- ⚠️ ถามก่อนรัน FT/pretrained web jobs บนห้องจริง เพราะใช้เวลาหลายนาที
- ตรวจว่าตัวเลขของ web ต่างจาก gallery ไม่เกิน ±20% (Open Question 2 ใน SPEC)

**Acceptance:**
- [ ] SPEC §8 ผ่านครบทั้ง 10 ข้อ และแนบ screenshot ของ compare + error + panel
- [ ] docs ตรงกับโค้ด
**Verify:** checklist SPEC §8, `make test`, `make lint`
**Dependencies:** T1–T11 · **Files:** `docs/architecture/README.md`, `docs/notes/demo_presentation_plan.md`, `SPEC.md` (สถานะ) · **Scope:** S

### Checkpoint: Complete
- [ ] SPEC §8 ผ่านครบ 1–10
- [ ] ทุก task commit แยกกัน และยังไม่ push
- [ ] ผู้ใช้ review ก่อนวันซ้อม

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| framing ของหลายช่องทำให้ความพองของ pretrained หายไป | High (ข้อความหลักของ demo) | framing จาก reference ครั้งเดียว มี acceptance ใน T4/T6 และตรวจด้วยตาที่ checkpoint 2 |
| `error.ply` ของ mesh ขนาด 12.8 MB ช้าเมื่อเปิดครั้งแรกบนเวที | Med | cache + `warm_web_cache.py` (T11) + ข้อ checklist วันก่อน |
| reference ของ gallery ต้องใช้ `data/arkitscenes` ถ้าไม่มี `build_dataset` จะพัง | Med | `gt_mesh()` คืน None → `has_reference:false` แล้วปิดปุ่ม ไม่ให้ขึ้น 500 และมี test กรณีไม่มี reference |
| ดาวน์โหลด three.js แล้วเวอร์ชันไม่ตรง ทำให้ addons import พัง | Low | pin 0.160.0 ตามที่ใช้อยู่ บันทึก `VERSION` และถามผู้ใช้ก่อนดาวน์โหลด |
| scissor rendering บน Retina หรือตอน resize | Med | ใช้ `setPixelRatio` เดียว คำนวณ rect เป็น CSS px เก็บ `updateStyle` ไว้ตามที่ comment เดิมเตือน และตรวจใน T4 |
| ตัวเลขของ web (stride 5, ไม่ mask) ต่างจากเปเปอร์มาก | Med (โดนถาม) | ป้าย "demo run" กับ "paper run" และตรวจความต่างใน T12 |
| `mono_sparse` เป็น Large ทำให้ช้ามาก | Low | ไม่ใช้รันสดบนเวที ใช้ผลจาก gallery แทน |

## Open Questions (จาก SPEC §9 — ตอนนี้ใช้ค่า default ไปก่อน)
1. `mono_sparse` บนเว็บใช้ Large (T1) หรือคง Small
2. ยอมให้ตัวเลขของ web ต่างจากเปเปอร์ได้ ±20% ของ Chamfer (T12)
3. ใน gallery แสดงแถว `gt`/`oracle` โดยมีป้าย "uses GT" (T9)
4. แก้ `demo_presentation_plan.md` ให้ตรงด้วย (T12)
