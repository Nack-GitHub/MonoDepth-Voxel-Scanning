# Spec: roomscan_web สำหรับ demo วันนำเสนอ IS (P0 + P1)

สถานะ: **draft รอ approve** (2026-09-30) · เขียนตาม `agent-skills:spec-driven-development`
ที่มา: `docs/notes/demo_presentation_plan.md` §1 (gap G1–G10, งานข้อ 1–10) และ §2.1 (ลำดับ demo A–E)
ขอบเขต: งาน P0 (ข้อ 1–6) + P1 (ข้อ 7–10) · **ไม่รวม** P2 (กราฟ scale รายเฟรม, ภาพ depth รายเฟรม, เครื่องมือวัดระยะ)

การตัดสินใจที่ยืนยันแล้ว (2026-09-30):
- front-end เป็น **vanilla ES modules** ไม่มี build step ไม่มี npm และ three.js vendored ไว้ในเครื่อง
- label บนหน้าเว็บเป็น**ภาษาอังกฤษ** ตรงกับชื่อในเปเปอร์ (FT-LiDAR-24, DA-v2 Metric-Indoor, …)
- โหมดเปรียบเทียบเลือกได้**ทั้ง web jobs และ gallery** ผสมกันได้ ถ้าเป็นคนละห้องให้เตือน

---

## 1. Objective

**สร้างอะไร:** ขยาย `src/roomscan_web/` ให้เป็นเครื่องมือ demo ที่ทำให้กรรมการ "เห็นด้วยตา" สามข้อความหลักของเปเปอร์:
1. pretrained DA-v2 Metric-Indoor วางห้องผิด ~54 cm (ห้องพองออก 1.18–1.45×)
2. FT-LiDAR-24 ลดเหลือ ~16 cm โดยใช้ RGB อย่างเดียว
3. แต่ยังห่างจาก iPad LiDAR (~3 cm) และยังไม่ผ่านเกณฑ์ use case ไหนเลย

**ผู้ใช้:** ผู้นำเสนอ (ใช้เว็บบน laptop ต่อโปรเจกเตอร์ 1280×720 แบบ offline) และกรรมการที่ดูจอ

**ภาพที่ต้องได้:** mesh 3 ก้อน (pretrained | FT-LiDAR-24 | LiDAR) ของห้องเดียวกันวางเทียบกัน หมุนพร้อมกัน สลับเป็นสี error เทียบ Faro ได้ ซ้อน Faro wireframe ได้ และมีแผง metric พร้อม badge use case

**User stories:**
- US1: ในฐานะผู้นำเสนอ ฉันอัปโหลด `47429736.zip` ด้วย preset "iPad LiDAR" แล้วเห็นตัวจับเวลาเดินจนงานเสร็จ จากนั้นหมุน mesh ได้ (demo ช่วง A)
- US2: ฉันเลือกงาน/รายการ gallery ได้สูงสุด 3 รายการ แล้วดูเทียบกันแบบกล้องเดียว (ช่วง B)
- US3: ฉันกดสลับ "Color: real / error" และ "Faro overlay" แล้วเห็นผนังของ pretrained เป็นสีแดงและพองเกิน wireframe (ช่วง C)
- US4: ฉันเปิดแผง metric แล้วชี้ได้ว่า FT ได้ recall@10 cm แค่ ~0.5 จึงไม่ผ่าน Overview (ช่วง D)
- US5: ถ้า server ถูกปิดแล้วเปิดใหม่ งานที่รันไว้ยังอยู่ครบ
- US6: ถ้ากรรมการขอดูห้องอื่น ฉันเปิดผลของทั้ง 6 ห้องจาก gallery ได้เลยโดยไม่ต้องรันใหม่
- US7: ปิด Wi-Fi แล้วทุกอย่างข้างบนยังทำงานได้

---

## 2. Tech Stack

| ชั้น | ใช้อะไร | หมายเหตุ |
|---|---|---|
| Backend | FastAPI (มีอยู่แล้ว, `.[web]`), Python 3.12 | ไม่เพิ่ม dependency ใหม่ |
| คำนวณ error / อ่าน mesh | Open3D 0.19 + numpy (มีอยู่แล้ว) | reuse `roomscan.evaluation.metrics_3d` |
| Colormap | `turbo` 0–10 cm เท่ากับ Fig. exp6 (`scripts/make_figures.py`) | ทำ LUT 256 สีด้วย numpy เพราะ `.[web]` ไม่มี matplotlib |
| Frontend | HTML + CSS + vanilla ES modules | ไม่มี bundler/TS/npm |
| 3D | three.js **0.160.0** (เวอร์ชันเดิม) `three.module.js`, `OrbitControls.js`, `PLYLoader.js` | vendored ใน `static/vendor/three/` |
| Test | pytest + `fastapi.testclient` | ใช้ synthetic scene และไม่ใช้ torch |

---

## 3. Commands

```bash
make setup                                   # .venv พร้อม .[mono,dev,web]
make test                                    # pytest -q ต้องผ่านโดยไม่มี data/ และไม่มี torch
make lint                                    # ruff check src tests scripts
make web                                     # uvicorn --factory roomscan_web.app:create_app --port 8765
HF_HUB_OFFLINE=1 make web                    # วันจริง: ห้ามพึ่งเน็ต
make capture-zip SCENE=47429736              # outputs/captures/47429736.zip (+ reference.ply, meta.json)
.venv/bin/python scripts/warm_web_cache.py --scenes 47429736 47333774   # (ใหม่) สร้าง error.ply ล่วงหน้า
```

ตัวแปรแวดล้อม (มีอยู่แล้ว + ใหม่):
- `ROOMSCAN_WORK_DIR` (default `outputs/web`), `ROOMSCAN_PRESET`, `ROOMSCAN_MAX_UPLOAD_MB` เป็นของเดิม
- **ใหม่** `ROOMSCAN_GALLERY_ROOT` (default `experiments/results`) และ `ROOMSCAN_GALLERY_EXPS` (default `exp1_depth_source,exp6_finetune`)

---

## 4. Project Structure

```
src/roomscan_web/
  app.py            routes เท่านั้น: รับ request, ตรวจ input, เรียก module อื่น
  jobs.py           JobRunner + เขียน/อ่าน job.json (persist), timestamps
  presets.py        (ใหม่) PRESETS: key → {label, overrides, note, eta}; แหล่งเดียวของชื่อบนเว็บ
  gallery.py        (ใหม่) สแกน experiments/results/<exp>/<scene>_<run>/ → รายการ, label ตาม run
  errorcolor.py     (ใหม่) ระยะ vertex→reference, turbo LUT, เขียน/cache error.ply
  static/
    index.html      layout + importmap ที่ชี้ไป vendor/
    app.css
    js/api.js       fetch wrappers (/presets, /scans, /gallery, …)
    js/viewer.js    renderer ตัวเดียว + viewport หลายช่อง (scissor), กล้อง/OrbitControls ร่วมกัน
    js/panels.js    รายการงาน, gallery table, metric panel, badge, legend, timer
    js/format.js    ฟังก์ชัน pure: cm, badge(recall), mm:ss
    js/main.js      state + ต่อทุกอย่างเข้าด้วยกัน
    vendor/three/   three.module.js, OrbitControls.js, PLYLoader.js (0.160.0) + VERSION
scripts/
  export_capture.py เพิ่มการเขียน meta.json
  warm_web_cache.py (ใหม่) สร้าง error.ply ของ gallery + web jobs ล่วงหน้า
tests/
  test_web.py       ขยาย (ดู §6)
docs/architecture/README.md   อัปเดตบรรทัด Phase 6 (endpoint ใหม่)
```

**ไม่แตะ:** `src/roomscan/pipeline.py`, `src/roomscan/models/`, `depth_sources/`, `experiments/results/**` (อ่านอย่างเดียว)

### 4.1 API (ของเดิม + ใหม่)

| Method + path | ใหม่? | คืนอะไร |
|---|---|---|
| `GET /presets` | ใหม่ | `[{key,label,note,eta_s_per_frame,overrides}]` ตามลำดับที่ใช้ใน dropdown |
| `POST /scans` | เพิ่ม field | form `file`, `overrides` (เดิม) + `preset` (optional; ถ้าส่งมาจะ merge overrides ของ preset และเก็บ key ไว้ติดป้าย) |
| `GET /scans`, `GET /scans/{id}` | เพิ่ม field | `public()` เพิ่ม `preset`, `label`, `scene`, `up`, `created_at`, `started_at`, `finished_at`, `has_reference`, `origin:"web"` |
| `GET /scans/{id}/mesh.ply` | เดิม | |
| `GET /scans/{id}/reference.ply` | ใหม่ | Faro reference จาก capture; 404 ถ้าไม่มี |
| `GET /scans/{id}/error.ply` | ใหม่ | mesh ที่ vertex color = turbo(error, 0–10 cm); 409 ถ้ายังไม่ done, 404 ถ้าไม่มี reference |
| `GET /gallery` | ใหม่ | `[{id:"exp6_finetune/47429736_mono_ft_lidar_all", exp, scene, run, label, metrics, up:"z", has_reference, origin:"paper"}]` |
| `GET /gallery/{exp}/{run}/mesh.ply` / `reference.ply` / `error.ply` | ใหม่ | เหมือนฝั่ง scans |
| `GET /static/...` | ใหม่ | `StaticFiles` สำหรับ css/js/vendor |

**Presets (`presets.py`)** เรียงตามลำดับ dropdown:

| key | label บนเว็บ | overrides | note |
|---|---|---|---|
| `lidar` | iPad LiDAR (sensor) | `{}` | needs `depth/` in zip |
| `ft_lidar_24` | FT-LiDAR-24 (ours, RGB only) | mono, `depth_anything_v2_ft_lidar_all`, identity | ~3–7 min/room |
| `mono_metric` | DA-v2 Metric-Indoor (pretrained) | mono, `depth_anything_v2_metric_indoor`, identity | |
| `mono_sparse` | DA-v2 Large + sparse points (upper bound) | mono, `depth_anything_v2_large`, sparse_points | label ว่า "LiDAR-sampled VIO proxy — upper bound" (ADR-011) |

**Gallery labels** (`gallery.py` map ชื่อ run → label เดียวกับเปเปอร์):
`gt`→"Faro depth (pipeline ceiling)", `lidar`→"iPad LiDAR", `mono_metric`→"DA-v2 Metric-Indoor (pretrained)",
`mono_ft_lidar_all`→"FT-LiDAR-24", `mono_ft_lidar`→"FT-LiDAR-10", `mono_ft_faro`→"FT-Faro-10",
`mono_oracle`→"DA-v2 Large + per-frame oracle scale (uses GT)", `mono_sparse`→"DA-v2 Large + sparse (upper bound)",
`mono_scene`→"DA-v2 Large + once-per-room scale", `mono_depth_pro`→"Depth Pro"; run ที่ไม่รู้จักให้ใช้ชื่อ run ตรง ๆ
ถ้า run เดียวกันอยู่ทั้ง exp1 และ exp6 (`mono_metric`) ให้แสดงแค่อันเดียว และเลือกของ exp6 ก่อน

**Persist (`jobs.py`):** เขียน `results/web/<id>/job.json` ทุกครั้งที่ status เปลี่ยน โดยเก็บ `{id, scene_dir, overrides, preset, status, error, created_at, started_at, finished_at}`
- ตอน `JobRunner.__init__` จะอ่าน `job.json` ทุกไฟล์กลับมา ถ้า `done` ให้โหลด `metrics.json` และ `mesh_path` ด้วย
- ถ้า status ที่อ่านได้คือ `queued` หรือ `running` ให้เปลี่ยนเป็น `failed` พร้อม error `"interrupted by server restart"` และ**ไม่ requeue เอง**
- `job.json` ต้องเขียนไว้ใน `results/web/<id>/` ที่ pipeline ใช้อยู่แล้ว ห้ามเพิ่มที่เก็บข้อมูลที่อื่น (ADR-006 "results are files")

**Error cache:** ของ web jobs เก็บที่ `results/web/<id>/error.ply` ส่วนของ gallery เก็บที่ `outputs/web/cache/<exp>/<run>/error.ply` และ**ห้ามเขียนลง `experiments/results/`**
ให้ทิ้ง cache ถ้า mtime ของ `mesh.ply` หรือ reference ใหม่กว่า cache

**ระยะ error:** สำหรับแต่ละ vertex ของ pred mesh ให้หาระยะไปยังจุดที่ใกล้ที่สุดใน point cloud ที่ sample จาก reference (200k จุด, seed 0 เหมือน `per_point_error`)
แล้ว map สีด้วย turbo(clip(d/0.10, 0, 1)) และเขียนเป็น PLY ที่มี vertex color

**meta.json (capture):** `export_capture.py` จะเขียน `{"source":"arkitscenes","scene":"47429736","up":"z","stride":5}`
- server อ่านค่า `scene` และ `up` ไปใส่ใน `public()`
- ถ้าไม่มีไฟล์ ให้ใช้ `up:"y"` และ `scene:null`
- ฝั่ง gallery ใช้ `up:"z"` เสมอ
- `dataio/custom.py` ต้องไม่สนใจไฟล์นี้ (ตรวจด้วย test)

### 4.2 UI

- **Sidebar:** แท็บ "Scans" (อัปโหลดและรายการงาน) กับแท็บ "Gallery" (ตารางห้อง × source ที่มี Chamfer ในแต่ละช่อง)
  - ทุกแถวมี checkbox "compare" เลือกได้สูงสุด 3 รายการ
  - ถ้าคลิกที่ตัวแถว จะเปิดดูรายการนั้นเดี่ยว ๆ
- **ตัวจับเวลา:** งานที่ `running` แสดง `running 01:23` นับจาก `started_at` ของ server (reload หน้าแล้วยังนับถูก) พร้อมเวลาที่คาดไว้ = `eta_s_per_frame × n_frames`
- **Viewer:** มี 1 ถึง 3 ช่อง ใช้ `WebGLRenderer` ตัวเดียว แล้วแบ่งแต่ละช่องด้วย `setScissor`/`setViewport`
  - ทุกช่องใช้ camera และ OrbitControls ร่วมกัน
  - ใต้แต่ละช่องมีป้าย label, Chamfer (cm) และ origin badge: "paper run" สำหรับ gallery หรือ "demo run (stride 5, no GT mask)" สำหรับ web
- **Framing:** ตั้งจาก reference mesh ถ้ามี ถ้าไม่มีให้ใช้ช่องที่เป็น LiDAR ถ้าไม่มีอีกให้ใช้ช่องแรก และตั้ง**ครั้งเดียวสำหรับทุกช่อง** ห้าม fit แยกช่อง (ถ้า fit แยก ห้องที่พองจะถูกย่อจนภาพดูพอดีกันหมด)
- **Toolbar:** "Color: real / error" (มี legend turbo 0–10 cm), "Faro overlay" (wireframe สีเทาโปร่งใส) และ "Z-up" (ค่าตั้งต้นมาจาก `up`)
  - ถ้ารายการที่เลือกไม่มี reference ให้ปุ่ม error และ overlay ใช้ไม่ได้ พร้อม tooltip "no Faro reference"
- **ห้องไม่ตรงกัน:** ถ้าเลือกรายการที่ `scene` ต่างกัน ให้ขึ้นแถบเตือน "different rooms — not comparable" แต่ยังแสดงให้
- **Metric panel:** ใช้กับรายการที่ focus อยู่ ซึ่งคือรายการที่คลิกหรือช่องที่ hover
  - แสดง Chamfer, Accuracy, Completeness, F@5, Recall@2/5/10 cm และ `timing.depth/fusion/total`
  - มี badge 3 ตัว: Overview ใช้ R@10, Furniture ใช้ R@5 และ Renovation ใช้ R@2 โดย ≥ 0.9 = ✓, 0.75–0.9 = ~, < 0.75 = ✗
  - ใต้ badge มีหมายเหตุว่า "thresholds are illustrative, not a standard"
- ต้องอ่านออกที่ 1280×720 และรองรับ light/dark ตาม CSS variables เดิม

---

## 5. Code Style

ให้ตามสไตล์ไฟล์เดิม: Python มี type hints, `from __future__ import annotations`, docstring สั้นบนหัวไฟล์ และ ruff line-length 120
ให้ `app.py` บางไว้ ส่วน logic อยู่ในโมดูลที่ test ได้โดยไม่ต้องผ่าน HTTP

```python
# gallery.py
RUN_LABELS = {"lidar": "iPad LiDAR", "mono_ft_lidar_all": "FT-LiDAR-24", ...}
_RUN_RE = re.compile(r"^(\d{8})_([a-z0-9_]+)$")


def resolve_run(root: Path, exps: Sequence[str], exp: str, run: str) -> Path:
    """The run folder for a gallery URL, or ValueError — never trust path parts from the client."""
    if exp not in exps or not _RUN_RE.match(run):
        raise ValueError(f"unknown gallery item {exp}/{run}")
    d = (root / exp / run).resolve()
    if d.parent != (root / exp).resolve() or not (d / "mesh.ply").is_file():
        raise ValueError(f"unknown gallery item {exp}/{run}")
    return d
```

```js
// format.js — pure, no DOM, so they are easy to eyeball and reuse
export const cm = (m) => `${(m * 100).toFixed(1)} cm`;
export const badge = (recall) => (recall >= 0.9 ? '✓' : recall >= 0.75 ? '~' : '✗');
```

- เขียน JS เป็น ES module หนึ่งหน้าที่ต่อไฟล์ ใช้ `const`/`let` และ `async/await` ไม่ใช้ framework และไม่ตั้ง global นอกจาก state ใน `main.js`
- ชื่อ method บนเว็บต้องตรงกับเปเปอร์ ห้ามโชว์ชื่อภายใน เช่น `R3` หรือ `ft_lidar_all` เป็น label
- comment ในโค้ดเขียนภาษาอังกฤษ ส่วน docs และ spec เขียนภาษาไทย

---

## 6. Testing Strategy

ทุก test อยู่ใน `tests/test_web.py`, ใช้ synthetic scene (`write_synthetic_scene`), ไม่ใช้ torch, `make test` ต้องผ่านบนเครื่องที่ไม่มี `data/`

| ความสามารถ | test |
|---|---|
| Presets | `GET /presets` มีครบ 4 key ตามลำดับ; `ft_lidar_24` ชี้ `depth_anything_v2_ft_lidar_all`; `POST /scans` ด้วย `preset` ที่ไม่รู้จัก → 400; overrides ของทุก preset ⊆ `ALLOWED_OVERRIDES` |
| Persist | รันงาน → สร้าง `create_app` ใหม่บน work dir เดิม → งานยัง `done` มี metrics/mesh; งานที่ค้าง `queued` → `failed: interrupted…` |
| Timestamps | `started_at ≤ finished_at`, มีครบเมื่อ done |
| meta.json | capture ที่มี `meta.json` → `up:"z"`, `scene` ถูก; pipeline ยังรันผ่าน (custom loader ไม่พัง) |
| reference.ply / error.ply | ได้ `ply` ที่มี vertex color; จำนวน vertex = mesh.ply; เรียกซ้ำใช้ cache (mtime ไม่เปลี่ยน); ไม่มี reference → 404; ยังไม่ done → 409 |
| Error math | หน่วยของ `errorcolor`: mesh = reference → error ≈ 0 → สีเท่ากับ turbo(0); เลื่อน 5 cm → ≈ turbo(0.5) |
| Gallery | ใช้ tmp gallery root ที่มี run สังเคราะห์ 2 run (รัน pipeline จริงบน synthetic) → `/gallery` คืน 2 รายการพร้อม label/metrics; `mesh.ply` 200; `../`, exp นอก allowlist, run ไม่ตรง regex → 404; ไม่มีการเขียนไฟล์ใต้ gallery root |
| Static/offline | `GET /static/vendor/three/three.module.js` 200; `index.html` ไม่มีสตริง `cdn.` / `http` ใน importmap |

**ตรวจ front-end ด้วยตา** (ไม่มี JS test runner ตามข้อตกลงว่าไม่ใช้ npm): เปิดผ่าน browser pane ด้วย `make web` + gallery จริง
1. compare 3 ช่องของ 47429736 (pretrained/FT/LiDAR) ต้องหมุนพร้อมกัน และ mesh ของ pretrained ต้องใหญ่กว่าช่องอื่นอย่างเห็นได้ชัด
2. สลับเป็นสี error แล้ว legend ต้องขึ้น และผนังของ pretrained ต้องออกโทนแดงมากกว่า FT
3. เปิด overlay แล้วต้องเห็น wireframe ของ Faro
4. ที่ viewport 1280×720 ป้ายต้องอ่านออก และ legend ต้องไม่ถูกบัง
5. ตัด network ด้วย DevTools offline แล้ว reload หน้า ต้องไม่มี request ไปนอก localhost (ดูจาก `read_network_requests`)
6. ปิด server แล้วเปิดใหม่ รายการงานต้องยังอยู่

---

## 7. Boundaries

**Always**
- แก้เฉพาะ `src/roomscan_web/`, `scripts/export_capture.py`, `scripts/warm_web_cache.py`, `tests/test_web.py` และ docs
- ตรวจ path ที่มาจาก client ทุกครั้ง (gallery `exp/run`, job id) ห้ามเอาไปต่อเป็น path ตรง ๆ
- ติดป้ายแหล่งของตัวเลขทุกครั้ง: "paper run" หรือ "demo run"
- sparse points ต้องมีคำว่า "upper bound" ติดอยู่เสมอ
- รัน `make test` และ `make lint` ก่อน commit ทุกครั้ง แล้ว commit ทีละ task ได้เลยโดยไม่ต้องถาม แต่**ห้าม push**

**Ask first**
- ดาวน์โหลดไฟล์ three.js 0.160.0 จาก jsdelivr/npm เพื่อ vendor (ต้องบอกชื่อไฟล์และขนาดก่อน)
- เพิ่ม dependency ใหม่ใน `pyproject.toml`
- เปลี่ยนรูปแบบ `metrics.json` หรือโฟลเดอร์ผลลัพธ์ของ pipeline
- รันงาน mono (FT/pretrained) บนห้องจริง ซึ่งใช้เวลาหลายนาทีและต้องมี weights จาก HF

**Never**
- แก้ `pipeline.py` หรือ `models/` (ถ้าต้องมี progress callback รายเฟรม ให้หยุดแล้วเขียน ADR ก่อน)
- เขียนลง `experiments/results/**` หรือ commit อะไรก็ตามใต้ `data/` หรือ `outputs/`
- แสดงผลของ synthetic scene (exp0) ใน gallery ว่าเป็นผลการทดลอง
- โหลด asset ใด ๆ จาก CDN หรืออินเทอร์เน็ตตอน runtime
- แสดงชื่อภายใน (`R3`, `ft_lidar_all`) เป็น label

---

## 8. Success Criteria

1. `make test` และ `make lint` ผ่าน โดยมี test ใหม่ครอบทุกแถวใน §6
2. dropdown มี 4 preset ตาม §4.1 และ label ตรงกับเปเปอร์
3. restart server แล้ว web jobs ที่ `done` ยังอยู่ครบพร้อม label ของ preset
4. จาก Gallery เลือก 47429736 pretrained + FT-LiDAR-24 + LiDAR ได้ 3 ช่องที่หมุนพร้อมกัน, กรอบกล้องมาจาก reference, ป้ายใต้ช่องแสดง Chamfer ≈ 54.1 / 12.4 / 3.1 cm (ต้องตรงกับ `metrics.json` ถึงทศนิยมหนึ่งตำแหน่ง)
5. สลับสี error แล้วได้ turbo 0–10 cm พร้อม legend; เปิดครั้งแรกหลัง `warm_web_cache.py` ต้องขึ้นภายใน 3 วินาทีต่อ mesh
6. Faro overlay แสดงได้ทั้งใน gallery และ web job ที่มาจาก `export_capture.py`
7. แผง metric แสดง 8 ค่า + badge 3 ตัว; FT-LiDAR-24 ห้อง 47429736 ได้ Overview ✗ (R@10 = 0.58)
8. อัปโหลด `47429736.zip` (ที่ export ใหม่พร้อม meta.json) ด้วย iPad LiDAR → mesh ตั้งตรงโดยไม่ต้องติ๊ก Z-up, ตัวจับเวลาเดินระหว่างรัน
9. ปิด Wi-Fi + `HF_HUB_OFFLINE=1` → ข้อ 2–8 ยังทำงาน, network log ไม่มี request ออกนอก localhost
10. ที่ 1280×720 ทุกป้าย/legend/badge อ่านออก ไม่มี horizontal scroll

---

## 9. Open Questions

1. **`mono_sparse` บนเว็บ:** จะเปลี่ยนเป็น Large ให้ตรงกับเปเปอร์ (ช้ากว่า ~6×) หรือคง Small แล้วเขียนบนป้ายว่า "Small"? spec นี้ใช้ Large ไปก่อน
2. **ตัวเลขของ web ต่างจากเปเปอร์ได้แค่ไหน** ก่อนต้องหาสาเหตุ (เพราะ stride 5 และไม่ได้ mask ด้วย GT)? เสนอให้ยอมรับได้ถ้าต่างไม่เกิน ±20% ของ Chamfer
3. **Gallery ควรโชว์ทุก run หรือไม่:** รวม `gt` / `mono_oracle` ที่ใช้ GT ด้วยไหม หรือซ่อนไว้หลังตัวเลือก "show oracle rows" เพื่อไม่ให้สับสนตอน demo? spec นี้โชว์ทั้งหมดแต่ติดป้ายว่า "uses GT"
4. จะอัปเดตข้อ 1.3 ใน `docs/notes/demo_presentation_plan.md` ให้ตรงกับ spec นี้ด้วยไหม เช่นเรื่อง timer ที่ใช้เวลาจาก server แทน client
