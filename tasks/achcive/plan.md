# Implementation Plan: Fine-tune DA-v2 ด้วยครู LiDAR (ฝั่งเครื่องเทรน)

อ้างอิง: `SPEC.md` (rev 2, commit 12378f2) · แผนนี้เขียนตาม `agent-skills:planning-and-task-breakdown`
สถานะ: **รอ approve** (2026-09-21)

## Overview

สร้าง `src/roomscan/training/` ที่ fine-tune DA-v2 Metric-Indoor Large (freeze DINOv2, เทรน DPT neck + head) บน ARKitScenes
ด้วย label 2 แบบ (Faro / LiDAR) แล้วส่ง checkpoint ขึ้น HF Hub ให้ pipeline เดิมโหลดผ่าน registry ได้
งานแบ่งเป็น 5 phase, 15 task; **โค้ดเกือบทั้งหมดเขียนและเทสต์บน Mac ได้** (มี torch MPS + synthetic scene) —
เครื่อง Windows/WSL ใช้เฉพาะโหลดข้อมูลจริง + รันเทรน (T2, T6–T7, T10–T13, T15)

## Architecture decisions (จากการอ่านโค้ดจริง — ไม่ต้องแก้ loader เลย)

- **ใช้ `ARKitScenesScene` เดิมเป็น data source ของการเทรน** โดยส่ง
  `fusion_resolution=(640, 480), rgb_asset="vga_wide", frame_source="highres_depth"|"vga_wide"` —
  `_read_depth()` ย่อ/ขยาย depth ทุกตัวด้วย **nearest** ลง grid นั้นอยู่แล้ว (`_to_fusion_grid`) ตรงกับกฎเหล็กใน SPEC §5.3
  และ `vga_wide` 640×480 ถูกอ่านที่ native resolution → rgb/depth/mask ขนาดตรงกันโดยไม่ต้องเขียน resample ใหม่
  - Faro target: `frame_source="highres_depth"` (เฟรม = timestamp ของ Faro ~10 fps; `vga_wide` 30 fps อยู่ใน tol 0.02 s เสมอ)
  - LiDAR target: `frame_source="vga_wide"` + stride 3 (≈10 fps); `gt_depth` จะเป็น None เกือบทุกเฟรม — ไม่ใช้
- **ไม่ใช้ `AutoImageProcessor` ตอนเทรน** — processor ของ HF resize รูปเองแบบ keep-aspect ทำให้ depth/mask ไม่ตรงกับ pred
  → `transforms.py` ทำ resize สั้น = 518 (กว้าง 686 = 49×14 สำหรับ 4:3), normalize ImageNet mean/std เอง ทั้งเทรนและ val
  (`models/depth_anything.py` ตอน inference ยังใช้ processor ตามเดิม — ไม่แตะ)
- **หมุนภาพตั้งตรงด้วย `_ROT` เดียวกับ `depth_sources/monocular.py`** (`{"Up":0,"Left":1,"Down":2,"Right":3}` → `np.rot90`)
  ใช้กับ rgb/depth/mask พร้อมกัน; ไม่ import จาก `depth_sources` (dependency rule) → ย้าย `_ROT` ไป `types.py`? **ไม่** — คัดลอก dict 4 ค่า
  พร้อม comment ชี้ต้นทาง (เล็กกว่าการย้ายโมดูลและไม่แตะ pipeline)
- **Split ผ่าน `ARKitScenesScene(split=fold)`** — loader รับ `split="Training"|"Validation"` อยู่แล้ว (`raw/<split>/<id>`), V จาก fold Validation จึงอ่านได้ทันที
- **Val ใช้ `evaluation.metrics_2d.depth_metrics` + `mean_metrics` ตรง ๆ** บน pred ที่ resize กลับ 640×480 (nearest) — ตัวเลขนิยามเดียวกับตารางเปเปอร์
  (`max_depth=5.0` ตามค่า default ของ metrics_2d = ค่าเดียวกับ `fusion.depth_trunc`)
- **Config ใช้ OmegaConf + `roomscan.config` เดิม** (`_base_` + dotlist) — ไม่เพิ่ม lib config
- **Freeze = `model.backbone.requires_grad_(False)` + `model.backbone.eval()`** (DINOv2 ใน HF `DepthAnythingForDepthEstimation` อยู่ที่ `.backbone`, DPT ที่ `.neck`, head ที่ `.head`) — เทสต์ยืนยันว่า weight backbone ไม่เปลี่ยนหลัง 2 steps
- **Open3D** อยู่ใน core deps ของ `pyproject` อยู่แล้ว (`pip install -e .` ติดตั้งเสมอ) — โค้ด `training/` ห้าม import แต่ test fixture ใช้ `write_synthetic_scene` ที่ต้อง open3d ได้ (`importorskip`)
- **ทดสอบ train loop บน CPU ด้วย `..._metric_indoor_small`** 2 steps (โหลดจาก HF; skip ถ้าออฟไลน์) — Large ใช้เฉพาะบน GPU

## Dependency graph

```
T1 ADR-013 + pyproject[train] + Makefile train
 │
 ├── T3 pairs/dataset (numpy DepthPair + torch Dataset)  ──┐
 ├── T4 transforms (518/14, crop, flip, normalize, rot)   ──┼── T8 train.py (loop, freeze, bf16, log, last/)
 ├── T5 loss (SiLog, L1)                                  ──┘        │
 │                                                                    ├── T9 validate.py + best/ + step-0 row
 ├── T2 WSL2 env ──┬── T6 select scenes → splits.yaml               │
 │                 └── T7 download + --verify  ─────────────────────┼── T10 R0_smoke (gate VRAM/loss)
 │                                                                    │        │
 │                                                                    │        ├── T11 R1_ft_faro
 │                                                                    │        ├── T12 R2_ft_lidar
 │                                                                    │        └── T13 R3_ft_lidar_all
 └── T14 push.py + models hf_id= + registry  ────────────────────────┴──────── T15 push HF + Mac smoke + push branch
```

ขนาน: {T3, T4, T5} อิสระกัน · T2/T6/T7 (WSL, รอเน็ต) ทำคู่กับ T8/T9 (Mac) · T14 ทำระหว่างรอ T11–T13 รัน

## Task list

### Phase 0 — Foundation (Mac หรือ WSL, โค้ดล้วน)

#### Task 1: ADR-013 + extra `train` + `make train`
**Description:** เปิดทางให้มี training loop ใน repo อย่างเป็นทางการ: ADR-013 supersede ADR-008 (บันทึกเหตุผล split จาก SPEC §5.2 + กฎ weight ไม่ commit), เพิ่ม extra `train` ใน `pyproject.toml`, target `train` ใน `Makefile`, อัปเดต index ADR ใน README และบรรทัดใน CLAUDE.md
**Acceptance:**
- [ ] `docs/architecture/adr-013-finetune-with-lidar-teacher.md` มี Status/Context/Decision/Consequences + หัวข้อ "Split" ที่คัดจาก SPEC §5.1–5.2 และระบุ ADR-008 → Superseded
- [ ] `pip install -e ".[train]"` ติดตั้ง torch, torchvision, transformers, huggingface_hub, safetensors, accelerate
- [ ] `make train RUN=x.yaml` = `$(BIN)/python -m roomscan.training.train --config x.yaml`
- [ ] README §8 มีแถว 013; ADR-008 แถวเดิมเปลี่ยนสถานะเป็น Superseded by 013; CLAUDE.md บรรทัด Phase/ADR-008 ชี้ 013
**Verification:** `make lint` · `make test` (ยังเขียวเหมือนเดิม) · `grep -n "013" docs/architecture/README.md CLAUDE.md`
**Dependencies:** None · **Where:** Mac
**Files:** `docs/architecture/adr-013-finetune-with-lidar-teacher.md`, `docs/architecture/README.md`, `pyproject.toml`, `Makefile`, `CLAUDE.md` · **Scope:** M (5 ไฟล์, เนื้อหาเล็ก)

#### Task 2: WSL2 environment พร้อมใช้ (manual, Windows)
**Description:** ทำตาม SPEC §3 "setup ครั้งเดียว" บน WSL2 Ubuntu — clone, branch `training`, venv 3.12, torch cu124, `pip install -e ".[train,dev]"`, symlink data ไป Linux fs, `hf auth login`
**Acceptance:**
- [ ] `nvidia-smi` ใน WSL เห็น RTX 3070; `python -c "import torch;print(torch.cuda.is_available())"` → True
- [ ] `make synthetic && make test && make lint` ผ่าน (training tests อาจ skip ถ้า T3–T5 ยังไม่ merge — ยอมรับ)
- [ ] `readlink data/arkitscenes` ชี้ `~/data/arkitscenes` (ไม่ใช่ `/mnt/c/...`); `hf auth whoami` ตอบชื่อบัญชี
**Verification:** รันคำสั่ง 3 ข้อข้างบน; บันทึก output ของ `nvidia-smi` + `torch.__version__` ลง `experiments/training/ENV.md` (commit — ใช้เขียน Setup ของเปเปอร์)
**Dependencies:** T1 (ต้องมี extra `train`) · **Where:** WSL
**Files:** `experiments/training/ENV.md` · **Scope:** XS (ไม่มีโค้ด)

### Phase 1 — Data slice บน synthetic (Mac)

#### Task 3: `training/pairs.py` + `dataset.py` — ฉาก → (rgb, depth, mask)
**Description:** `pairs.py` (numpy ล้วน): `DepthPair`, `make_pair(scene, idx, target, max_depth, lidar_min_confidence)`, `open_training_scene(root, video_id, fold, target)` ที่ห่อ `ARKitScenesScene` ด้วย `fusion_resolution=(640,480)`, `rgb_asset="vga_wide"`, `frame_source` ตาม target และ `iter_indices(scene, target, stride)`; `dataset.py` (torch): `DepthPairDataset(scenes, target, transform)` รวมหลายฉาก, `__getitem__` คืน tensor dict `{image, depth, mask, scene, idx}`
**Acceptance:**
- [ ] บน synthetic (hires 512×384): `make_pair(target="faro")` → rgb (480,640,3) uint8, depth (480,640) float32 เมตร (median ∈ [1, 5]), mask bool; depth==0 ทุกที่ที่ mask False
- [ ] `target="lidar"`: mask False ทุกพิกเซลที่ `confidence < lidar_min_confidence`; depth > max_depth ถูก mask
- [ ] `DepthPairDataset` รวม 2 ฉาก: `len == sum(len(iter_indices))`, `__getitem__` คืน `image` float32 (3,H,W), `depth`/`mask` (H,W)
- [ ] `training/` ไม่ import `pipeline`, `geometry`, `open3d` (test ตรวจด้วย `sys.modules` หลัง import)
**Verification:** `pytest tests/test_training.py -k "pair or dataset"` · `make lint`
**Dependencies:** T1 · **Where:** Mac
**Files:** `src/roomscan/training/__init__.py`, `src/roomscan/training/pairs.py`, `src/roomscan/training/dataset.py`, `tests/test_training.py` · **Scope:** M

#### Task 4: `training/transforms.py` — 518/14, crop, flip, jitter, upright, normalize
**Description:** ฟังก์ชัน numpy/torch: `upright(rgb, depth, mask, sky_direction)` (rot90 k เดียวกับ `monocular.py`), `resize_short(rgb, depth, mask, short=518)` (rgb bilinear/area, depth+mask **nearest**, ด้านยาวปัดเป็นพหุคูณ 14), `random_crop(…, 518)`, `hflip(p=0.5)`, `color_jitter` (rgb เท่านั้น), `normalize_imagenet` → `TrainTransform` / `EvalTransform` (eval = upright + resize_short + normalize, ไม่ crop)
**Acceptance:**
- [ ] หลัง `resize_short` ค่า depth ทุกค่าที่ไม่ใช่ 0 อยู่ใน `set(np.unique(depth_in))` (nearest จริง) และ H, W หาร 14 ลงตัว, H == 518
- [ ] `random_crop`/`hflip` ไม่เปลี่ยน histogram ของ depth (เรียงใหม่เท่านั้น); `color_jitter` ไม่แตะ depth/mask
- [ ] `upright("Left")` แล้ว `upright` ย้อนกลับ (`k=-1`) ได้ array เดิม; rgb/depth/mask หมุนเท่ากัน
- [ ] `EvalTransform` บน 640×480 ให้ (518, 686)
**Verification:** `pytest tests/test_training.py -k transform` · `make lint`
**Dependencies:** T1 · **Where:** Mac
**Files:** `src/roomscan/training/transforms.py`, `tests/test_training.py` · **Scope:** S

#### Task 5: `training/loss.py` — SiLog (+ L1)
**Description:** `silog(pred, target, mask, beta=0.15, scale=10.0)` = `scale * sqrt(var(g) + beta * mean(g)^2)`, `g = log(pred) - log(target)` บน mask (ตาม ZoeDepth/DA-v2 metric); `l1(pred, target, mask)`; `DepthLoss(lambda_l1=0.0)` รวม; clamp pred ≥ 1e-3 ก่อน log; batch ที่ mask ว่างคืน 0 ไม่ NaN
**Acceptance:**
- [ ] `silog(gt, gt) == 0` (± 1e-6); `silog(2*gt, gt) > silog(1.1*gt, gt) > 0`
- [ ] gradient ไหลผ่าน (`pred.requires_grad` → `.grad` ไม่เป็น None, finite)
- [ ] mask ว่างทั้ง batch → loss = 0, ไม่ NaN
**Verification:** `pytest tests/test_training.py -k loss`
**Dependencies:** T1 · **Where:** Mac
**Files:** `src/roomscan/training/loss.py`, `tests/test_training.py` · **Scope:** XS

### Checkpoint A — Data slice
- [ ] `make test` เขียวบน Mac (ใน `.venv` ที่มี torch) และ training tests **skipped** (ไม่ failed) เมื่อรัน `pytest -p no:cacheprovider` ใน env ที่ไม่มี torch (จำลองด้วย `python -c "import sys; sys.modules['torch']=None; ..."` หรือ venv `setup-gt`)
- [ ] `make lint` สะอาด · commit บน branch `training`
- [ ] Review กับผู้ใช้ก่อนเริ่มโหลดข้อมูลจริง

### Phase 2 — Scene selection + download (WSL, ใช้เวลารอ)

#### Task 6: `screen_scenes.py --fold` + `select_training_scenes.py` → `configs/training/splits.yaml`
**Description:** เพิ่ม `--fold {Training,Validation}` ให้ `screen_scenes.py` (default Training; ปรับ `BASE` URL ตาม fold); เขียน `select_training_scenes.py` ที่อ่าน metadata + candidates csv 2 ไฟล์, กัน `visit_id` ของ T (อ่าน `scenes:` จาก `exp1_depth_source.yaml` → map เป็น visit ผ่าน metadata), เลือก A (10, Faro, Training), B (14, LiDAR-only, Training — ไม่ต้อง `is_in_upsampling`), V (3, Faro, Validation) แบบ deterministic ตาม `--seed`, เขียน `splits.yaml` ที่มี `test/val/train_faro/train_lidar` แต่ละรายการ `{video_id, visit_id, fold, has_faro}` + `seed` + `generated_by`
**Acceptance:**
- [ ] `splits.yaml`: |T|=6 (hardcode + visit), |V|=3, |A|=10, |B|=14; ไม่มี `visit_id` ซ้ำข้าม set; A/B ทั้งหมด `fold: Training`, V `fold: Validation`; A/V ทั้งหมด `has_faro: true`
- [ ] script exit ≠ 0 ถ้า candidates ไม่พอ หรือพบ visit ของ T
- [ ] เกณฑ์ screening เดิม (scan 60–120 s, `hi_MB ≥ 100` สำหรับ Faro) ใช้กับ A/V; B ใช้แค่ scan 60–120 s
**Verification:** รันคำสั่ง SPEC §3 "ข้อมูล" 3 บรรทัดแรก · `python -c` นับ set ตาม acceptance · `make lint` · commit `splits.yaml`
**Dependencies:** T2 · **Where:** WSL (ต้องต่อเน็ตไป CDN ของ Apple; ทำบน Mac ก็ได้แล้ว commit)
**Files:** `scripts/screen_scenes.py`, `scripts/select_training_scenes.py`, `configs/training/splits.yaml` · **Scope:** S

#### Task 7: `scripts/download_training_scenes.py` + `--verify`
**Description:** อ่าน `splits.yaml` (ข้าม `test`), ต่อฉาก: curl zip `vga_wide lowres_depth confidence` + `lowres_wide.traj` + `lowres_wide_intrinsics.zip` (+ `highres_depth` ถ้า `has_faro`) จาก `…/raw/<fold>/<video_id>/…` → unzip ลง `<root>/raw/<fold>/<video_id>/` → ลบ zip; ข้ามที่มีแล้ว (idempotent); `--verify`: เปิดทุกฉากด้วย `open_training_scene()` (T3), นับเฟรม Faro/LiDAR, `du`, และ **fail ถ้ามีโฟลเดอร์ `raw/*/<video_id>` ที่ visit อยู่ใน T**; พิมพ์ตาราง markdown
**Acceptance:**
- [ ] รัน 2 ครั้งติดกัน ครั้งที่ 2 ไม่โหลดซ้ำ
- [ ] `--verify` ผ่านครบ 27 ฉาก, ตารางถูกบันทึกเป็น `experiments/training/DATA.md` (commit); `du -sh ~/data/arkitscenes` ≤ 50 GB
- [ ] ไม่มี zip ค้าง (`find … -name "*.zip"` ว่าง)
**Verification:** `nohup python scripts/download_training_scenes.py … &` แล้ว `--verify` · `git status` ไม่มีอะไรใต้ `data/`
**Dependencies:** T3 (ใช้ `open_training_scene`), T6 · **Where:** WSL (2–4 ชม. รอเน็ต — เริ่มทันทีที่ T6 เสร็จ แล้วไปทำ T8/T9 ระหว่างรอ)
**Files:** `scripts/download_training_scenes.py`, `experiments/training/DATA.md` · **Scope:** S

### Checkpoint B — ข้อมูลพร้อม
- [ ] `DATA.md` แสดงเฟรม Faro ≈ 300–500/ฉาก (A, V) และเฟรม vga ≈ 2,700/ฉาก (ทุกฉาก)
- [ ] ฉากใน T ไม่มีบนดิสก์ (verify ยืนยัน)

### Phase 3 — Train loop slice (Mac เขียน/เทสต์ CPU → WSL รัน GPU)

#### Task 8: `training/train.py` — loop ขั้นต่ำ + `configs/training/base.yaml`, `r0_smoke.yaml`
**Description:** CLI `--config` (+ `--set k=v` dotlist ผ่าน `roomscan.config`); โหลด `AutoModelForDepthEstimation` จาก `model.hf_id`; freeze `backbone`; `DepthPairDataset` จาก `splits.yaml` + `data.scenes` (ชื่อ set: `train_faro` / `train_faro+train_lidar`); AdamW เฉพาะ param ที่ `requires_grad`; warmup+cosine; `torch.autocast(device_type, dtype=bfloat16)` เมื่อ cuda; grad accumulation; `log.jsonl` (บรรทัดแรก = env: device, torch, n_frames ต่อ set, config hash); เซฟ `last/` ด้วย `save_pretrained` ทุก `save_every` และตอนจบ; `max_steps` หยุดได้; seed; บันทึก `config.yaml` ที่ resolve แล้วลง run dir
**Acceptance:**
- [ ] `python -m roomscan.training.train --config configs/training/r0_smoke.yaml --set model.hf_id=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf data.root=<synthetic> train.max_steps=2 train.device=cpu` จบใน < 3 นาที บน Mac, ได้ `experiments/training/r0_smoke/{config.yaml,log.jsonl,last/}`
- [ ] `log.jsonl` มี 1 env row + 2 step rows (`step, loss, lr, sec_per_step`)
- [ ] weight ของ `backbone` ก่อน/หลัง เท่ากันทุก tensor; ของ `head` มีอย่างน้อย 1 tensor เปลี่ยน
- [ ] `last/` โหลดกลับด้วย `AutoModelForDepthEstimation.from_pretrained` และ predict ได้ (H, W) float32
**Verification:** `pytest tests/test_training.py -k train_two_steps` (importorskip torch; skip ถ้าโหลด HF ไม่ได้) · `make lint`
**Dependencies:** T3, T4, T5 · **Where:** Mac
**Files:** `src/roomscan/training/train.py`, `configs/training/base.yaml`, `configs/training/r0_smoke.yaml`, `tests/test_training.py` · **Scope:** M

#### Task 9: `training/validate.py` + best/ + step-0 baseline
**Description:** `validate(model, scenes_V, transform=EvalTransform, stride=5, device)` → forward ทั้งเฟรม (518×686) → pred resize nearest กลับ 640×480 → `depth_metrics`/`mean_metrics` จาก `evaluation.metrics_2d` → dict `val/abs_rel, val/delta1, val/rmse, val/n_frames`; ต่อเข้า `train.py`: รันก่อน step 1 (บันทึกเป็น `step: 0`), ทุก `val_every`, และตอนจบ; เก็บ `best/` เมื่อ `val/abs_rel` ต่ำสุด; `configs/training/r1_ft_faro.yaml`, `r2_ft_lidar.yaml`, `r3_ft_lidar_all.yaml`
**Acceptance:**
- [ ] บน synthetic: `validate()` คืนค่า finite ทุกตัว, `n_frames == len(indices[::5])`; ถ้าป้อน pred = gt ผ่าน mock model → abs_rel == 0
- [ ] `log.jsonl` มีแถว `step 0` ที่มี `val/*` ก่อนแถว `step 1`; `best/` ถูกสร้างเมื่อ val ดีขึ้น และไม่ถูกเขียนทับเมื่อแย่ลง
- [ ] r1/r2 ใช้ `max_steps` เท่ากัน (6000), scenes = `train_faro`; r3 = 12000, scenes = `train_faro+train_lidar`; r2/r3 `target: lidar`, `lidar_min_confidence: 1`
**Verification:** `pytest tests/test_training.py -k validate` · `make lint` · `make test`
**Dependencies:** T8 · **Where:** Mac
**Files:** `src/roomscan/training/validate.py`, `src/roomscan/training/train.py`, `configs/training/r1_ft_faro.yaml`, `configs/training/r2_ft_lidar.yaml`, `configs/training/r3_ft_lidar_all.yaml`, `tests/test_training.py` · **Scope:** M (6 ไฟล์ แต่ config 3 ไฟล์เป็น override สั้น ๆ)

#### Task 10: `R0_smoke` บน RTX 3070 (gate)
**Description:** รัน Large 30 steps บน 1 ฉากของ A (Faro) ด้วย `batch 2 × accum 4`, bf16; ดู `nvidia-smi` peak; ถ้า OOM → fallback ตามลำดับ: (1) `batch 1 × accum 8`, (2) `model.neck` gradient checkpointing, (3) input 434 (31×14) — บันทึกว่าใช้ข้อไหนลง `base.yaml` comment + ENV.md
**Acceptance:**
- [ ] จบ 30 steps ไม่ OOM; peak VRAM < 7.5 GB (บันทึกตัวเลขจริง)
- [ ] `loss` step 30 < step 1; `sec_per_step` บันทึกแล้ว (ใช้ประมาณเวลา R1–R3 ใหม่)
- [ ] val step 0 (pretrained) บน V ให้ `abs_rel` ในช่วง 0.2–0.4 (sanity เทียบ failure analysis 0.30 — ถ้าห่างมาก = bug data ก่อนไปต่อ)
**Verification:** `make train RUN=configs/training/r0_smoke.yaml` (foreground ได้ < 10 นาที) · `cat experiments/training/r0_smoke/log.jsonl` · commit log + config
**Dependencies:** T7, T9 · **Where:** WSL
**Files:** `configs/training/base.yaml` (ถ้าต้อง fallback), `experiments/training/r0_smoke/{config.yaml,log.jsonl}`, `experiments/training/ENV.md` · **Scope:** XS

### Checkpoint C — Gate ก่อนรันจริง
- [ ] R0 ผ่านทั้ง 3 ข้อ · `make test`/`make lint` เขียวบน WSL · push branch `training` (Mac pull ได้)
- [ ] Review ตัวเลข step-0 กับผู้ใช้ก่อนใช้ GPU หลายชั่วโมง

### Phase 4 — Runs + ส่งกลับ

#### Task 11: `R1_ft_faro`
**Description:** `nohup make train RUN=configs/training/r1_ft_faro.yaml > experiments/training/r1_ft_faro.out 2>&1 &` (~1 ชม.); ตรวจ `tail -f log.jsonl`
**Acceptance:**
- [ ] จบ 6000 steps; `best/` มี `val/abs_rel` < step 0 (ถ้าไม่ → หยุด, debug data ตาม SPEC §10 ข้อ 5)
- [ ] เป้าสัญญาณ: `val/abs_rel` best < 0.10
**Verification:** `python - <<EOF` อ่าน log.jsonl หา min val/abs_rel และ step · commit `config.yaml` + `log.jsonl`
**Dependencies:** T10 · **Where:** WSL · **Files:** `experiments/training/r1_ft_faro/{config.yaml,log.jsonl}` · **Scope:** XS

#### Task 12: `R2_ft_lidar`
**Description:** เหมือน T11 ด้วย `r2_ft_lidar.yaml` (ฉาก A เดียวกัน, label LiDAR, steps เท่ากัน) — รันหลัง R1 จบ (ห้ามซ้อน GPU)
**Acceptance:**
- [ ] จบ 6000 steps; best `val/abs_rel` < step 0; บันทึกอัตราส่วน R2/R1 (เป้าสัญญาณ ≤ 1.5×)
**Verification:** เหมือน T11 · **Dependencies:** T11 · **Where:** WSL · **Files:** `experiments/training/r2_ft_lidar/…` · **Scope:** XS

#### Task 13: `R3_ft_lidar_all`
**Description:** เหมือน T12 ด้วย `r3_ft_lidar_all.yaml` (A ∪ B, 12000 steps, ~2 ชม.)
**Acceptance:**
- [ ] จบ; best `val/abs_rel` ≤ R2 (ถ้าแย่กว่า R2 = finding ที่ต้องรายงาน ไม่ใช่ bug — ตรวจ log ว่า B ถูกโหลดจริง `n_frames` ≈ 22k)
**Verification:** เหมือน T11 · **Dependencies:** T12 · **Where:** WSL · **Files:** `experiments/training/r3_ft_lidar_all/…` · **Scope:** XS

#### Task 14: `training/push.py` + `models/depth_anything.py` `hf_id=` + registry (ทำระหว่างรอ T11–T13)
**Description:** `push.py --run <dir> --repo <id> --private [--which best|last]` → `huggingface_hub.HfApi.create_repo(private=True, exist_ok=True)` + `upload_folder(best/)` + README model card สั้น (run, split seed, val abs_rel, config hash); `DepthAnythingV2Metric.__init__(size=…, hf_id: str | None = None, device)` — ถ้าให้ `hf_id` ใช้แทน `HF_IDS[size]` และ `name = f"depth_anything_v2_{size}"`; registry 3 บรรทัด `depth_anything_v2_ft_faro / _ft_lidar / _ft_lidar_all` → `{"size": "ft_faro", "hf_id": "<hf_user>/roomscan-dav2-metric-large-ft-faro"}` (ชื่อ user ใส่ตอน T15; ก่อนนั้นเป็น placeholder ที่ test mock)
**Acceptance:**
- [ ] `build_depth_model("depth_anything_v2_ft_faro")` เมื่อ `hf_id` ชี้ path local (`last/` จาก T8 test) → สร้างได้, `is_metric is True`, `output_kind == "depth"`, `predict(rgb)` คืน (H, W) float32
- [ ] `push.py --dry-run` แสดงไฟล์ที่จะอัปโหลดโดยไม่ต่อเน็ต
- [ ] ไม่มี token/ชื่อ user ใน repo นอกจาก registry hf_id (public identifier ไม่ใช่ secret)
**Verification:** `pytest tests/test_training.py -k registry` · `make lint` · `make test`
**Dependencies:** T8 · **Where:** Mac
**Files:** `src/roomscan/training/push.py`, `src/roomscan/models/depth_anything.py`, `src/roomscan/models/__init__.py`, `tests/test_training.py` · **Scope:** S

#### Task 15: ส่ง checkpoint + smoke บน Mac + push branch
**Description:** WSL: `push.py` ทั้ง 3 run ขึ้น HF private; แก้ registry ใส่ `<hf_user>` จริง; commit log/config ทั้งหมด; `git push -u origin training`. Mac: `git pull`, `.venv/bin/python -c "from roomscan.models import build_depth_model; m=build_depth_model('depth_anything_v2_ft_lidar'); …predict(frame.rgb ของ 42444474 เฟรม 0)"` → พิมพ์ median depth (เมตร) เทียบ `gt_depth` median
**Acceptance:**
- [ ] HF Hub มี 3 repo private, แต่ละอันโหลดด้วย `from_pretrained` ได้บน Mac (MPS)
- [ ] median(pred)/median(gt) บนเฟรมเดียวของ T อยู่ใน [0.7, 1.4] (sanity เท่านั้น — ไม่ใช่ผล)
- [ ] `git status` สะอาดทั้งสองเครื่อง; `git count-objects -vH` size-pack โตจาก main < 5 MB; ไม่มี `*.safetensors` ใน history (`git log --all --stat | grep safetensors` ว่าง)
**Verification:** คำสั่งข้างบน · SPEC §10 ทุกข้อติ๊กได้
**Dependencies:** T13, T14 · **Where:** WSL แล้ว Mac
**Files:** `src/roomscan/models/__init__.py`, `experiments/training/*/…` · **Scope:** S

### Checkpoint D — Complete
- [ ] SPEC §10 success criteria 1–9 ครบ
- [ ] `tasks/todo.md` ติ๊กครบ; ส่งต่อไป spec ฝั่ง Mac (Level 0 Depth Pro + `exp6_finetune.yaml` + เปเปอร์)

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large + 518² ไม่พอ 8 GB แม้ freeze | High | T10 gate ก่อนรันจริง; fallback 3 ขั้นตามลำดับ (batch 1 / grad-ckpt neck / 434 px); ทางหนีสุดท้าย = Colab เฉพาะ R1–R3 ด้วยโค้ดเดิม |
| ประมาณเวลา 0.6 s/step ผิด (อาจ 1.5 s) | Med | ใช้ `sec_per_step` จริงจาก R0 คำนวณใหม่; ลด `max_steps` R3 ก่อน R1/R2 |
| Faro ↔ vga_wide timestamp ไม่แมตช์ใน tol 0.02 s | Med | vga 30 fps → ระยะห่างสูงสุด 0.017 s < 0.02 ✓; `--verify` รายงาน % เฟรม Faro ที่ไม่มี vga คู่ (ต้อง < 5 %) |
| LiDAR label sparse/ผิดตรงกระจก → โมเดลเรียนความผิด | Med (คือ finding) | conf mask ≥ 1 default; R4 conf = 2 เป็น optional; รายงานเป็นผล ไม่ใช่ bug |
| Overfit 10 ฉาก (R1/R2) | Med | val ทุก 500 steps + best/ ตาม val; รายงาน curve; ไม่ยืด steps เกิน 6k |
| `sky_direction` ของฉาก B/V บางฉากไม่มีใน metadata | Low | loader fallback "Up" อยู่แล้ว; `--verify` list ฉากที่ fallback |
| โหลด HF model ใน WSL ช้า/ล้ม | Low | `HF_HOME` บน Linux fs; โหลด Large ครั้งเดียว (1.3 GB) ก่อน T10 |
| Apple CDN ช้า → T7 นานกว่า 4 ชม. | Low | detach + idempotent; เริ่ม A+V ก่อน (จำเป็นสำหรับ R0–R2) แล้ว B ตาม |
| ผู้ใช้เผลอโหลดฉาก T มาเครื่องเทรน | High (leakage) | `--verify` fail hard; `download_training_scenes.py` ปฏิเสธ video_id ที่ visit อยู่ใน T |
| Mac ไม่มี CUDA → test train 2 steps ช้า | Low | ใช้ Small + synthetic 6 เฟรม + 2 steps, CPU < 3 นาที; ถ้า HF ออฟไลน์ skip |

## Open questions
- Q1 `lidar_min_confidence` default 1 (ตาม SPEC §11) — ยืนยันตอน T9
- Q3 `<hf_user>` — ใส่ตอน T15
- ต้องการ `R4_ft_lidar_conf2` / `R5_base_ft_lidar_all` ไหม — ตัดสินหลังเห็น R2/R3
