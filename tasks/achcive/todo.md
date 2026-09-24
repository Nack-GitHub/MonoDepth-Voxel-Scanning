# TODO — Fine-tune DA-v2 ด้วยครู LiDAR (ฝั่งเครื่องเทรน)

รายละเอียดแต่ละ task: `tasks/plan.md` · spec: `SPEC.md` · branch: `training`
Where: 🍎 = Mac ทำได้ · 🪟 = ต้องทำบน WSL2/RTX 3070

## Phase 0 — Foundation
- [x] T1 🍎 ADR-013 + `pyproject[train]` + `make train` + README §8 + CLAUDE.md
  - Acceptance: ADR-013 supersede 008 พร้อมหัวข้อ Split; `pip install -e ".[train]"` ได้ torch/transformers/huggingface_hub/safetensors
  - Verify: `make lint && make test`
  - Files: `docs/architecture/adr-013-finetune-with-lidar-teacher.md`, `docs/architecture/README.md`, `pyproject.toml`, `Makefile`, `CLAUDE.md`
- [x] T2 🪟 WSL2 env: clone → branch `training` → venv 3.12 → torch cu124 → `pip install -e ".[train,dev]"` → symlink data → `hf auth login`
  - Acceptance: `torch.cuda.is_available()` True; `make synthetic && make test && make lint` ผ่าน; `readlink data/arkitscenes` → `~/data/arkitscenes`
  - Verify: บันทึก `nvidia-smi` + versions ลง `experiments/training/ENV.md`
  - Files: `experiments/training/ENV.md`

## Phase 1 — Data slice (synthetic)
- [x] T3 🍎 `training/pairs.py` (numpy: `DepthPair`, `make_pair`, `open_training_scene`, `iter_indices`) + `training/dataset.py` (torch `DepthPairDataset`)
  - Acceptance: faro/lidar pair บน synthetic ถูก shape/หน่วย/mask; `training/` ไม่ import pipeline/geometry/open3d
  - Verify: `pytest tests/test_training.py -k "pair or dataset"`
  - Files: `src/roomscan/training/{__init__,pairs,dataset}.py`, `tests/test_training.py`
- [x] T4 🍎 `training/transforms.py`: upright, resize_short 518 (÷14, depth nearest), random_crop, hflip, color_jitter, normalize; `TrainTransform`/`EvalTransform`
  - Acceptance: depth values ⊆ ต้นทางหลัง resize; crop/flip ไม่เปลี่ยน histogram; 640×480 → 518×686
  - Verify: `pytest tests/test_training.py -k transform`
  - Files: `src/roomscan/training/transforms.py`, `tests/test_training.py`
- [x] T5 🍎 `training/loss.py`: `silog` (β 0.15, ×10), `l1`, `DepthLoss`
  - Acceptance: silog(gt,gt)=0; ลงโทษ scale; mask ว่าง → 0 ไม่ NaN; grad finite
  - Verify: `pytest tests/test_training.py -k loss`
  - Files: `src/roomscan/training/loss.py`, `tests/test_training.py`

### ✅ Checkpoint A
- [x] `make test` เขียว (Mac, torch) · training tests = skipped ใน env ไม่มี torch · `make lint` · commit · review กับผู้ใช้

## Phase 2 — Scenes + download (🪟, รอเน็ต)
- [x] T6 `screen_scenes.py --fold` + `scripts/select_training_scenes.py` → `configs/training/splits.yaml`
  - Acceptance: T=6 / V=3 (Validation, Faro) / A=10 (Training, Faro) / B=14 (Training); ไม่มี visit ซ้ำ; fail hard ถ้าพบ visit ของ T
  - Verify: รันคำสั่ง SPEC §3; นับ set; commit `splits.yaml`
  - Files: `scripts/screen_scenes.py`, `scripts/select_training_scenes.py`, `configs/training/splits.yaml`
- [x] T7 `scripts/download_training_scenes.py` (+ `--verify`) — `nohup … &` 2–4 ชม.; ทำ T8/T9 ระหว่างรอ
  - Acceptance: idempotent; 27 ฉากผ่าน verify; ≤ 50 GB; ไม่มี zip ค้าง; ไม่มีฉาก T บนดิสก์
  - Verify: `--verify` → `experiments/training/DATA.md` (commit)
  - Files: `scripts/download_training_scenes.py`, `experiments/training/DATA.md`

### ✅ Checkpoint B
- [x] `DATA.md`: Faro ≈ 300–500 เฟรม/ฉาก (A,V), vga ≈ 2,700/ฉาก · ฉาก T ไม่อยู่บนดิสก์

## Phase 3 — Train loop
- [x] T8 🍎 `training/train.py` + `configs/training/base.yaml` + `r0_smoke.yaml`: freeze backbone, AdamW, warmup+cosine, bf16 autocast, grad-accum, `log.jsonl`, `last/`
  - Acceptance: Small + synthetic + 2 steps บน CPU < 3 นาที; backbone ไม่เปลี่ยน, head เปลี่ยน; `last/` โหลดกลับได้
  - Verify: `pytest tests/test_training.py -k train_two_steps`
  - Files: `src/roomscan/training/train.py`, `configs/training/base.yaml`, `configs/training/r0_smoke.yaml`, `tests/test_training.py`
- [x] T9 🍎 `training/validate.py` + best/ + step-0 row + `r1_ft_faro.yaml`, `r2_ft_lidar.yaml`, `r3_ft_lidar_all.yaml`
  - Acceptance: val ใช้ `metrics_2d` เดิม; `step 0` มี `val/*`; `best/` เขียนเฉพาะเมื่อดีขึ้น; r1=r2 6000 steps บน A, r3 12000 บน A∪B
  - Verify: `pytest tests/test_training.py -k validate && make test && make lint`
  - Files: `src/roomscan/training/validate.py`, `src/roomscan/training/train.py`, `configs/training/r{1,2,3}_*.yaml`, `tests/test_training.py`
- [x] T10 🪟 `R0_smoke` (Large, 30 steps, 1 ฉาก A)
  - Acceptance: ไม่ OOM, peak VRAM < 7.5 GB; loss step30 < step1; step-0 val abs_rel ≈ 0.2–0.4
  - Verify: `make train RUN=configs/training/r0_smoke.yaml`; commit log/config; อัปเดต ENV.md (VRAM, sec/step, fallback ที่ใช้)
  - Files: `configs/training/base.yaml` (ถ้า fallback), `experiments/training/r0_smoke/*`, `experiments/training/ENV.md`

### ✅ Checkpoint C
- [x] R0 ผ่าน · `make test`/`make lint` เขียวบน WSL · push branch · review step-0 กับผู้ใช้ก่อนใช้ GPU หลายชั่วโมง

## Phase 4 — Runs + ส่งกลับ
- [x] T11 🪟 `R1_ft_faro` (`nohup`, ~1 ชม.) — best val/abs_rel < step 0 (เป้า < 0.10) · commit log
- [x] T12 🪟 `R2_ft_lidar` (หลัง R1 จบ) — best < step 0; บันทึก R2/R1 (เป้า ≤ 1.5×) · commit log
- [x] T13 🪟 `R3_ft_lidar_all` (~2 ชม.) — best ≤ R2 หรือรายงานเป็น finding; `n_frames` ≈ 22k · commit log
- [x] T14 🍎 `training/push.py` + `DepthAnythingV2Metric(hf_id=)` + registry `depth_anything_v2_ft_{faro,lidar,lidar_all}` (ทำระหว่างรอ T11–T13)
  - Acceptance: `build_depth_model("depth_anything_v2_ft_faro")` กับ hf_id = path local ใช้ได้, `is_metric`; `push.py --dry-run` ไม่ต่อเน็ต
  - Verify: `pytest tests/test_training.py -k registry && make test && make lint`
  - Files: `src/roomscan/training/push.py`, `src/roomscan/models/depth_anything.py`, `src/roomscan/models/__init__.py`, `tests/test_training.py`
- [x] T15 🪟→🍎 push 3 checkpoint ขึ้น HF private; ใส่ `<hf_user>` ใน registry; commit + `git push -u origin training`; บน Mac pull แล้ว predict 1 เฟรมของ T (median ratio ∈ [0.7, 1.4])
  - Verify: `git status` สะอาด; ไม่มี safetensors ใน history; SPEC §10 ครบ

### ✅ Checkpoint D — Complete
- [x] SPEC §10 ข้อ 1–9 ครบ (2026-09-23) — checkpoint ทั้ง 3 ตัวโหลดบน Mac ได้, predict ฉาก T ผ่านเกณฑ์ median ratio

## ฝั่ง Mac (ต่อจาก Checkpoint D)
- [x] `exp6_finetune` 4 แถว × 6 ฉากเทสต์ → `experiments/results/exp6_finetune/` + `paper/analysis/finetune_teacher.md`
  - ผล: pretrained 54.3 cm → ft_faro 14.6 / ft_lidar 16.6 / ft_lidar_all 15.7 cm (AbsRel 0.382 → 0.134–0.136)
- [x] รัน exp1 ใหม่หลัง PR #1 (การหมุนภาพ) — 4 ฉาก `Left` ขยับ, 2 ฉาก `Up`/`Down` ไม่ขยับ
- [x] เปเปอร์: abstract, intro, setup §4.5, results §5.6/Exp6, discussion, business, limitations, conclusion + `numbers.tex` (ทั้งไทยและอังกฤษ)
- [x] รัน exp3 (mono) + exp4 ใหม่หลังการหมุนภาพ → §5.3/§5.4 + "ขนาดโมเดลมีผลน้อยกว่า scale" 6 เท่า → **17 เท่า**
- [x] regenerate `paper/analysis/` scale drift (12 ไฟล์) + บันทึกส่วนต่างท้าย `42444474_failure_analysis.md`
- [x] `make figures` + `make paper-figures` ใหม่ + รูป `exp6_finetune_topdown_grid.png` เข้าเปเปอร์แล้ว
- [x] Level 0 baseline: `mono_depth_pro` (Depth Pro + focal จริง 533 px) รันครบ 6 ฉาก → **155.4 ± 39.0 cm, F@5cm 0.00**
  - wrapper ตรวจกับห้องสังเคราะห์แล้ว (AbsRel 0.008 หลัง scale เดียว) — ตัวเลขนี้เป็นผลของโมเดลบนภาพ VGA ระยะใกล้ ไม่ใช่บั๊ก
  - เข้าเปเปอร์แล้วทั้งตาราง Exp6, §5.6, related work, intro, limitations (ทั้งไทยและอังกฤษ)
  - ถ้าจะเสริม: UniDepth v2 / Metric3D v2 หรือทดสอบ Depth Pro ด้วย reference implementation ของ Apple
