# Spec: Fine-tune Depth Anything V2 บน ARKitScenes ด้วยครู LiDAR — ฝั่งเครื่องเทรน (Windows / WSL2 / RTX 3070)

สถานะ: **draft รอ approve** (2026-09-21, rev 2: split ตาม §5.2) · เขียนตาม `agent-skills:spec-driven-development`
ขอบเขต: เฉพาะงานที่ทำบนเครื่อง Windows — เขียนโค้ดเทรน, โหลดข้อมูลเทรน, เทรน, ส่ง checkpoint ขึ้น HF Hub
งานฝั่ง Mac (Level 0 Depth Pro, `exp6_finetune.yaml`, ประเมิน, อัปเดตเปเปอร์) จะเป็น spec แยกหลังจากมี checkpoint แรก

---

## 1. Objective

**สร้างอะไร:** training loop ที่เอา DA-v2 Metric-Indoor (Large) มา fine-tune บนห้องจริงจาก ARKitScenes โดยใช้ depth จาก LiDAR ของ iPad เป็น label (และ Faro เป็น label เปรียบเทียบ) แล้วส่ง weight ที่ได้กลับมาให้ pipeline เดิมประเมินได้เหมือน model ตัวอื่นทุกประการ

**ทำไม:** Exp1 ชี้ว่า DA-v2 metric พังบนห้องจริง (chamfer 49 cm) จาก scale ต่อเฟรมแกว่ง ±40 % ทั้งที่รูปทรงถูก (oracle 6.5 cm) — สาเหตุคือ domain gap (เทรนบน Hypersim สังเคราะห์) ดู `paper/analysis/42444474_failure_analysis.md` §2
คำถามวิจัยใหม่ของเปเปอร์: *"ใช้ LiDAR จาก iPhone/iPad เป็นครู สอนโมเดล mono ให้ตอบเป็นเมตรบนเครื่องที่ไม่มี LiDAR ได้ดีแค่ไหน และเสียจากครูสมบูรณ์ (Faro) เท่าไหร่"*

**ผู้ใช้ผลลัพธ์:** pipeline `roomscan` บน Mac ผ่าน registry `models/` (ไม่ต้องรู้ว่าเทรนมายังไง)

**สิ่งที่ ADR-008 ต้องเปลี่ยน:** เขียน **ADR-013** supersede ADR-008 ("pretrained-only") — อนุญาต training loop ใน repo ภายใต้กฎในเอกสารนี้

**Runs ที่ต้องได้ (ทั้งหมด Large, encoder freeze):**

| run | label ตอนเทรน | ฉากเทรน | ตอบคำถาม |
|---|---|---|---|
| `R0_smoke` | Faro | 1 ฉาก, 30 steps | โค้ดรันจบบน 8 GB, loss ลด |
| `R1_ft_faro` | Faro (highres_depth) | 10 ฉาก set **A** | เพดานของ fine-tune (ครูสมบูรณ์) |
| `R2_ft_lidar` | LiDAR (lowres_depth, conf ≥ 1) | 10 ฉาก set **A** (ฉากเดียวกับ R1) | ครู noisy เสียเท่าไหร่ — เทียบ R1 แบบแฟร์ |
| `R3_ft_lidar_all` | LiDAR | 24 ฉาก set **A ∪ B** | ข้อได้เปรียบของ LiDAR: ข้อมูลหาได้เยอะกว่า |

ทุก run วัดระหว่างเทรนด้วย Faro บน val set **V** (3 ฉากจาก fold Validation) และวัดผลจริงบน Mac ด้วย test set **T** (6 ฉากเดิมของ Exp1) ซึ่ง **โมเดลห้ามเห็นเด็ดขาด**

---

## 2. Tech stack

| | |
|---|---|
| OS / runtime | Windows 11 + **WSL2 Ubuntu 22.04+**, NVIDIA driver บน Windows ที่รองรับ WSL (`nvidia-smi` ใน WSL ต้องเห็น RTX 3070) |
| GPU | RTX 3070, **VRAM 8 GB** (ตัวจำกัดหลัก) — bf16 autocast, batch 2, grad accumulation |
| Python | 3.12 ใน `./.venv` (เหมือน Mac) |
| Deps เพิ่ม | `torch` (CUDA 12.x build), `transformers>=4.40`, `huggingface_hub`, `safetensors`, `accelerate` (ไม่บังคับ) — เพิ่มเป็น extra `train` ใน `pyproject.toml` |
| **ไม่ต้องติดตั้ง** | Open3D (training ไม่แตะ 3D), fastapi — ใช้ `pip install -e ".[train,dev]"` ไม่ใช่ `[mono,dev,web]` |
| โมเดลฐาน | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` (HF, `AutoModelForDepthEstimation`) — ตัวเดียวกับ `models/depth_anything.py` ใช้อยู่ |
| Dataset | ARKitScenes raw (`vga_wide`, `lowres_depth`, `confidence`, `highres_depth`, `lowres_wide.traj`, `lowres_wide_intrinsics`) — **ไม่โหลด `color` 1920×1440** (ประหยัด 2–4 GB/ฉาก; RGB ใช้ `vga_wide` 640×480 ซึ่งพอสำหรับ input 518 px) |
| ส่ง checkpoint | HuggingFace Hub **private** repo (ฟรี) `<hf_user>/roomscan-dav2-metric-large-<run>` |
| Git | branch `training` จาก `main`, push ขึ้น `origin` — Mac pull มาใช้ (ห้าม push จากเครื่อง Mac โดยไม่ถูกสั่ง กฎเดิม) |

**WSL2 ข้อบังคับ:** เก็บข้อมูลและ venv บน **Linux filesystem** (`~/data/arkitscenes`, `~/MyProjects/...`) — ห้ามวางบน `/mnt/c/...` (I/O ช้า 10×) แล้ว symlink `data/arkitscenes -> ~/data/arkitscenes` (ทั้งโฟลเดอร์ `data/` ถูก gitignore อยู่แล้ว)

---

## 3. Commands (รันใน WSL2 ทั้งหมด, cwd = root ของ repo)

```bash
# ---- setup ครั้งเดียว ----
git clone https://github.com/Nack-GitHub/MonoDepth-Voxel-Scanning.git ~/MyProjects/MonoDepth-Voxel-Scanning
cd ~/MyProjects/MonoDepth-Voxel-Scanning && git checkout -b training
python3.12 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e ".[train,dev]"
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True, RTX 3070
mkdir -p ~/data/arkitscenes && ln -s ~/data/arkitscenes data/arkitscenes
.venv/bin/hf auth login          # token สร้างเองที่ huggingface.co/settings/tokens (write) — ห้ามใส่ token ในไฟล์ใด ๆ ใน repo

# ---- ข้อมูล ----
make synthetic                                                    # ฉากสังเคราะห์สำหรับ test (ไม่ต้องมีข้อมูลจริง)
.venv/bin/python scripts/screen_scenes.py --n 80 --seed 1 > data/arkitscenes/candidates_train_seed1.csv
.venv/bin/python scripts/screen_scenes.py --n 20 --seed 1 --fold Validation > data/arkitscenes/candidates_val_seed1.csv
.venv/bin/python scripts/select_training_scenes.py --metadata data/arkitscenes/raw/metadata.csv \
    --train-candidates data/arkitscenes/candidates_train_seed1.csv --val-candidates data/arkitscenes/candidates_val_seed1.csv \
    --exclude-test configs/experiments/exp1_depth_source.yaml --n-faro 10 --n-lidar-only 14 --n-val 3 --seed 0 \
    --out configs/training/splits.yaml                            # A(10, Training, Faro) B(14, Training, LiDAR) V(3, Validation, Faro); กัน visit_id ของ T
.venv/bin/python scripts/download_training_scenes.py --splits configs/training/splits.yaml --root data/arkitscenes
.venv/bin/python scripts/download_training_scenes.py --splits configs/training/splits.yaml --root data/arkitscenes --verify   # นับเฟรม/ตรวจ zip ครบ + assert ไม่มี visit ของ T

# ---- เทรน ----
make train RUN=configs/training/r0_smoke.yaml        # = .venv/bin/python -m roomscan.training.train --config <yaml>
make train RUN=configs/training/r1_ft_faro.yaml
make train RUN=configs/training/r2_ft_lidar.yaml
make train RUN=configs/training/r3_ft_lidar_all.yaml
# run ยาว (> 10 นาที) ต้อง detach เหมือนกฎ sweep บน Mac:
nohup make train RUN=configs/training/r1_ft_faro.yaml > experiments/training/r1_ft_faro.out 2>&1 &
tail -f experiments/training/r1_ft_faro/log.jsonl

# ---- ส่งกลับ ----
.venv/bin/python -m roomscan.training.push --run experiments/training/r1_ft_faro --repo <hf_user>/roomscan-dav2-metric-large-ft-faro --private
git add configs/training experiments/training/*/config.yaml experiments/training/*/log.jsonl && git commit && git push -u origin training

# ---- คุณภาพ ----
make test          # pytest — ต้องผ่านโดยไม่มีข้อมูลจริง; training tests ถูก skip ถ้าไม่มี torch
make lint          # ruff
```

`make train` เป็น target ใหม่ใน `Makefile` (บรรทัดเดียว) — ทุก target อื่นใช้บน WSL ได้ตามเดิม

---

## 4. Project structure (ไฟล์ใหม่/แก้ — เคารพ dependency rule §4 ของ architecture README)

```
docs/architecture/adr-013-finetune-with-lidar-teacher.md   supersede ADR-008; กฎ split, กฎ weight ไม่ commit
src/roomscan/training/                    ← import ได้เฉพาะ dataio, evaluation.metrics_2d, types  (ห้าม import pipeline / geometry / open3d)
├── __init__.py
├── dataset.py     DepthPairDataset(torch Dataset): ARKitScenesScene → (rgb 518², depth m, mask) ต่อเฟรม; target="faro"|"lidar"
├── transforms.py  resize สั้น 518 (หาร 14 ลงตัว), random crop 518², hflip, color jitter; depth ใช้ nearest — ห้าม scale/zoom aug
├── loss.py        silog(pred, target, mask) (+ l1 option)
├── train.py       loop: freeze backbone → AdamW → bf16 → val ทุก N steps → best/ + last/ ; CLI --config
├── validate.py    abs_rel / δ1 / rmse บน val scenes ทั้งเฟรม (ไม่ crop) ใช้ evaluation/metrics_2d.py เดิม
└── push.py        save_pretrained + processor → HF Hub private
src/roomscan/models/depth_anything.py     เพิ่ม hf_id= kwarg ให้ DepthAnythingV2Metric (โหลด repo ที่เทรนแล้ว)
src/roomscan/models/__init__.py           +3 บรรทัด registry: depth_anything_v2_ft_faro / _ft_lidar / _ft_lidar_all
configs/training/
├── base.yaml            model, optimizer, schedule, batch, max_depth, lidar_min_confidence, val_every
├── splits.yaml          test T (6 เดิม, hardcode + visit_id), val V (3, fold Validation), train A (10), train B (14) — commit
├── r0_smoke.yaml … r3_ft_lidar_all.yaml   override จาก base (target, scenes, max_steps)
scripts/screen_scenes.py                  เพิ่ม --fold {Training,Validation} (default Training เหมือนเดิม)
scripts/select_training_scenes.py         อ่าน metadata + candidates → กัน visit_id ของ T → เลือก A/B จาก Training, V จาก Validation → splits.yaml
scripts/download_training_scenes.py       curl zip ตาม URL เดียวกับ download_data.py ของ Apple (split ตาม fold) → unzip → ลบ zip; --verify
experiments/training/<run>/               config.yaml + log.jsonl (commit) ; best/ last/ (weights — gitignore)
tests/test_training.py                    ใช้ data/synthetic (มี lidar+gt ใน layout ARKitScenes อยู่แล้ว)
pyproject.toml                            extra "train"
Makefile                                  target train
CLAUDE.md                                 อัปเดตบรรทัด ADR-008 → ADR-013
```

**ไม่แตะ:** `pipeline.py`, `geometry/`, `depth_sources/`, `evaluation/metrics_3d.py`, configs ของ Exp1–5

---

## 5. Data spec

### 5.1 Split (scene-disjoint ที่ระดับ `visit_id`)

| set | มาจาก fold | ฉาก | ต้องมี Faro | ใช้ทำอะไร | รายงานในเปเปอร์ | ดิสก์ (ประมาณ) |
|---|---|---|---|---|---|---|
| **T** test | Training | 6 เดิม: 42444474 47333774 47115299 42897743 47429736 45261556 (visit 421069 467370 470046 423306 471364 467324) | ✓ | ประเมินบน Mac เท่านั้น — **ไม่โหลดมาเครื่องนี้** | ✓ ต่อฉาก + mean ทุกตาราง | 0 |
| **V** val | **Validation** | 3 | ✓ | เลือก checkpoint / early stop เท่านั้น | ภาคผนวก (val curve) | 3 × 2.5 GB |
| **A** train-faro | Training | 10 | ✓ | R1, R2, R3 | ตาราง Setup (จำนวนฉาก/เฟรม) | 10 × 2.5 GB |
| **B** train-lidar | Training | 14 | ✗ | R3 เพิ่ม | ตาราง Setup | 14 × 1.2 GB |
| รวมบนเครื่องนี้ | | 27 | | | | **≈ 49 GB** (budget 50) |

สัดส่วนตามฉาก: R1/R2 = train 10 / val 3 / test 6; R3 = train 24 / val 3 / test 6
สัดส่วนตามเฟรม (ประมาณ): คู่ Faro ≈ 300–500/ฉาก → A ≈ 4k, V ≈ 1k (stride 5 → ~250), T ≈ 2.35k; คู่ LiDAR ≈ 900/ฉาก (stride → ~10 fps) → A ≈ 9k, A∪B ≈ 22k

- ต่อฉาก: `vga_wide` ≈ 1.0 GB, `lowres_depth`+`confidence`+traj+intrinsics ≈ 0.2 GB, `highres_depth` ≈ 1–2 GB
- คัดจาก `screen_scenes.py` (หนึ่ง video ต่อ visit) — เกณฑ์เดิม: scan 60–120 s, `hi_MB ≥ 100` สำหรับฉากที่ต้องมี Faro; A/B จาก fold Training, V จาก fold Validation
- `splits.yaml` **commit** พร้อม seed และห้ามแก้หลังเริ่ม R1 (ถ้าต้องแก้ = ADR/บันทึกใน log)
- ลบ zip หลังแตกทุกครั้ง; `--verify` นับเฟรม, fail ถ้าโฟลเดอร์ว่าง และ fail ถ้าพบ `visit_id` ของ T ในเครื่อง

### 5.2 ทำไม split แบบนี้ (บันทึกไว้เขียนลง ADR-013 และ Setup section ของเปเปอร์)

สิ่งที่ reviewer ตรวจไม่ใช่สัดส่วน แต่คือ 3 ข้อ: แยกตามฉากจริงไหม · test ถูกใช้ตัดสินใจอะไรระหว่างทางหรือเปล่า · test หลากหลายพอให้เชื่อไหม การออกแบบข้างบนตอบทีละข้อ:

1. **Test = 6 ฉากเดิมของ Exp1 ไม่เปลี่ยน** — เป็นฉากเดียวที่มี reference mesh (Faro fused 1 cm) และผล Exp1–5 ครบทุกแถว ถ้าเปลี่ยน test จะเทียบ `pretrained → fine-tuned` กับตารางเดิมไม่ได้เลย และ 6 ฉากนี้ *ถูกเลือกก่อนมีแนวคิดเทรน* (2026-09-15) ด้วยเกณฑ์ประเภทห้อง: ห้องนอนมาตรฐาน ×2, ห้องนั่งเล่นใหญ่, ห้องน้ำ (กระจก/กระเบื้อง), ห้องครัว ×2 รวม open-plan ใต้หลังคาลาด → เขียนได้ตรง ๆ ว่า *"stratified by room type, fixed before any training experiment"* ซึ่งเป็นจุดแข็ง
2. **Val ดึงจาก fold Validation ของ ARKitScenes เอง** (มี Faro ให้เลือก 107 visit) — ใช้ขอบเขตที่ dataset กำหนดมา ไม่มีใครสงสัยว่าเราเลือกเอง; ประโยคในเปเปอร์สั้น: *"train on the Training fold (visit-disjoint from test), select checkpoints on the Validation fold, report on 6 held-out scenes"* — 3 ฉากพอเพราะ val ไม่ถูกรายงาน ใช้แค่เลือก checkpoint (≈ 1,000 เฟรม Faro)
3. **แยกตาม `visit_id` ไม่ใช่ `video_id`** — ARKitScenes มีหลาย video ต่อบ้านหลังเดียว (visit) ถ้ากันแค่ video_id อาจได้ห้องเดียวกันอีกมุมไปเทรน = leakage ที่จับยากที่สุด; `--verify` เช็คเรื่องนี้ที่ระดับไฟล์บนดิสก์ด้วย ไม่ใช่แค่ใน yaml
4. **Train ซ้อนกันสองชั้น A ⊂ A∪B** — R1 (Faro) กับ R2 (LiDAR) ใช้ A ชุดเดียวกันเป๊ะ จึงเทียบคุณภาพครูได้แฟร์ (ตัวแปรเดียว = label); R3 ใช้ A∪B แสดงข้อได้เปรียบเชิงปฏิบัติของ LiDAR ("หาข้อมูลเพิ่มได้โดยไม่ต้อง laser scanner") ทั้งสามต้องอยู่ตารางเดียวกัน
5. **ไม่ทำ k-fold cross-validation** — แต่ละ fold = เทรน Large ใหม่ 1–2 ชม. และ 3D metrics ต้องมี reference mesh ต่อฉาก (มีแค่ 6) แทนด้วยการรายงาน **ต่อฉากทั้ง 6 + mean** (ทำอยู่แล้วใน `per_scene.md`) ซึ่งเป็นธรรมเนียมของงาน 3D reconstruction และแสดงความแปรปรวนข้ามห้องได้ตรงกว่า
6. **test 6/27 ≈ 22 % ใหญ่กว่าปกติ (70/15/15)** — ถูกกำหนดจากของที่มี ไม่ใช่ตัวเลือก และ test ไม่กินงบดิสก์เครื่องเทรน; งบ 50 GB จึงเทไปที่ train ให้มากที่สุด (val ลดเหลือ 3 ได้เพราะไม่ถูกรายงาน)

กฎที่ต้องเขียนใน Setup ของเปเปอร์: ทุก split เป็น scene-disjoint ที่ระดับ visit + `splits.yaml`/seed อยู่ใน repo · **ไม่มีตัวเลขจาก test ถูกใช้ตัดสินใจใด ๆ** (hyperparameter, checkpoint, epoch — ทุกอย่างตัดสินบน val) · limitation: 6 ฉาก, บ้านใน ARKitScenes (US) ล้วน, กล้อง iPad — generalization ข้ามอุปกรณ์ยังไม่ได้วัด

### 5.3 คู่ข้อมูลตอนเทรน (ต่อเฟรม)

| | Faro target | LiDAR target |
|---|---|---|
| RGB | `vga_wide` 640×480 ที่ timestamp ของ `highres_depth` (nearest ≤ 1/30 s — เหมือน `_AssetIndex.nearest`) | `vga_wide` ทุกเฟรม stride ให้ได้ ~10 fps (≈ 900/ฉาก) |
| depth (m) | `highres_depth` 1920×1440 uint16 mm → /1000 → **nearest** ลง 640×480 → ตาม transform เดียวกับ RGB | `lowres_depth` 256×192 uint16 mm → /1000 → **nearest** ขึ้น 640×480 |
| mask | depth > 0 และ depth ≤ `max_depth` (10 m) | เพิ่ม `confidence ≥ lidar_min_confidence` (default 1; ablation = 2) |
| upright | หมุนตาม `sky_direction` ทั้ง rgb/depth/mask (โมเดลต้องเห็นภาพตั้งตรง — เหมือน `MonocularDepth` ทำอยู่) | เหมือนกัน |
| intrinsics | ไม่ใช้ตอนเทรน (DA-v2 ไม่รับ) แต่ log ไว้ใน dataset stats | เหมือนกัน |

**กฎเหล็ก:** ห้าม interpolate depth (bilinear/area) ข้ามขอบวัตถุ → nearest เท่านั้น; ห้าม augmentation ที่เปลี่ยน scale ของภาพ (zoom, random-resized-crop) เพราะ target เป็นเมตร; hflip ได้, color jitter ได้

---

## 6. Training spec

| | ค่า | เหตุผล |
|---|---|---|
| โมเดล | DA-v2 Metric-Indoor **Large** (335 M) | ตัวที่ Exp1 ใช้ — เทียบก่อน/หลังตรง ๆ |
| freeze | `model.backbone` (DINOv2) ทั้งหมด, `requires_grad=False`, forward ใน `torch.no_grad()` | ประหยัด memory; ปัญหาไม่ได้อยู่ที่ encoder |
| เทรน | `model.neck` (DPT) + `model.head` (~30 M) | ส่วนที่แปลง feature → เมตร |
| input | 518×518 random crop จากภาพ resize สั้น = 518 (ตอน val: ทั้งเฟรม, ด้านหาร 14 ลงตัว) | patch 14 ของ DINOv2 |
| loss | **SiLog** (β = 0.15, ×10 — แบบ ZoeDepth/DA-v2 metric) บน mask; `+ λ·L1` option (default λ = 0) | ลงโทษ scale ผิด ตรงกับอาการ |
| optimizer | AdamW, lr 5e-5 (decoder), wd 0.01, warmup 200 steps, cosine → 0 | ค่ามาตรฐาน DA-v2 metric fine-tune (×10 ของ encoder lr) |
| precision | `torch.autocast(bfloat16)` | Ampere รองรับ; ลด VRAM ครึ่งหนึ่ง |
| batch | 2 × grad-accum 4 = effective 8; ถ้า OOM → batch 1 × 8 + gradient checkpointing ที่ neck | 8 GB |
| steps | R0: 30 · R1: ~6k (≈ 12 epoch ของ ~4k เฟรม Faro) · R2: ~6k (เท่ากับ R1 เพื่อความแฟร์) · R3: ~12k (~4 epoch ของ ~22k เฟรม LiDAR) | เวลาโดยประมาณ 0.6 s/step → R1/R2 ≈ 1 ชม., R3 ≈ 2 ชม. |
| val | ทุก 500 steps บน V (3 ฉาก fold Validation, เฟรมที่มี Faro, stride 5): `abs_rel`, `delta1`, `rmse` ด้วย `metrics_2d` เดิม; เก็บ `best/` ตาม `abs_rel` ต่ำสุด | ตัวเลขเดียวกับตารางเปเปอร์ |
| seed | 0 fix ทุก run; log ทุก step เป็น `log.jsonl` (`step, loss, lr, val/*`) | reproducible |
| baseline row | ก่อน step 0 รัน val ด้วย pretrained → บันทึกเป็น `step 0` ใน log | เห็นทันทีว่าดีขึ้นจากตรงไหน |
| output | `experiments/training/<run>/{config.yaml, log.jsonl, best/, last/}`; `best/` = `save_pretrained` (safetensors) + processor config | โหลดกลับด้วย `from_pretrained` เหมือนโมเดลปกติ |

---

## 7. Code style

ตามโค้ดเดิม: `from __future__ import annotations`, type hints, docstring บรรทัดแรกบอก "ทำอะไร + ข้อเท็จจริงที่ยืนยันแล้ว", comment ภาษาอังกฤษ, ruff line-length 120, ไม่มี global state, torch import ภายในฟังก์ชัน/คลาสเมื่อโมดูลอาจถูก import โดยไม่มี torch

```python
"""Depth-pair dataset for fine-tuning: ARKitScenesScene -> (rgb, depth_m, mask) at 518 px.
Depth is resampled with NEAREST only (no interpolation across object edges); metric
scale is never augmented. target="lidar" masks by confidence (ADR-013)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.dataio.arkitscenes import ARKitScenesScene


@dataclass(frozen=True)
class DepthPair:
    rgb: np.ndarray      # (H, W, 3) uint8, upright
    depth: np.ndarray    # (H, W) float32 metres, 0 = invalid
    mask: np.ndarray     # (H, W) bool


def make_pair(scene: ARKitScenesScene, idx: int, *, target: str, max_depth: float,
              lidar_min_confidence: int) -> DepthPair:
    frame = scene.frame(idx)
    if target == "faro":
        depth = frame.gt_depth
        mask = depth > 0
    elif target == "lidar":
        depth = frame.extra["lidar_depth"]
        mask = (depth > 0) & (frame.extra["lidar_confidence"] >= lidar_min_confidence)
    else:
        raise ValueError(f"unknown target {target!r}")
    mask &= depth <= max_depth
    return DepthPair(rgb=frame.rgb, depth=np.where(mask, depth, 0.0).astype(np.float32), mask=mask)
```

- ชื่อ run/ไฟล์: `r<N>_<what>` ตัวพิมพ์เล็ก; config key ตรงกับ `base.yaml` เดิม (dot-list override ได้)
- ทุกตัวเลขที่เป็นหน่วย ระบุใน comment (`# metres`, `# mm`)
- ไม่ hardcode path — รับจาก config/CLI; ใช้ `pathlib`

---

## 8. Testing strategy

`pytest` ใน `tests/test_training.py` — ใช้ `data/synthetic` (`make synthetic`) ซึ่งมี `lowres_depth`, `confidence`, `highres_depth` ใน layout ARKitScenes อยู่แล้ว; **ต้องผ่านโดยไม่มีข้อมูลจริงและไม่มี GPU**; ทั้งไฟล์ `pytest.importorskip("torch")` เพื่อให้ `make test` บน Mac แบบไม่มี torch ยังผ่าน

| ระดับ | เทสต์ | ตรวจอะไร |
|---|---|---|
| unit | `test_pair_faro_shapes_and_units` | rgb/depth/mask ขนาดเท่ากัน, depth เป็นเมตร (median ของห้องสังเคราะห์ ≈ 2–3 m ไม่ใช่ 2000) |
| unit | `test_pair_lidar_mask_uses_confidence` | conf < min → mask False, depth = 0 |
| unit | `test_transform_keeps_depth_nearest` | หลัง resize ไม่มีค่า depth ใหม่ที่ไม่อยู่ในภาพต้นทาง (set(unique) ⊆ ต้นทาง) |
| unit | `test_silog_zero_on_perfect_prediction` และ `test_silog_penalises_scale` | loss(pred=gt)=0; loss(2·gt) > loss(1.1·gt) |
| unit | `test_no_scale_augmentation` | random crop/flip ไม่เปลี่ยน depth values (เฉพาะจัดเรียงใหม่) |
| integration | `test_train_two_steps_tiny_model` | รัน `train.py` 2 steps บน CPU ด้วย `depth_anything_v2_metric_indoor_small` (ถ้าโหลด HF ไม่ได้ → skip) → มี `log.jsonl` 2 บรรทัด, `last/` โหลดกลับด้วย `from_pretrained` ได้, backbone weights ไม่เปลี่ยน (freeze จริง) |
| integration | `test_registry_ft_entries_build` | `build_depth_model("depth_anything_v2_ft_faro")` สร้างได้ `is_metric=True` (mock hf_id เป็น path local) |
| manual gate | `R0_smoke` | บน 3070: 30 steps ไม่ OOM, loss step 30 < step 1, `nvidia-smi` peak < 7.5 GB |

`make lint` (ruff) ต้องสะอาดก่อน commit ทุกครั้ง

---

## 9. Boundaries

**Always**
- แยก train/val/test ตาม *ฉาก* (`visit_id`) ตาม §5.1–5.2 — 6 ฉาก T ต้องไม่ถูกโหลดมาเครื่องนี้เลย; train จาก fold Training, val จาก fold Validation
- depth resample ด้วย nearest; ไม่มี scale augmentation
- run > 10 นาที ต้อง `nohup … &` และห้ามรัน 2 run เขียนโฟลเดอร์เดียวกัน
- commit `config.yaml` + `log.jsonl` + `splits.yaml`; ตรวจ `git status` ก่อน commit ว่าไม่มี weight/zip/png หลุด
- `make test` + `make lint` ผ่านก่อน push
- ตัวเลข val ทุกตัวมาจาก `evaluation/metrics_2d.py` เดิม ไม่เขียนสูตรซ้ำ
- บันทึกเวลาต่อ step, peak VRAM, จำนวนเฟรมจริงต่อ set ลง `log.jsonl` บรรทัดแรก (ไปเขียนใน Limitations/Setup ของเปเปอร์)

**Ask first**
- เปลี่ยน `splits.yaml` หลัง R1 เริ่ม
- unfreeze encoder / เพิ่ม LoRA / เปลี่ยน loss จาก SiLog
- เพิ่ม dependency นอกจาก `torch torchvision transformers huggingface_hub safetensors accelerate`
- โหลดเกิน budget 50 GB หรือโหลด `color` 1920×1440
- แตะ `pipeline.py`, `geometry/`, `depth_sources/` (ถ้าจำเป็น = เขียน ADR)
- ทำ HF repo เป็น public

**Never**
- commit weight (`*.safetensors`, `*.pt`, `checkpoints/`, `experiments/training/*/best|last`) หรืออะไรใต้ `data/`
- ใส่ HF token / path ส่วนตัว ในไฟล์ใน repo
- ใช้ฉาก T ในการเทรน/val/เลือก checkpoint — ไม่ว่าจะแค่ "ดูเฉย ๆ"
- รายงานตัวเลขจาก `data/synthetic` เป็นผล
- push ขึ้น `main` โดยตรง (ทำงานบน `training`, merge ตอนกลับมาที่ Mac)
- แก้ ADR เก่า — เขียน ADR-013 ใหม่แทน

---

## 10. Success criteria (ต้องเป็นจริงทุกข้อจึงถือว่า spec นี้เสร็จ)

1. `make test` ผ่านบน WSL (มี torch) และบน Mac ใน env ที่ไม่มี torch (training tests = skipped ไม่ใช่ failed)
2. `configs/training/splits.yaml` ถูก commit พร้อม seed; ไม่มี `visit_id` ซ้ำกับ T; A/B มาจาก fold Training, V จาก fold Validation; A/V/B ไม่ทับกัน (`--verify` ยืนยันบนดิสก์)
3. `download_training_scenes.py --verify` รายงานเฟรมครบทั้ง 27 ฉาก; `du -sh ~/data/arkitscenes` ≤ 50 GB
4. `R0_smoke` รันจบบน 3070 ไม่ OOM, peak VRAM < 7.5 GB, loss ลดจาก step 1 → 30
5. R1, R2, R3 รันจบ, แต่ละ run มี `log.jsonl` ที่มี `step 0` (pretrained baseline) และ `val/abs_rel` ที่ `best/` **ต่ำกว่า step 0** — ถ้าไม่ต่ำกว่า = bug ใน data ก่อนโทษโมเดล (กฎ Phase 1 gate)
6. เป้าเชิงตัวเลขบน V (ไม่ใช่เงื่อนไขผ่าน/ตก แต่เป็นสัญญาณ): `abs_rel` pretrained ≈ 0.30 (จาก failure analysis) → R1 < 0.10; R2 ภายใน 1.5× ของ R1
7. checkpoint ของ R1–R3 อยู่บน HF Hub private และบน Mac `build_depth_model("depth_anything_v2_ft_lidar").predict(rgb)` คืน (H, W) float32 เมตร ได้ (ทดสอบด้วยเฟรมเดียวจาก T)
8. `docs/architecture/adr-013-*.md` เขียนแล้ว, README §8 index + CLAUDE.md อัปเดต
9. branch `training` push แล้ว, `git status` สะอาด, ไม่มีไฟล์ใหญ่หลุด (`git count-objects -vH` ไม่โตเกิน 5 MB จาก main)

---

## 11. Open questions

| # | คำถาม | ค่า default ถ้าไม่ตอบ |
|---|---|---|
| Q1 | `lidar_min_confidence` default 1 หรือ 2? (Exp5 บอกว่าไม่ต่างตอน fuse แต่ตอนเป็น label อาจต่าง) | **1** ใน R2/R3; ทำ `R4_ft_lidar_conf2` เป็น optional ถ้ามีเวลา |
| Q2 | val stride 5 (≈ 80 เฟรม/ฉาก × 3 ฉาก ≈ 250 เฟรม) พอไหม หรือทุกเฟรม | stride 5 — val ควรใช้ < 2 นาที |
| Q3 | ชื่อ HF user / prefix repo | `<hf_user>/roomscan-dav2-metric-large-<run>` — ผู้ใช้ระบุตอน push |
| Q4 | หลัง R1–R3 มีเวลาทำ Base (97 M) เพื่อเป็นโมเดลที่รันบนมือถือได้จริงไหม | ไม่อยู่ใน spec นี้ — ถ้าทำ = run `R5_base_ft_lidar_all` config เดิมเปลี่ยน `model` |

---

## 12. ลำดับงานโดยสังเขป (รายละเอียดจะไปอยู่ใน `tasks/plan.md` + `tasks/todo.md` หลัง approve)

1. ADR-013 + `pyproject` extra + `Makefile` target + WSL setup ผ่าน (`cuda available`)
2. `training/dataset.py` + `transforms.py` + `loss.py` + tests บน synthetic → `make test` เขียว
3. `screen_scenes.py --fold Validation` + `select_training_scenes.py` → `splits.yaml` (commit) → `download_training_scenes.py` (รัน detach, ~2–4 ชม. ขึ้นกับเน็ต) → `--verify`
4. `train.py` + `validate.py` → `R0_smoke` ผ่าน gate ข้อ 4
5. R1 → R2 → R3 (detach ทีละ run) — ระหว่างรอ เขียน `push.py` + registry entries + `models/depth_anything.py` (`hf_id=`)
6. push checkpoint, commit log/config, push branch `training` → กลับ Mac ทำ spec ฝั่งประเมิน
