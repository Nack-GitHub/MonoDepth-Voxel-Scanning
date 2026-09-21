# ADR-013: Fine-tune DA-v2 metric บน ARKitScenes ด้วยครู LiDAR — มี training loop ใน repo

## Status
Accepted (2026-09-21) — **supersede ADR-008** (pretrained-only) · spec: `SPEC.md`, แผน: `tasks/plan.md`

## Context
Exp1 ชี้ว่า DA-v2 Metric-Indoor พังบนห้องจริง (chamfer 49 cm) เพราะ scale ต่อเฟรมแกว่ง ±40 % ทั้งที่รูปทรงถูก
(oracle 6.5 cm) — สาเหตุคือ domain gap จากการเทรนบน Hypersim สังเคราะห์ (`paper/analysis/42444474_failure_analysis.md` §2)
ADR-008 บอกว่าให้ *รายงานตามนั้น ไม่ fine-tune* ซึ่งทำไปแล้ว (Exp1–5) คำถามถัดไปของเปเปอร์คือ
*"ใช้ LiDAR ของ iPhone/iPad เป็นครู สอนโมเดล mono ให้ตอบเป็นเมตรบนเครื่องที่ไม่มี LiDAR ได้ดีแค่ไหน และเสียจากครูสมบูรณ์ (Faro) เท่าไหร่"*
ตอบไม่ได้โดยไม่มี training loop

## Options

| Option | Pros | Cons |
|---|---|---|
| A. คง ADR-008, ทำ fine-tune ใน repo แยก | repo นี้ไม่มีโค้ดเทรน | ต้องคัดลอก loader/metrics → นิยามตัวเลขเพี้ยนได้; split ตรวจข้าม repo ไม่ได้ |
| B. **`src/roomscan/training/` ใน package เดิม** ใช้ `ARKitScenesScene` + `metrics_2d` เดิม | ข้อมูล/ตัวเลขนิยามเดียวกับตารางเปเปอร์; checkpoint กลับเข้า registry `models/` ตรง ๆ | package มี torch-only module เพิ่ม (ต้อง lazy import + skip test) |
| C. Unfreeze ทั้งโมเดล / LoRA | อาจได้ผลดีกว่า | Large ไม่พอ 8 GB; ปัญหาอยู่ที่ scale (decoder) ไม่ใช่ feature |

## Decision
**B.** เพิ่ม `src/roomscan/training/` (pairs, dataset, transforms, loss, train, validate, push)
- **Dependency rule:** `training/` import ได้เฉพาะ `dataio`, `evaluation.metrics_2d`, `types`, `config` — ห้าม `pipeline`, `geometry`, `depth_sources`, `open3d`
  ผลลัพธ์กลับเข้าระบบผ่าน registry `models/` เท่านั้น (`DepthAnythingV2Metric(hf_id=...)`) — `pipeline.py` ไม่เปลี่ยน
- **โมเดล:** DA-v2 Metric-Indoor Large, freeze `backbone` (DINOv2) ทั้งหมด, เทรน `neck` (DPT) + `head`; loss SiLog (β 0.15, ×10)
- **Label:** Faro `highres_depth` (ครูสมบูรณ์) หรือ ARKit `lowres_depth` ที่ `confidence ≥ 1` (ครู LiDAR)
- **กฎข้อมูล:** depth resample ด้วย nearest เท่านั้น; ห้าม augmentation ที่เปลี่ยน scale ของภาพ (target เป็นเมตร); hflip / color jitter ได้
- **Weight ไม่ commit:** `experiments/training/<run>/{config.yaml, log.jsonl}` commit; `best/`, `last/` gitignore; ส่งผ่าน HuggingFace Hub private
- extra `train` ใน `pyproject.toml`; `make train RUN=<yaml>`

### Split (scene-disjoint ระดับ `visit_id`) — `configs/training/splits.yaml` commit พร้อม seed

| set | fold ARKitScenes | ฉาก | Faro | ใช้ทำอะไร |
|---|---|---|---|---|
| T test | Training | 6 ฉากเดิมของ Exp1 (visit 421069 467370 470046 423306 471364 467324) | ✓ | ประเมินบน Mac เท่านั้น — ไม่โหลดมาเครื่องเทรน |
| V val | **Validation** | 3 | ✓ | เลือก checkpoint / early stop เท่านั้น |
| A train-faro | Training | 10 | ✓ | R1 (Faro), R2 (LiDAR), R3 |
| B train-lidar | Training | 14 | ✗ | R3 เพิ่ม (A ∪ B) |

เหตุผล:
1. **Test = 6 ฉากเดิมไม่เปลี่ยน** — เป็นฉากเดียวที่มี reference mesh (Faro fused 1 cm) และผล Exp1–5 ครบ; ถูกเลือกก่อนมีแนวคิดเทรน (2026-09-15)
   ด้วยเกณฑ์ประเภทห้อง ⇒ *"stratified by room type, fixed before any training experiment"*
2. **Val มาจาก fold Validation ของ ARKitScenes** — ใช้ขอบเขตที่ dataset กำหนด ไม่ใช่เราเลือกเอง; 3 ฉากพอเพราะ val ไม่ถูกรายงาน
3. **แยกตาม `visit_id` ไม่ใช่ `video_id`** — หลาย video ต่อบ้านเดียว ถ้ากันแค่ video อาจได้ห้องเดียวกันอีกมุม = leakage;
   `download_training_scenes.py --verify` ตรวจบนดิสก์ ไม่ใช่แค่ใน yaml
4. **A ⊂ A∪B** — R1/R2 ใช้ A เดียวกันเป๊ะ (ตัวแปรเดียว = label); R3 แสดงข้อได้เปรียบเชิงปฏิบัติของ LiDAR (หาข้อมูลเพิ่มได้โดยไม่ต้องมี laser scanner)
5. **ไม่ทำ k-fold** — แต่ละ fold = เทรน Large ใหม่ และ 3D metrics ต้องมี reference mesh ต่อฉาก (มีแค่ 6) → รายงานต่อฉาก 6 + mean แทน
6. **Test 6/27 ≈ 22 %** — ถูกกำหนดจากของที่มี; งบดิสก์ 50 GB เทไปที่ train

**ไม่มีตัวเลขจาก test ถูกใช้ตัดสินใจใด ๆ** (hyperparameter, checkpoint, epoch — ทุกอย่างตัดสินบน V)

## Consequences
- ✅ คำถามครู LiDAR vs Faro ตอบได้ด้วยตัวเลขนิยามเดียวกับ Exp1 (`metrics_2d`, registry เดิม, pipeline เดิม)
- ✅ `make test` บนเครื่องที่ไม่มี torch ยังผ่าน — `tests/test_training.py` ใช้ `pytest.importorskip("torch")`
- ⚠️ เปลี่ยน `splits.yaml` หลังเริ่ม R1 = ต้องบันทึกเป็น ADR ใหม่
- ⚠️ Limitation ที่ต้องเขียน: test 6 ฉาก, บ้านใน ARKitScenes (US) ล้วน, กล้อง iPad — ข้ามอุปกรณ์ยังไม่ได้วัด
- 🔜 Base (97 M) สำหรับมือถือ = config เดิมเปลี่ยน `model` (ไม่อยู่ใน ADR นี้)
