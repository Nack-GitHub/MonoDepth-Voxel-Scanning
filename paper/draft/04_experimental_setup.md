# 4. Experimental Setup

> ร่าง 2026-09-16 — โปรโตคอลทั้งหมดอ่านจากโค้ด (`metrics_3d.py`, `metrics_2d.py`, `scale_align.py`, `dataio/arkitscenes.py`) และ `configs/base.yaml`
> ตาราง 4.1 จาก `scripts/scene_info.py --md` (2026-09-16)

## 4.1 Dataset: ARKitScenes

เราใช้ ARKitScenes (ADR-009) เพราะเป็น dataset เดียวที่ให้ **ทั้งสาม** สิ่งที่คำถามวิจัยต้องการในห้องเดียวกัน:
วิดีโอ RGB จากอุปกรณ์จริง (iPad Pro), depth จาก LiDAR ของเครื่องนั้น (`lowres_depth` 256×192), และ depth จาก Faro laser scanner
ที่ register เข้ากับ pose ของกล้อง (`highres_depth` 1920×1440) — แถว "iPhone Pro วันนี้" และแถว "reference" จึงมาจาก scan เดียวกัน ไม่ต้อง align ข้าม sensor เอง

สิ่งที่ dataset ให้และเราใช้:

| asset | ความละเอียด / อัตรา | บทบาทในงานนี้ |
|---|---|---|
| `vga_wide` RGB | 640×480, 30 fps | input ของโมเดล monocular (หมุนให้ตั้งตรงตาม `sky_direction`) |
| `lowres_depth` + `confidence` | 256×192, 60 fps | depth source `lidar`; แหล่ง proxy ของ `sparse_points`; weight ใน Exp5 |
| `highres_depth` (Faro) | 1920×1440, **~2.7–4 fps** | depth source `gt` และ **reference mesh** |
| `lowres_wide.traj` | ~10 Hz | pose (VIO ของ ARKit) — interpolate slerp/lerp มาที่ timestamp ของแต่ละเฟรม |
| `lowres_wide_intrinsics` | ต่อเฟรม | ใช้ค่า median ทั้ง scan แล้ว scale ตาม resolution |

ข้อจำกัดสองข้อของ dataset กำหนดรูปแบบการทดลอง: (1) Faro depth มีเฉพาะเฟรมที่ Apple กรองไว้ (~3 fps) เราจึงประเมินทุก source
**บนเฟรมชุดเดียวกันนี้** และ Exp3 (frame stride) กลายเป็นการวัด coverage มากกว่า compute (§5.3); (2) ไม่มี laser mesh ให้ตรง ๆ เราสร้างเอง (§4.2)

### ตาราง 4.1 ฉากที่ใช้ (6 scans)

เลือกด้วย `scripts/screen_scenes.py` จาก metadata โดยไม่โหลด: ต้องมี `highres_depth`, อยู่ใน 3DOD split, และครอบคลุมประเภทห้องที่ต่างกัน
(มาตรฐาน / ใหญ่ / ผิวมัน-กระจก) ซึ่งเป็น failure case ที่คาดไว้ของ monocular depth

| video_id | ประเภทห้อง (จาก 3DOD labels) | ขอบเขต Faro (m) | scan (s) | เฟรม Faro | บทบาท |
|---|---|---|---|---|---|
| 42444474 | ครัว-กินข้าว open-plan ใต้หลังคาลาด, โต๊ะ/ราวกระจก, ผนังอิฐ | 9.9 × 5.1 × 4.2 | 88 | 236 | ใหญ่ + กระจก (ฉากพัฒนา) |
| 47333774 | ห้องนอนมาตรฐาน (bed, shelf, cabinet, table) | 5.1 × 4.9 × 3.3 | 90 | 375 | มาตรฐาน #1 |
| 47115299 | ห้องนั่งเล่นใหญ่ (table×4, sofa×2, tv) | 8.1 × 6.1 × 2.5 | 91 | 324 | ใหญ่ |
| 42897743 | ห้องน้ำ (bathtub, sink, toilet) | 8.0 × 3.4 × 3.6 | 114 | 448 | กระจก/กระเบื้อง |
| 47429736 | ห้องนอน + ตู้ (bed, cabinet×5, table×2) | 6.6 × 6.4 × 2.7 | 98 | 504 | มาตรฐาน #2 |
| 45261556 | ห้องครัว (cabinet×10, stove, sink) | 7.1 × 6.1 × 2.4 | 98 | 465 | ผิวมัน #2 |

ขอบเขต = axis-aligned extent ของ reference mesh (พื้นที่ที่ Faro เห็น) ไม่ใช่ขนาดห้องจากแบบแปลน; เฟรม Faro = จำนวน `highres_depth` = จำนวนเฟรมที่ทุกแถวถูกประเมิน
รวม 2,352 เฟรม / 577 s ของวิดีโอ (`scripts/scene_info.py --md`)

ฉาก 42444474 ถูกใช้ระหว่างพัฒนา (Phase 0–4, ADR-010) ตัวเลขของมันจึงไม่ใช่ "held-out" — เราแยกรายงานเป็นต่อฉากใน §5 เพื่อให้เห็นว่า
ข้อสรุปยังคงอยู่บนอีก 5 ฉากที่ pipeline ไม่เคยเห็น

## 4.2 Reference mesh

ARKitScenes ให้ Faro เป็น depth map ต่อเฟรม ไม่ใช่ mesh เราจึง fuse `highres_depth` **ทุกเฟรม** ด้วย TSDF ที่ voxel **1 cm** (`sdf_trunc` 3 cm) หนึ่งครั้งต่อฉาก
→ `reference_mesh.ply` (`scripts/build_reference_mesh.py`) แล้วไม่แตะอีกตลอดการศึกษา

ผลที่ตามมาซึ่งต้องอ่านตารางให้ถูก:
- แถว `gt` ของทุกการทดลองรันที่ voxel ของการทดลอง (4 cm) จึงมี Chamfer ไม่เป็นศูนย์ — ค่านั้นคือ **discretisation loss ของ pipeline เอง** และเป็นขอบล่างของทุกแถว
- reference ครอบคลุมเฉพาะที่ Faro เห็น ทุก source จึง integrate เฉพาะ pixel ที่ Faro มีค่า (`eval.mask_to_gt=true`) — ไม่มีแถวใดถูกลงโทษเพราะ "เห็นมากกว่า reference"
- reference ใช้ pose เดียวกับทุกแถว (VIO ของ ARKit) ⇒ pose drift **ตัดออกจากการเปรียบเทียบ** โดยการออกแบบ ตัวเลขทั้งหมดคือผลของ depth source ล้วน ๆ (ข้อจำกัด: §8)

## 4.3 Pipeline configuration (คงที่ทุกแถว)

| ส่วน | ค่า | เหตุผล |
|---|---|---|
| fusion resolution | 256×192 | resolution ของ LiDAR — ทุก source ถูก resample ลงมาเท่ากันก่อน integrate |
| voxel / sdf_trunc | 4 cm / 12 cm | Exp2 แสดงว่า 4 cm คือจุดคุ้มบน Apple Silicon (§5.2) |
| depth range | 0.1–5.0 m | ตัด sensor noise ใกล้ และ Faro noise ไกล |
| post-process | ลบ cluster < 1000 สามเหลี่ยม | ไม่ decimate |
| frame stride | 1 (ทุกเฟรม Faro) | ยกเว้น Exp3 |
| device | Apple M-series (MPS) | เวลาที่รายงานคือเวลาบนเครื่องนี้ |

โมเดล monocular ที่ใช้ (pretrained ทั้งหมด, ADR-008): Depth Anything V2 Large (335 M params) / Small (25 M) และ DA-v2 Metric-Indoor (Large, fine-tune Hypersim);
MiDaS v2.1 small (21 M) สำหรับ Exp4 ทุกโมเดลรับภาพที่หมุนตั้งตรงแล้ว output ถูกแปลงเป็น depth-like (`1/disparity`) ก่อนเข้า aligner

## 4.4 Scale alignment protocol

aligner ทุกตัวที่ไม่ใช่ `identity` fit affine **ในสเปซ inverse-depth** (ADR-010): หา `(s, t)` ที่ `1/gt ≈ s·(1/pred) + t`
ด้วย least squares แบบ trimmed — 3 รอบ ตัด residual แย่สุด 20 % ทุกรอบ — บน pixel ที่ทั้ง GT และ pred valid ในช่วง 0.1–5 m แล้ว invert กลับเป็นเมตร

| aligner | fit บน | ใช้ตอน |
|---|---|---|
| `oracle_frame` | GT depth ของเฟรมนั้น, ทุกเฟรม | เพดานของโมเดล (upper bound) |
| `per_scene` | GT depth ของ **10 เฟรมกระจายทั่ว scan** (`frames[::k][:10]`) stack รวมกัน fit ครั้งเดียว แล้วใช้ `(S, T)` เดียวทุกเฟรม | สิ่งที่ระบบจริงทำได้ด้วย calibration ครั้งเดียว |
| `sparse_points` | 200 pixel/เฟรม sample จาก LiDAR ที่ confidence 2 (seed = เฟรม; ADR-011) fit ต่อเฟรม trim 10 % 2 รอบ; เฟรมที่จุดใช้ได้ < 20 ใช้ `(s, t)` ของเฟรมก่อน | ขอบบนของระบบที่ใช้จุด VIO จริง |

`per_scene` ใช้ GT ในการ fit ด้วย — ดังนั้นมันคือ **ขอบบนของ calibration ครั้งเดียว** ระบบจริงที่ calibrate จากไม้บรรทัดหรือจุด VIO จะได้ไม่ดีกว่านี้

## 4.5 โปรโตคอลการ fine-tune (Exp6, ADR-013)

**ฉาก** — แยกจาก 6 ฉากเทสต์ที่ระดับ `visit_id` สุ่มด้วย seed 0 จากรายการฉากที่คัดไว้: 10 ฉากเทรนที่มี **ทั้ง Faro และ LiDAR**
(ครูสองแบบจึงเห็นเฟรมชุดเดียวกัน), อีก 14 ฉากเทรนที่มีแต่ LiDAR (ไม่มี laser scan เลย) และ 3 ฉากจาก fold Validation ที่มี Faro ไว้เลือก checkpoint
รวม: เทสต์ 6 / val 3 / เทรน 24 ฉาก — ไม่มีห้องเดียวกัน (หรือการสแกนอื่นของห้องเดียวกัน) โผล่ในสองชุด
ไฟล์แบ่งฉาก (`configs/training/splits.yaml`) และจำนวนเฟรมต่อฉากเผยแพร่พร้อมโค้ด

**label** — R1 ใช้ Faro `highres_depth` (3,594 เฟรม, 10 ฉาก); R2 ใช้ `lowres_depth` ของ iPad ที่ ARKit confidence ≥ 1 บน 10 ฉากเดียวกัน
(8,656 เฟรม — LiDAR มีในเฟรมที่ laser scan ไม่ครอบคลุมด้วย); R3 ใช้ label LiDAR เดียวกันบนทั้ง 24 ฉาก (19,896 เฟรม)
depth ถูก resample ลงกริด RGB ด้วย nearest เท่านั้น; pixel ที่ไกลเกิน 5 m หรือไม่มี label ถูก mask ทิ้ง

**การเทรน** — freeze DINOv2 encoder (300M พารามิเตอร์, โหมด `eval`) เทรนเฉพาะ DPT neck + metric head ด้วย SiLog loss (β = 0.15),
AdamW (lr 5e-5, weight decay 0.01, warmup 200 step แล้ว cosine), bf16 autocast, batch 2 + gradient accumulation 4
บน crop ขนาด 518×518 จากภาพที่ย่อให้ด้านสั้น = 518; R1/R2 เทรน 6,000 step, R3 12,000 step
validate ทุก 500 step และเก็บ checkpoint ที่ AbsRel ต่ำสุดบน 3 ฉาก val — หนึ่ง run ใช้เวลา ~1 ชม. บน RTX 3070 (VRAM 3.2 GB)

**ตอนเทสต์** — โมเดลที่ fine-tune แล้วเป็นแถว `DepthSource` ธรรมดา: RGB เข้า เมตรออก `aligner: identity` ไม่เห็น depth/pose/intrinsics

## 4.6 Metrics

**3D (ตัวเลขหลักของเปเปอร์)** — โปรโตคอลใน `metrics_3d.py` ใช้ตรงตามนี้ทุกแถว:
สุ่ม 200,000 จุดสม่ำเสมอบน mesh ที่สร้างและบน reference (seed 0);
accuracy = ระยะเฉลี่ย pred→ref (สร้างสิ่งที่ไม่มีจริง?), completeness = ref→pred (พลาดสิ่งที่มี?), **Chamfer = ค่าเฉลี่ยของสอง**;
precision/recall/F-score ที่ τ = 2 / 5 / 10 cm — **F@5cm คือตัวเลขหลัก** (5 cm ≈ ความคลาดเคลื่อนที่ยอมรับได้ในการวางเฟอร์นิเจอร์);
normal consistency = mean |n_pred · n_ref| ของคู่จุดใกล้สุด

**2D (วินิจฉัย)** — ต่อเฟรม เทียบ depth *หลัง align* กับ Faro depth บน pixel ที่ทั้งคู่อยู่ใน 0.1–5 m: RMSE, AbsRel, δ₁ (สัดส่วน pixel ที่ max(p/g, g/p) < 1.25)
เฉลี่ยทุกเฟรม ใช้แยกว่า error เกิดที่ depth (ก่อน fusion) หรือที่ fusion

**เวลา** — `time_total_s` = โหลดข้อมูล + inference + align + fusion + extract บนเครื่องเดียว; รายงานแยก inference ใน §7

## 4.7 การทดลอง

| Exp | ตัวแปร | ค่า | ส่วนที่คงที่ |
|---|---|---|---|
| 1 depth source | `depth.source` × `depth.aligner` | gt / lidar / mono+oracle / mono+sparse / mono+per_scene / mono-metric+identity | DA-v2 L, voxel 4 cm, stride 1 |
| 2 voxel size | `fusion.voxel_size` | 2 / 4 / 8 cm (sdf_trunc = 3×) | gt |
| 3 frame stride | `dataset.frame_stride` | 1 / 5 / 10 / 20 × {gt, mono+oracle, mono+per_scene} | DA-v2 L, voxel 4 cm |
| 4 model size | `depth.model` | DA-v2 L / DA-v2 S / MiDaS small | oracle_frame, voxel 4 cm |
| 5 LiDAR confidence | `depth.lidar_min_confidence`, `fusion.confidence_weights` | ทุก pixel / mask ≥1 / mask =2 / weight [0,1,2] / [1,2,4] | lidar, voxel 4 cm |
| 6 ครูตอน fine-tune | `depth.model` | pretrained / ft-Faro / ft-LiDAR / ft-LiDAR 24 ฉาก | identity, voxel 4 cm |

ทุก run = 1 ไฟล์ `config.yaml` ที่ resolve แล้ว + `metrics.json` ใน `experiments/results/<exp>/<scene>_<run>/` (commit ไว้ทั้งหมด);
ตารางใน §5 สร้างจาก `roomscan report` โดยไม่แก้มือ ตัวเลขเป็น mean ± std ข้าม 6 ฉาก และมีตารางต่อฉากใน `per_scene.md`

การทดลองทั้งหมดทำซ้ำได้ด้วย `make reference && make sweep-exp1 sweep-exp2 sweep-exp3 sweep-exp4 sweep-exp5 sweep-exp6 report` หลังโหลดข้อมูลตาม `data/README.md`
