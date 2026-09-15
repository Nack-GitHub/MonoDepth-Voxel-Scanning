# 3. System Design

> ร่างแรก 2026-09-15 — สรุปจาก `docs/architecture/README.md` §2–§5 และ ADR-001/002/004/005/009/010
> รูป 3.1 ใช้ mermaid ใน architecture README แปลงเป็น figure; ตารางในบทนี้ไม่ขึ้นกับผลการทดลอง

## 3.1 หลักการออกแบบ: หนึ่ง pipeline หนึ่งตัวแปร

คำถามวิจัย ("เปลี่ยนแหล่ง depth อย่างเดียว ผลต่างเท่าไร") บังคับข้อกำหนดเชิงสถาปัตยกรรม 3 ข้อ:

1. **ทุกแหล่ง depth ต้องผ่านโค้ดเดียวกันหลังจุดที่มันถูกสร้าง** — ไม่มี branch พิเศษสำหรับ mono หรือ LiDAR ใน fusion / evaluation
2. **สิ่งที่สลับได้ต้องสลับจาก config** ไม่ใช่จากการแก้โค้ด — ตารางผลการทดลอง 1 ตาราง = ไฟล์ YAML 1 ไฟล์ (ADR-005)
3. **โค้ดเปเปอร์คือโค้ดผลิตภัณฑ์** — ระบบเดียวกันต้องเสียบ ARCore depth หรือโมเดลที่เล็กกว่าได้ในวันที่ต้องส่ง MVP โดยไม่เขียนใหม่ (ADR-007)

ข้อ 1 นำไปสู่ interface `DepthSource`; ข้อ 2 นำไปสู่ registry แบบ string → class และ config แบบ base + override; ข้อ 3 ทำให้เราเลือก
monolith package เดียว ไม่มี service/DB/experiment tracker (ADR-006) ซึ่งเหมาะกับทีม 1 คน ~10 สัปดาห์

## 3.2 ภาพรวม

```
video frames ─▶ SceneDataset ─▶ DepthSource ─▶ ScaleAligner ─▶ TSDFFusion ─▶ postprocess ─▶ mesh
                                 ▲ ตัวแปร (Exp1, 4)  ▲ ตัวแปร (Exp1)    ▲ voxel (Exp2)              │
                                                                       stride (Exp3) มาจาก dataset  ▼
                                                                                   metrics_3d vs reference mesh
                                                     metrics_2d ต่อเฟรม ◀──── depth หลัง align vs GT depth
```

**รูป 3.1** ส่วนประกอบ: สองกล่องซ้ายของ "ตัวแปร" คือสิ่งเดียวที่เปลี่ยนระหว่างแถวของตาราง; ทุกอย่างทางขวาคงที่ (Open3D `ScalableTSDFVolume`, marching cubes, cluster filtering)

Data flow ต่อเฟรมมี type ชัดเจน 3 จุด:

| จุด | type | ใครรับผิดชอบ |
|---|---|---|
| `Frame{rgb, pose_c2w, gt_depth?, intrinsics, extra}` | จาก `SceneDataset` | dataset mapping (ADR-009) |
| `pred: (H,W) float32` — relative *หรือ* metric | `DepthSource.get_depth(frame)` | source บอกเองว่า `is_metric` |
| `depth_m: (H,W) float32 เมตร` (0 = invalid) | `ScaleAligner.align(pred, frame)` | invariant: หลัง aligner เป็นเมตรเสมอ |

Invariant ที่บังคับในโค้ด (ไม่ใช่แค่เอกสาร): (i) หลัง aligner ค่าเป็นเมตร (ii) `pose_c2w` เป็น camera→world และถูก invert ที่ `TSDFFusion.integrate` ที่เดียว
(iii) intrinsics ผูกกับ resolution — resize แล้วต้อง `Intrinsics.scaled()` (iv) `is_metric=True ⇒ aligner=identity`, `is_metric=False ⇒ aligner≠identity` — pipeline fail ตอน construct ถ้าคู่ผิด

## 3.3 DepthSource — แหล่ง depth เป็น strategy (ADR-001)

```python
class DepthSource(ABC):
    name: str
    is_metric: bool
    def get_depth(self, frame: Frame) -> np.ndarray: ...   # (H,W) float32, 0/NaN = invalid
```

รับ `Frame` ทั้งก้อนแทน `(rgb)` เพราะแต่ละ source ต้องการ input ต่างกัน: `gt` อ่าน `frame.gt_depth`, `lidar` อ่าน `frame.extra["lidar_depth"]` (+ confidence),
`mono` อ่าน `frame.rgb` — signature เดียวรองรับทุกกรณีโดยไม่ต้องแก้เมื่อเพิ่ม `arcore`
`MonocularDepth` ทำ 3 อย่างที่ *ไม่ใช่* หน้าที่ของโมเดลหรือ aligner: หมุนภาพให้ตั้งตรงตาม `sky_direction`, แปลง disparity → 1/x ให้ "มากกว่า = ไกลกว่า", และ mask ค่าไม่ถูกต้อง
โมเดล (`DepthModel`: DA-v2, MiDaS) เป็นอีก registry หนึ่งภายใต้ `mono` ทำให้ Exp4 (ขนาดโมเดล) เป็นแค่ `depth.model=...`

## 3.4 ScaleAligner — สเกลเป็นขั้นตอนแยก (ADR-002, ADR-010)

โมเดล relative ให้ depth ถูกต้องขึ้นกับ affine transform การเลือกว่าจะหา `(s, t)` อย่างไรคือ **การตัดสินใจเชิงระบบ** ที่ควรวัดแยกจากโมเดล จึงแยกเป็น stage:

| aligner | fit `(s, t)` | ความหมาย |
|---|---|---|
| `identity` | — | สำหรับ GT / LiDAR / โมเดล metric |
| `oracle_frame` | ต่อเฟรม กับ GT depth | เพดานของโมเดล (deploy ไม่ได้) |
| `per_scene` | ครั้งเดียว บน N เฟรมกระจายทั่ว scan แล้วแช่แข็ง | calibration ครั้งเดียว — ระบบจริงทำได้ |
| `sparse_points` | ต่อเฟรม กับจุด sparse จาก VIO | เส้นทางสู่ MVP (ยังไม่ประเมินในงานนี้) |

ช่องว่าง `oracle_frame` → `per_scene` = "ราคาของการไม่รู้ scale" ซึ่งเป็นผลการทดลอง ไม่ใช่ noise

**สเปซที่ fit (ADR-010):** โมเดล affine-invariant เป็น affine ใน *disparity* ไม่ใช่ depth การ fit `gt ≈ s·pred + t` ในสเปซ depth เป็นโมเดลผิด
เราจึง fit `1/gt ≈ s·(1/pred) + t` (least squares แบบ trim 20 % ของ residual 3 รอบ) แล้ว invert กลับ
บนฉากจริง การเปลี่ยนสเปซนี้อย่างเดียวย้าย mono+oracle จาก 15.2 → 5.4 cm — ตัวอย่างว่าทำไม aligner ต้องเป็นสิ่งที่ทดสอบแยกได้

## 3.5 ส่วนคงที่: fusion, post-process, evaluation

- **Fusion:** Open3D `ScalableTSDFVolume` (ADR-004), voxel 4 cm, `sdf_trunc = 3 × voxel`, depth 0.1–5 m; ทุก source ถูก resample ลง 256×192 (ความละเอียดของ LiDAR) ก่อน integrate เพื่อความแฟร์
  และ integrate เฉพาะ pixel ที่ reference มี depth (`mask_to_gt`) — ไม่ลงโทษ source ที่มองเห็นเกินขอบเขต Faro
- **Post-process:** ลบ cluster เล็ก, ไม่ decimate
- **Reference:** ARKitScenes ไม่มี laser mesh ให้ตรง ๆ เราสร้าง `reference_mesh.ply` โดย fuse Faro `highres_depth` ทุกเฟรมที่ voxel 1 cm ครั้งเดียวต่อฉาก (ADR-009) — แถว `gt` ที่ voxel 4 cm จึงวัด "pipeline loss" ไม่ใช่ศูนย์
- **Metrics 3D** (โปรโตคอลตายตัวใน `metrics_3d.py`): สุ่ม 200k จุดบนทั้งสอง mesh (seed คงที่); accuracy = mean dist pred→ref, completeness = ref→pred, Chamfer = ค่าเฉลี่ยของสอง;
  precision/recall/F-score ที่ 2 / 5 / 10 cm (**F@5cm คือตัวเลขหลัก**); normal consistency = mean |n_pred·n_ref|
- **Metrics 2D** ต่อเฟรม (RMSE, AbsRel, δ₁₋₃) วัด depth *หลัง align* กับ GT depth — ใช้แยกว่า error เกิดก่อนหรือหลัง fusion

## 3.6 Config, registry, และการทดลอง

`configs/base.yaml` กำหนดทุกค่า; `configs/experiments/expN.yaml` = `base` + รายการ `runs` ที่แต่ละ run เป็น dotlist override ไม่กี่ key
`roomscan sweep` รันทุก run × ทุกฉาก, เขียน `config.yaml` (ที่ resolve แล้ว) + `metrics.json` ต่อ run; `roomscan report` รวมเป็น `summary.csv` + `table.md` ต่อการทดลอง
ตารางในบทที่ 5 คัดลอกจาก `table.md` เหล่านี้โดยตรง

การเพิ่มความสามารถทุกชนิด = ไฟล์ใหม่ 1 ไฟล์ + registry 1 บรรทัด:

| อยากเพิ่ม | ไฟล์ | แก้ `pipeline.py`? |
|---|---|---|
| depth source (ARCore) | `depth_sources/arcore.py` | ✗ |
| โมเดล (รุ่นบีบอัด) | `models/<x>.py` | ✗ |
| aligner (sparse VIO) | class ใน `geometry/scale_align.py` | ✗ |
| dataset (ห้องจริงจากมือถือ) | `dataio/<x>.py` | ✗ |
| การทดลอง | `configs/experiments/<x>.yaml` | ✗ |

กฎที่ใช้ตรวจว่าการออกแบบยังไม่รั่ว: ถ้าการเพิ่มอะไรสักอย่างต้องแก้ `pipeline.py` ให้หยุดและเขียน ADR — ตลอด Phase 0–4 ยังไม่เกิดขึ้น
(การเปลี่ยนที่ใหญ่สุดคือ ADR-010 ซึ่งอยู่ใน aligner ทั้งหมด)

## 3.7 สิ่งที่ตั้งใจไม่ทำ

GPU TSDF, experiment tracker, pose estimation (ใช้ VIO ของ ARKit ที่มากับ dataset), web API/viewer — ทั้งหมดเป็นงานหลังเปเปอร์
และทุกอย่างเสียบเข้าที่ interface เดิมได้ (`TSDFFusion`, `SceneDataset`) โดยไม่กระทบผลการทดลอง
