# 1. Introduction and Business Motivation

> ร่างแรก 2026-09-15, อัปเดตตัวเลขเป็น 6 ฉาก 2026-09-17 (mean ± std จาก `experiments/results/exp1_depth_source/table.md`)
> ภาษา: ร่างเป็นไทยตาม outline; ถ้าส่งเวทีอังกฤษให้แปลจากร่างนี้หลังตัวเลขนิ่ง

## 1.1 ปัญหา

การสร้างแบบจำลอง 3 มิติของห้องจากสมาร์ตโฟน (room scanning) เป็นฟีเจอร์ที่ผู้ใช้ทั่วไปเข้าถึงได้แล้วบน iPhone Pro / iPad Pro
ผ่าน Apple RoomPlan และแอปอย่าง Polycam ซึ่งพึ่งพา **เซ็นเซอร์ LiDAR** ที่ให้ depth เชิงเมตรโดยตรง
แต่เครื่องที่มี LiDAR เป็นส่วนน้อยของตลาด — iPhone รุ่นไม่ใช่ Pro และ Android เกือบทั้งหมดไม่มี
ทางเลือกสำหรับเครื่องเหล่านี้คือ **monocular depth estimation**: เครือข่ายประสาทที่ทำนาย depth จากภาพ RGB ภาพเดียว
ซึ่งก้าวหน้าอย่างมากในช่วง 2020–2024 (MiDaS, Depth Anything) จนให้ผลที่ "ดูดี" บนภาพเดี่ยว

คำถามที่ผู้พัฒนาผลิตภัณฑ์ต้องตอบก่อนตัดสินใจลงทุนคือ: **ถ้าแทนที่ LiDAR ด้วยโมเดล monocular โดยที่ส่วนที่เหลือของระบบเหมือนเดิม
ห้องที่ได้จะคลาดเคลื่อนกี่เซนติเมตร และพอสำหรับ use case ไหนบ้าง** (วัดพื้นที่ขาย, วางเฟอร์นิเจอร์, ประเมินงานตกแต่ง)
งานวิจัย monocular depth รายงานผลเป็นตัวชี้วัดต่อภาพ (AbsRel, δ₁) บน benchmark ซึ่งไม่แปลตรง ๆ เป็น "ผนังเบี้ยวกี่เซนติเมตร"
ส่วนงาน 3D reconstruction แบบ end-to-end (Atlas, NeuralRecon, SimpleRecon) ออกแบบ pipeline ทั้งชุดรอบโมเดลของตน ทำให้เทียบ "แหล่ง depth" อย่างเดียวไม่ได้

## 1.2 แนวทาง

งานนี้เป็น **system development**: สร้าง pipeline การสร้างห้อง 3 มิติแบบคลาสสิก
(depth ต่อเฟรม → scale alignment → TSDF voxel fusion → mesh) ที่ **แหล่ง depth เป็นตัวแปรเดียวที่สลับได้**
แล้ววัดผลเชิงเรขาคณิตของ mesh เทียบกับ reference จาก laser scanner ในห้องจริง (ARKitScenes) ภายใต้ pipeline เดียวกัน 3 แหล่ง:

| แหล่ง depth | แทน |
|---|---|
| Faro laser (`highres_depth`) | ขอบเขตบนของ pipeline — ถ้า depth สมบูรณ์ pipeline เสียไปเท่าไร |
| ARKit LiDAR (`lowres_depth` 256×192) | สิ่งที่ iPhone Pro ทำได้วันนี้ |
| Monocular (Depth Anything V2, MiDaS) × วิธีหา scale | สิ่งที่เครื่องไม่มี LiDAR จะทำได้ |

การแยก **ScaleAligner** ออกเป็นขั้นตอนต่างหากทำให้แยกได้ว่า error ของ monocular มาจาก "รูปทรงผิด" หรือ "สเกลผิด":
`oracle_frame` (fit ต่อเฟรมกับ GT) = เพดานของโมเดล, `per_scene` (calibrate ครั้งเดียว) = calibration ครั้งเดียว, `sparse_points` (fit ต่อเฟรมกับจุด metric ไม่กี่ร้อยจุด) = สิ่งที่ระบบจริงที่มี VIO tracker ทำได้

## 1.3 ผลหลัก (6 ฉาก, voxel 4 cm)

- Pipeline เองเสีย 2.1 ± 0.5 cm (Chamfer) เมื่อป้อน depth จาก laser; LiDAR ของ iPad เสีย 3.0 ± 0.6 cm — ต่างกัน 1 cm เท่ากันทุกประเภทห้อง
- Monocular ที่รู้ scale ทุกเฟรม (oracle) เสีย 5.3 ± 1.3 cm — ช่องว่างจาก LiDAR 2.3 cm หรือ "Android ห่างจาก iPhone Pro ราว 2 cm ถ้าแก้ปัญหา scale ได้"
- Monocular ที่ calibrate scale ครั้งเดียวต่อฉาก เสีย 22.9 ± 2.1 cm ทุกห้องเท่า ๆ กัน; โมเดล "metric" ที่ไม่ต้อง calibrate เสีย 54.3 ± 10.6 cm
  — **ปัญหาที่แท้จริงของ monocular ไม่ใช่รูปทรงแต่คือ scale ที่แกว่งต่อมุมมอง** (§6)
- Fit scale ต่อเฟรมด้วยจุด sparse ~200 จุด (proxy ของ VIO) ปิดช่องว่างนั้นเหลือ 6.0 ± 1.7 cm (F@5cm 0.74) — ห่าง oracle 0.6 cm (§5)
- โมเดล metric ที่ condition ด้วย intrinsics (Depth Pro) ป้อน focal จริงให้และไม่ fine-tune ได้เพียง 155.4 cm (F@5cm 0.00)
  — การรู้จักกล้องไม่ใช่สิ่งที่ปัญหานี้ต้องการ
- **fine-tune โมเดล metric บน ARKitScenes แก้แถวที่ไม่ calibrate ได้โดยไม่ต้อง align ตอนใช้งาน**: 14.6 cm เมื่อครูเป็น laser scanner
  และ 16.6 cm เมื่อครูเป็น LiDAR ของ iPad (1.13×) จากเดิม 54.3 cm ตอน pretrained (§5.6)
- โมเดลเล็กลง 13× (DA-v2 Small) เสียเพิ่มเพียง 1.0 cm แต่เร็วขึ้น 6×; MiDaS small เร็วอีก 3× แต่ error เป็น 2.5× ของ Large — ขนาดโมเดลมีผลน้อยกว่า scale 17 เท่า
- จำนวนเฟรมกำหนด coverage ไม่ใช่ accuracy: ต้องเก็บ ≥ 0.8 fps เพื่อให้ห้องครบ

## 1.4 Contribution

1. Pipeline แบบ open-source ที่ `DepthSource` และ `ScaleAligner` เป็น interface สลับได้จาก config โดยไม่แตะ orchestrator
   ทำให้ตารางการทดลองทุกตารางคือ "1 loop, 1 ตัวแปร" และโค้ดเดียวกันใช้เป็น backend ของผลิตภัณฑ์ได้ (§3)
2. ตัวเลข "ซม. บนผนัง" ของ Faro / LiDAR / monocular ภายใต้ pipeline เดียวกันบนห้องจริง 6 ห้อง พร้อม reference จาก laser scanner (§4–5)
3. การวิเคราะห์ว่าช่องว่าง oracle→per-scene มาจาก scale ต่อมุมมอง ไม่ใช่ drift ตามเวลา, การแสดงว่าจุด sparse ~200 จุดต่อเฟรมกู้ oracle คืนได้ และนัยต่อการออกแบบระบบ (§6)
4. คำตอบแบบควบคุมตัวแปรของคำถาม "ใช้เซนเซอร์ในมือถือแทน laser scanner เป็น *ครู* ได้ไหม": สูตร fine-tune เดียวกัน ฉากเดียวกัน
   เปลี่ยนเฉพาะ label ตอนเทรน — ครู Faro 14.6 cm vs ครู ARKit LiDAR 16.6 cm บนห้องที่กันไว้ โดย AbsRel เท่ากัน พร้อมเปิด checkpoint และการแบ่งฉาก (§5.6)
5. ต้นทุนต่อ scan (เวลา, จำนวนเฟรม, ขนาดโมเดล, confidence ของ LiDAR) สำหรับการตัดสินใจเชิงธุรกิจ (§7)

## สิ่งที่งานนี้ *ไม่* อ้าง

- ไม่อ้างว่าแม่นกว่า Atlas / NeuralRecon / SimpleRecon — ไม่ได้เทียบ และ pipeline นี้ไม่ได้ออกแบบมาแข่ง
- ไม่อ้างว่า "จุดเด่นคือไม่ต้องใช้ LiDAR" — ผลชี้ตรงข้าม: ไม่มี LiDAR ยังต้องมีข้อมูล metric บางส่วนต่อเฟรม
  หรือไม่ก็ต้องสอน scale ให้โมเดลล่วงหน้า (§5.6) ซึ่งยังห่างจากการมีจุด metric ตอนใช้งานอยู่ 11.2 cm
- เทรนเฉพาะ metric head ของโมเดลเดียว (freeze encoder, ARKitScenes, ADR-013) โมเดลอื่นทั้งหมดใช้ pretrained
