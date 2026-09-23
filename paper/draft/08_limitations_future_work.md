# 8. Limitations and Future Work

> ร่าง 2026-09-16, ตรวจกับผล 6 ฉาก 2026-09-17

## 8.1 ข้อจำกัดของการวัด

**(ก) Reference ไม่ใช่ ground truth อิสระ** — reference mesh fuse มาจาก Faro depth ด้วย pipeline เดียวกัน (voxel 1 cm) และใช้ **pose เดียวกัน** กับทุกแถว
ผลคือ (i) แถว `gt` ไม่เป็นศูนย์และวัด discretisation ของเราเอง (ii) pose drift ของ VIO ถูกตัดออกจากทุกตัวเลข — ระบบจริงบนมือถือจะได้ error เพิ่มจาก pose
ซึ่งงานนี้ไม่ได้วัด และ (iii) พื้นที่ที่ Faro ไม่เห็น (หลังเฟอร์นิเจอร์, เพดานสูง) ถูก mask ออกจากทุก source เท่ากัน — ตัวเลข "completeness" จึงหมายถึง
"ครบเทียบกับที่ Faro เห็น" ไม่ใช่ครบทั้งห้อง

**(ข) ประเมินที่ ~3 fps เท่านั้น** — Faro depth มีเฉพาะเฟรมที่ Apple กรอง ทุก source จึงถูกประเมินบนเฟรมชุดนี้ แม้ LiDAR มี 60 fps และ RGB มี 30 fps
สิ่งที่ pipeline จริงจะได้จากการ fuse เฟรมที่หนาแน่นกว่า (ทั้งข้อดี: smoothing, และข้อเสีย: error สะสมมากขึ้น) ไม่ได้วัด นี่คือเหตุที่ Exp3 ตอบเรื่อง coverage ได้ แต่ตอบเรื่อง compute-vs-accuracy ได้เพียงบางส่วน (§5.3)

**(ค) resolution ของ fusion = 256×192** — เลือกให้แฟร์กับ LiDAR ทำให้ monocular model (ซึ่งทำนายที่ resolution ของภาพ 640×480) ถูก downsample ก่อน fuse
ข้อได้เปรียบเรื่องขอบคมของ DA-v2 จึงไม่ปรากฏในตัวเลข 3D; ถ้าคำถามคือ "mono เท่าที่ทำได้" ต้อง fuse ที่ resolution สูงกว่าและยอมให้ LiDAR เสียเปรียบ

**(ง) ฉาก 6 ห้อง จาก dataset เดียว อุปกรณ์เดียว** — ทุก scan ถ่ายด้วย iPad Pro ในบ้านที่ Apple เลือก ห้องออฟฟิศ / ห้องว่าง / แสงน้อย ไม่มี
และห้อง 42444474 ถูกใช้ระหว่างพัฒนา (ADR-010 เกิดจากมัน) ตัวเลขของมันจึงเป็น in-sample — เรารายงานต่อฉากเพื่อให้เห็นว่าอีก 5 ฉากยืนยันหรือไม่

**(จ) แถวที่ fine-tune เลือก checkpoint จากฉาก val เพียง 3 ห้อง** — Exp6 เก็บ checkpoint ที่ AbsRel ต่ำสุดบน val 3 ห้อง
แต่ลำดับนั้น (R3 > R2 > R1) ไม่เกิดซ้ำบนฉากเทสต์ 6 ห้อง (R1 ≥ R3 ≥ R2 ใน Chamfer) — 3 ห้องน้อยเกินกว่าจะแยก checkpoint ที่ต่างกันระดับ 0.01 AbsRel
ดังนั้นลำดับระหว่าง "ครู" ใน Exp6 ควรอ่านว่า "ไม่ต่างอย่างวัดได้" และสิ่งแรกที่ควรเพิ่มคือ val ที่ใหญ่ขึ้น (ฉากเทสต์ 6 ห้องไม่เคยถูกใช้เทรน/วัด/เลือก checkpoint)

**(ฉ) แถว Level 0 = โมเดลเดียว implementation เดียว ความละเอียดเดียว** — รัน Depth Pro ผ่าน implementation ของ HuggingFace
บนภาพ 640×480 ที่ processor ของมัน upsample เป็น 1536² เอง พร้อมป้อน focal จริง; ยังไม่ได้ลองโค้ด reference ของ Apple,
ภาพความละเอียดสูงกว่า (ไม่ได้โหลด `color` 1920×1440) หรือโมเดลอื่นในตระกูลเดียวกัน (UniDepth, Metric3D)
⇒ เคลมของเราคือ "โมเดลตัวนี้บนอินพุตแบบนี้" ไม่ใช่ทั้งตระกูล

**(ช) สูตร fine-tune แบบเดียว สถาปัตยกรรมเดียว** — ทดลองเฉพาะ DA-v2 Metric-Indoor Large แบบ freeze encoder
ยังไม่ได้ทดสอบว่าการ unfreeze encoder, loss อื่น หรือสถาปัตยกรรมอื่น จะทำให้ช่องว่างครู Faro-vs-LiDAR กว้างขึ้นหรือแคบลง รวมถึงการเทรนบนฉากมากกว่า 24 ห้อง

**(ซ) เวลาที่รายงานคือบน Apple Silicon (MPS) ด้วย PyTorch ไม่ optimize** — ไม่ใช่ตัวแทนของ on-device inference (CoreML / NNAPI) ใช้เทียบ *สัดส่วน* ระหว่างโมเดลได้ ใช้เป็นตัวเลขสัมบูรณ์ไม่ได้

## 8.2 ข้อจำกัดของระบบ

**(ก) `per_scene` ยังใช้ GT ในการ calibrate** — เป็นขอบบนของ "calibration ครั้งเดียว" ระบบจริงที่ให้ผู้ใช้วัดผนัง 1 ด้าน หรือใช้ความสูงกล้อง จะได้แย่กว่านี้
และ §6 แสดงว่าแม้ขอบบนนี้ก็ยังห่างจาก oracle มาก เพราะ scale แกว่งต่อมุมมอง — ปัญหาไม่ได้อยู่ที่วิธี calibrate แต่อยู่ที่สมมติฐาน "scale เดียวต่อฉาก"

**(ข) `sparse_points` ประเมินบน proxy** — ARKitScenes ไม่ปล่อย feature points ของ ARKit aligner จึงเห็น 200 pixel ที่ sample จาก LiDAR frame ที่ confidence สูง (ADR-011)
จุด VIO จริง triangulate มา noise มากกว่า และเกาะตาม texture แถวนี้จึงเป็น **ขอบบน** ของระบบที่ใช้ VIO; loader มี knob `sparse_noise` และจำนวนจุดสำหรับ sensitivity; การประเมินบนจุด ARKit/ARCore จริงต้องเก็บข้อมูลเอง (`dataio/custom.py` อ่านได้แล้ว)

**(ค) ไม่มี outlier rejection ก่อน fuse** — ทุกเฟรมที่มี pose ถูก integrate เท่ากัน §6 แสดงว่าเฟรมส่วนน้อย (close-up, ผนังเรียบ) สร้าง error ส่วนใหญ่ของ `per_scene`
การ weight ต่อ pixel/เฟรมมีกลไกแล้ว (ADR-012) แต่ใช้เฉพาะกับ confidence ของ LiDAR ใน Exp5 เพื่อให้ตารางหลักวัด depth source ล้วน ๆ

**(ง) pose มาจาก dataset** — ไม่มี tracking ในระบบ; MVP บนมือถือจะใช้ ARKit/ARCore pose ซึ่งคุณภาพใกล้เคียง traj ที่ใช้ แต่บน Android รุ่นล่างอาจแย่กว่า

## 8.3 Future work (เรียงตามผลกระทบต่อการตัดสินใจ MVP)

1. **จุด VIO จริง** — เก็บห้องด้วยแอป (ARKit `rawFeaturePoints` / ARCore point cloud) พร้อม reference (วัดด้วยเทป / เช่า laser) แล้วรันแถว `sparse_points` ซ้ำบนจุดจริงที่ noisy และเกาะ texture; ระหว่างนั้น sweep `sparse_noise` และจำนวนจุดบน ARKitScenes
2. **Metric model + scale-only ต่อเฟรม** — §6.2 ชี้ว่าโมเดล metric ต้องการแค่ 1 พารามิเตอร์ต่อเฟรม; ทดสอบว่า 1 จุด VIO ต่อเฟรมพอไหม
3. **Frame selection / weighting** — ทิ้งหรือลด weight เฟรมที่ GT median < 1 m หรือ variance ของ disparity ต่ำ ก่อน fuse (post-hoc จาก §6.1) ผ่านช่อง weight ของ ADR-012
4. **ห้องจริงจาก Android (ARCore depth source)** — `depth_sources/arcore.py` มี slot ใน registry แล้ว; ต้องเก็บข้อมูลเอง พร้อม reference (เช่น วัดด้วยเทป / Faro เช่า)
5. **โมเดลบีบอัด (Project 1)** — Exp4 ให้ baseline L / S / MiDaS; โมเดล distilled หรือ quantized เสียบเป็น `models/<x>.py` แล้วรัน Exp4 ซ้ำ
6. **Fuse ที่ resolution สูงกว่า 256×192 สำหรับ mono** — วัดว่า downsample กินความละเอียดของ DA-v2 ไปเท่าไร (ข้อ 8.1 ค)
7. **Semantic layer** — ป้าย ผนัง/พื้น/เฟอร์นิเจอร์ บน mesh เพื่อคำนวณพื้นที่ใช้สอย; อยู่หลัง `postprocess` ไม่กระทบตัวเลขเรขาคณิต
