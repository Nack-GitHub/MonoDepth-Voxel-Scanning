# Abstract (ร่าง 2026-09-17, ปรับแกนของเปเปอร์ 2026-09-25 ตาม ADR-014)

เครื่องที่ไม่มี LiDAR สร้างห้อง 3 มิติได้ด้วยโมเดล monocular depth แต่โมเดล "metric" สำเร็จรูปอย่าง Depth Anything V2 Metric-Indoor ที่เทรนบนภาพสังเคราะห์
ให้ห้องที่คลาดเคลื่อน 54 cm เมื่อใช้ใน pipeline การสร้างห้องจริง งานนี้ถามว่า **transfer learning ด้วย depth จาก LiDAR ของมือถือเอง** — ไม่ต้องมี laser scanner —
ทำให้โมเดลนี้ดีกว่าตัว pretrained จริงหรือไม่ **เมื่อวัดกับ laser scanner (Faro) ซึ่งเป็น ground truth ดิบของ ARKitScenes**
เรา fine-tune เฉพาะ DPT head (freeze encoder) ด้วย label จาก LiDAR ของ iPad บน 24 ห้อง (R3) แล้วเทียบกับตัว pretrained ใน pipeline depth → TSDF → mesh เดียวกัน
ที่แหล่ง depth เป็นตัวแปรเดียวที่สลับได้จาก config บนห้องที่ไม่เคยถูกใช้เทรนหรือเลือก checkpoint: 6 ห้องที่มี reference mesh จาก Faro (3D และ 2D)
และอีก 20 ห้องจาก fold Validation (2D เทียบ Faro depth ต่อเฟรม, 2,090 เฟรม)
ผล: R3 ลด Chamfer จาก 54.3 เป็น 15.7 cm (3.5×, ดีขึ้นทุกห้อง), F@5cm จาก 0.03 เป็น 0.24 และ AbsRel ต่อเฟรมจาก 0.382 เป็น 0.134;
บน 20 ห้องใหม่ AbsRel ลดจาก 0.410 เป็น 0.130 (ดีขึ้น 19/20 ห้อง) — ทั้งหมดโดยไม่ต้อง calibrate หรือใช้เซนเซอร์ใด ๆ ตอนใช้งาน
ครู laser (14.6 cm) ไม่ได้ดีกว่าครู LiDAR อย่างวัดได้ และ Depth Pro ที่ได้ focal จริงของกล้องแต่ไม่ fine-tune ได้ 155 cm
บริบทจาก pipeline เดียวกัน: pipeline เองเสีย 2.1 cm, LiDAR ของ iPad 3.0 cm, โมเดลที่รู้ scale ทุกเฟรม 5.3 cm และการ fit scale ต่อเฟรมด้วยจุด sparse ~200 จุด 6.0 cm;
การวิเคราะห์ต่อเฟรมชี้ว่า error ที่เหลือของ R3 คือ scale ที่แกว่งตามเนื้อหาของแต่ละมุมมอง ซึ่ง fine-tune ลบ bias ของโดเมนได้แต่ไม่ได้ลดการแกว่งนั้น
⇒ supervision ที่เคยต้องใช้ laser scanner เก็บด้วยเซนเซอร์ในมือถือแทนได้ และทำให้เครื่องที่ไม่มีเซนเซอร์ได้ห้องระดับ "ดูภาพรวม" (~16 cm)
แต่ยังไม่ถึงระดับ "วางเฟอร์นิเจอร์" ซึ่งยังต้องมีข้อมูล metric ต่อเฟรม
โค้ด config ของทุกการทดลอง metrics ของทุก run checkpoint และการแบ่งห้องเปิดเผยทั้งหมด
