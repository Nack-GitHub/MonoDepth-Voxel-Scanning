# 2. Related Work

> ร่างแรก 2026-09-15 — เขียนจากความเข้าใจทั่วไปของแต่ละงาน **ต้องตรวจรายละเอียด/ปี/ชื่อผู้แต่งกับต้นฉบับก่อนส่ง**
> รายการอ้างอิงท้ายไฟล์ใส่แค่ key + ชื่อ — ข้อมูลเต็ม (venue/ปี/หน้า) อยู่ใน `paper/latex/refs.bib`

## 2.1 Monocular depth estimation

โมเดล monocular depth แบ่งได้ 2 ตระกูลตามสิ่งที่ output สัญญา:

**Relative / affine-invariant.** MiDaS [midas] เสนอการเทรนรวมหลาย dataset ด้วย loss ที่ไม่สนใจ scale และ shift
ทำให้โมเดล generalize ข้ามโดเมนได้ แต่ output เป็น *disparity ที่ถูกต้องขึ้นกับ affine transform* — ต้องรู้ค่าคงที่ 2 ตัว (s, t) ต่อภาพจึงจะได้เมตร
Depth Anything [da1] และ Depth Anything V2 [da2] ขยายแนวคิดนี้ด้วยข้อมูลไม่มี label จำนวนมากและ teacher สังเคราะห์
ให้ขอบคมและรายละเอียดดีขึ้นมาก โดยยังคงเป็น affine-invariant ใน disparity
งานนี้ใช้ DA-v2 (Large / Small) และ MiDaS small เป็นตัวแทนตระกูลนี้ และ **ปฏิบัติต่อ (s, t) เป็นตัวแปรการทดลอง** (ScaleAligner) แทนที่จะซ่อนไว้ในโมเดล

**Metric.** ZoeDepth [zoedepth] และ DA-v2 รุ่น metric (fine-tune บน Hypersim / Virtual KITTI) พยายามให้ depth เป็นเมตรโดยตรง
Metric3D [metric3d] ชี้ว่าความกำกวมหลักคือ focal length ของกล้อง และแก้ด้วยการ normalize ภาพเข้า canonical camera
งานเหล่านี้รายงาน AbsRel/δ₁ ต่อภาพบน NYUv2/KITTI; ผลของเรา (§6.2) ชี้ว่าบนวิดีโอห้องจริง scale ต่อเฟรมของโมเดล metric ยังแกว่งตั้งแต่ −60 % ถึง +80 % ภายใน scan เดียว
ซึ่งเป็นสิ่งที่ตัวชี้วัดต่อภาพไม่แสดง แต่ปรากฏทันทีเมื่อ fuse หลายเฟรมเข้า volume เดียว

**Depth จาก LiDAR ในมือถือ.** ARKit ให้ depth 256×192 ที่ fuse LiDAR กับภาพ; ARKitScenes [arkitscenes] เป็น dataset แรกที่ปล่อยทั้ง depth นี้
และ depth จาก Faro laser scanner ในห้องเดียวกัน ทำให้เทียบ "LiDAR มือถือ vs laser" ได้ตรง ๆ — งานนี้ใช้ทั้งสองเป็น 2 แถวของตาราง

## 2.2 Multi-view 3D reconstruction จากวิดีโอ

**Depth fusion แบบคลาสสิก.** TSDF volumetric fusion [curless96] และ KinectFusion [kinectfusion] รวม depth หลายเฟรมด้วย truncated signed distance ต่อ voxel
แล้วสกัดผิวด้วย marching cubes ข้อดีที่งานนี้ต้องการคือ **fusion ไม่รู้และไม่สนว่า depth มาจากไหน** — ตราบใดที่เป็นเมตรและมี pose
จึงเป็น "ส่วนคงที่" ที่แฟร์กับทุกแหล่ง depth เราใช้ implementation ใน Open3D [open3d]

**End-to-end learned reconstruction.** Atlas [atlas] ทำนาย TSDF ตรงจาก features ของหลายภาพ, NeuralRecon [neuralrecon] ทำแบบ incremental แบบ real-time,
SimpleRecon [simplerecon] แสดงว่า multi-view depth ที่ดีพอ + fusion คลาสสิกก็แข่งได้ งานเหล่านี้เป็น baseline ที่แข็งแรงบน ScanNet
แต่ทุกงานเรียนรู้ depth *จากหลายภาพ* — ไม่ตอบคำถามว่า "ถ้ามีแค่โมเดล monocular ต่อภาพ + pose จาก VIO ของเครื่อง จะได้ห้องแบบไหน"
ซึ่งเป็นสถานการณ์ของแอปบนเครื่องไม่มี LiDAR งานนี้จึง **ไม่เทียบกับงานกลุ่มนี้** และไม่อ้างว่าดีกว่า

**Scale ใน monocular SLAM / reconstruction.** การใช้ monocular depth ใน SLAM ต้องแก้ scale ต่อเฟรม เช่นด้วยจุด sparse จาก tracking
(แนวทางใน CNN-SLAM [cnnslam] และงานตามมา) หรือ optimize scale/shift ร่วมกับ pose (เช่น [monosdf] ใช้ monocular prior ใน neural surface)
งานนี้วาง `sparse_points` aligner ตามแนวทางนี้ และประเมินคู่กับ `oracle_frame` และ `per_scene` เพื่อวัด "ราคาของการไม่รู้ scale" ที่สามระดับของข้อมูล metric ที่มี

## 2.3 ผลิตภัณฑ์ room scanning

Apple RoomPlan [roomplan] ให้ floor plan + กล่องเฟอร์นิเจอร์จาก LiDAR บน iPhone/iPad Pro; Polycam และแอปคล้ายกันให้ mesh/photogrammetry
ทั้งหมดต้องใช้ LiDAR สำหรับโหมด room หรือใช้ photogrammetry แบบ offline ที่ช้าและต้องถ่ายเยอะ
ไม่มีรายงานสาธารณะว่าผลิตภัณฑ์เหล่านี้คลาดเคลื่อนกี่เซนติเมตรเทียบ laser scanner — งานนี้ให้ตัวเลขนั้นอย่างน้อยสำหรับ pipeline ของเราเอง

## 2.4 ตำแหน่งของงานนี้

| | เทรน | เทียบ depth source | reference | ตัวชี้วัด |
|---|---|---|---|---|
| งาน monocular depth | ✓ | ✗ (โมเดลเดียว) | depth ต่อภาพ | AbsRel, δ |
| Atlas / NeuralRecon / SimpleRecon | ✓ | ✗ (multi-view เท่านั้น) | ScanNet mesh | Chamfer, F-score |
| **งานนี้** | ✗ (pretrained) | ✓ laser / LiDAR / mono × aligner | Faro laser (ARKitScenes) | Chamfer, F@τ + AbsRel/δ ต่อเฟรม + เวลา |

## References (key → ต้องเติมข้อมูลจริง)

- [midas] Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer" (MiDaS)
- [da1] Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data"
- [da2] Yang et al., "Depth Anything V2"
- [zoedepth] Bhat et al., "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth"
- [metric3d] Yin et al., "Metric3D: Towards Zero-shot Metric 3D Prediction from a Single Image"
- [arkitscenes] Baruch et al., "ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data"
- [curless96] Curless & Levoy, "A Volumetric Method for Building Complex Models from Range Images"
- [kinectfusion] Newcombe et al., "KinectFusion: Real-Time Dense Surface Mapping and Tracking"
- [open3d] Zhou, Park, Koltun, "Open3D: A Modern Library for 3D Data Processing"
- [atlas] Murez et al., "Atlas: End-to-End 3D Scene Reconstruction from Posed Images"
- [neuralrecon] Sun et al., "NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video"
- [simplerecon] Sayed et al., "SimpleRecon: 3D Reconstruction Without 3D Convolutions"
- [cnnslam] Tateno et al., "CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction"
- [monosdf] Yu et al., "MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction"
- [roomplan] Apple, RoomPlan (developer documentation)
