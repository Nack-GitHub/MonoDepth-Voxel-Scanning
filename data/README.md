# data/

ทุกอย่างในโฟลเดอร์นี้ **ไม่ถูก commit** (ดู `.gitignore`) — ARKitScenes มี license ของ Apple และหลายร้อย GB ถ้าโหลดครบ

## Dataset: ARKitScenes (ADR-009)

Repo: https://github.com/apple-aiml-research/ARKitScenes — โหลดด้วย `download_data.py` ของเขา
เลือกเฉพาะ `raw` assets ที่ pipeline ใช้ (คำสั่งเต็มใน `scripts/README.md`)

```
data/
└── arkitscenes/
    ├── raw/
    │   └── Training/
    │       └── 41069021/                      ← video_id
    │           ├── wide/                       1920x1440 RGB   (<video_id>_<ts>.png)
    │           ├── wide_intrinsics/            .pincam: "w h fx fy cx cy"
    │           ├── lowres_wide/                256x192 RGB
    │           ├── lowres_wide_intrinsics/
    │           ├── lowres_depth/               256x192 uint16 mm, ARKit LiDAR  → depth.source=lidar
    │           ├── confidence/                 256x192, 0/1/2
    │           ├── highres_depth/              Faro laser GT (บางฉากเท่านั้น) → depth.source=gt
    │           ├── lowres_wide.traj            poses: "ts rx ry rz tx ty tz"
    │           ├── 41069021_3dod_mesh.ply      mesh ที่ ARKit สร้าง (ไม่ใช่ laser GT)
    │           └── reference_mesh.ply          สร้างเองด้วย scripts/build_reference_mesh.py
    └── metadata.csv                            รายชื่อ scan + ว่ามี highres_depth ไหม
```

## เลือกฉาก — ต้องมี `highres_depth`

ARKitScenes มี Faro GT depth เฉพาะ subset ดู `metadata.csv` / `raw/…/highres_depth/` ว่ามีไฟล์
เริ่มจาก **3 ฉาก** ที่:

| เกณฑ์ | ทำไม |
|---|---|
| มี `highres_depth` ครอบคลุมเฟรมส่วนใหญ่ | ไม่งั้นไม่มี GT ให้ eval |
| ห้องมาตรฐาน 1 / ห้องใหญ่ 1 / ผนังเรียบ-กระจก 1 | ครอบคลุม failure case ใน Phase 4 |
| ความยาว 1–3k เฟรม | Exp3 (stride 1) ยังรันไหว |

video_id ใน `configs/experiments/*.yaml` เป็นตัวอย่าง — แก้หลังโหลดจริง
