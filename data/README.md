# data/

ทุกอย่างในโฟลเดอร์นี้ **ไม่ถูก commit** (ดู `.gitignore`)

## Dataset: ARKitScenes (ADR-009)

Repo: https://github.com/apple-aiml-research/ARKitScenes — `download_data.py` curl ตรงจาก CDN ของ Apple
(ไม่มี prompt ยอมรับ license ใน script — แต่อ่าน `LICENSE` ก่อนใช้)

### ข้อเท็จจริงที่ยืนยันจาก source ของ Apple แล้ว (2026-09-12)

| | |
|---|---|
| `lowres_depth` / `confidence` / `lowres_wide` | 256×192, 60 FPS, uint16 mm / uint8 {0,1,2} / RGB |
| `highres_depth` | **1920×1440**, ~10 FPS, projected จาก Faro mesh — เฉพาะ video ที่ `is_in_upsampling=True` |
| RGB ความละเอียดสูง | **ไม่มีใน raw** — อยู่ใน dataset `upsampling` ชื่อ `color/` (1920×1440, timestamp เดียวกับ highres_depth) |
| `vga_wide` | 640×480, 30 FPS (raw) — ทางเลือกกลาง ๆ สำหรับ mono input |
| `.traj` | `ts rx ry rz tx ty tz` = **world→camera**; loader invert เป็น c2w แล้ว |
| `.pincam` | `w h fx fy cx cy` ต่อเฟรม |
| pose matching | ±0.005 s (Apple), เราใช้ nearest ภายใน 0.02 s |
| `sky_direction` | ใน `metadata.csv` — ภาพเก็บเป็น landscape แม้ถือเครื่องแนวตั้ง; `MonocularDepth` หมุนให้ upright เอง |
| Faro laser point clouds | โหลดได้ด้วย `--download_laser_scanner_point_cloud` (`has_laser_scanner_point_clouds=True`) |

Shortlist: **2,236 videos** มีทั้ง `highres_depth` และ laser point cloud (จาก `raw/metadata.csv`, 5,071 videos)

### ขนาดโดยประมาณต่อฉาก (~1–2 นาที)

| asset | ขนาด |
|---|---|
| raw: traj + intrinsics + lowres_* + confidence + mesh | ~0.3–0.6 GB |
| raw: highres_depth (1920×1440 uint16, ~600–1200 เฟรม) | ~1–2 GB |
| upsampling: color 1920×1440 | ~2–4 GB |
| **รวม/ฉาก** | **~4–7 GB** → 3 ฉาก ≈ 15–20 GB — เช็ค `df -h` ก่อน |

### คำสั่งโหลด (เลือก video_id จาก metadata ก่อน)

```bash
git clone https://github.com/apple-aiml-research/ARKitScenes.git /tmp/ARKitScenes && cd /tmp/ARKitScenes
DL=/Users/nack/MyProjects/MonoDepth-Voxel-Scanning/data/arkitscenes
# 1) metadata -> เลือก video ที่ is_in_upsampling & has_laser_scanner_point_clouds
curl -o /tmp/raw_metadata.csv https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv
# 2) raw assets ที่ pipeline ใช้ (เริ่ม 1 ฉากก่อน)
python3 download_data.py raw --split Training --video_id 42444474 --download_dir $DL \
  --raw_dataset_assets lowres_wide.traj lowres_wide_intrinsics lowres_wide lowres_depth confidence highres_depth mesh vga_wide vga_wide_intrinsics
# 3) (optional) RGB 1920x1440 สำหรับ mono
python3 download_data.py upsampling --split Training --video_id 42444474 --download_dir $DL
cp /tmp/raw_metadata.csv $DL/raw/metadata.csv     # ให้ loader อ่าน sky_direction
```

Layout ที่ได้ตรงกับที่ `dataio/arkitscenes.py` คาดหวัง (`$DL/raw/Training/<id>/...`, `$DL/upsampling/Training/<id>/color/`)

### เลือกฉาก — 3 ฉากพอ

| เกณฑ์ | ทำไม |
|---|---|
| `is_in_upsampling=True` | ไม่งั้นไม่มี Faro GT |
| ห้องมาตรฐาน 1 / ห้องใหญ่ 1 / ผนังเรียบ-กระจก 1 | failure case ใน Phase 4 |
| ~600–1200 เฟรม highres | Exp3 stride=1 ยังรันไหว |

`42444474` (visit 421069) ในตัวอย่างข้างบนคือแถวแรกของ shortlist — ยังไม่ได้ดูว่าเป็นห้องแบบไหน

## Synthetic scene (มีแล้ว ไม่ต้องโหลด)

`make synthetic` → `data/synthetic/` ห้อง 5×4×2.6 m + เฟอร์นิเจอร์ 2 ชิ้น ใน layout เดียวกับ ARKitScenes ทุกประการ
ใช้ทดสอบ pipeline / loader / metrics — ตัวเลขไม่ใช่ตัวเลขเปเปอร์
