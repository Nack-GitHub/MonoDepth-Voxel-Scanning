# scripts/

One-off helpers ที่ *ไม่ใช่* ส่วนของ pipeline

| script | phase | what |
|---|---|---|
| `sanity_check.py` | 0 | เฟรมเดียว → point cloud → `.ply` เปิดดูใน MeshLab; gate ของ Phase 0 |
| `build_reference_mesh.py` | 1 | fuse Faro `highres_depth` ทุกเฟรมที่ voxel 1 cm → `reference_mesh.ply` (ADR-009) |

## โหลด ARKitScenes

```bash
git clone https://github.com/apple-aiml-research/ARKitScenes.git /tmp/ARKitScenes
cd /tmp/ARKitScenes
# อ่านและยอมรับ LICENSE ก่อน — download_data.py จะถามครั้งแรก
python download_data.py raw \
    --split Training \
    --video_id 41069021 41159856 42445173 \
    --download_dir /Users/nack/MyProjects/MonoDepth-Voxel-Scanning/data/arkitscenes \
    --raw_dataset_assets wide wide_intrinsics lowres_wide lowres_wide_intrinsics \
                         lowres_depth confidence highres_depth lowres_wide.traj mesh
```

- `--video_id` เลือกจาก `metadata.csv` (โหลดด้วย `python download_data.py metadata ...`) — **ต้องมี highres_depth**
- ตัด `wide` ออกถ้าพื้นที่ไม่พอ (ใช้ `lowres_wide` แทน; mono model จะได้ input 256×192 ซึ่งแย่กว่า — บันทึกไว้ในเปเปอร์ถ้าทำ)
- อย่าโหลดทั้ง split — เริ่มจาก 3 ฉาก
