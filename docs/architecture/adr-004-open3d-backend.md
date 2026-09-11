# ADR-004: Open3D เป็น 3D backend (TSDF, marching cubes, mesh I/O, eval)

## Status
Accepted (2026-09-11)

## Options

| Option | Pros | Cons |
|---|---|---|
| **Open3D (legacy `pipelines.integration.ScalableTSDFVolume`)** | มี TSDF + marching cubes + KD-tree + I/O + visualizer ในตัว; API นิ่ง; ตัวอย่างเยอะ | CPU-only, ช้ากว่า GPU; voxel 2 cm บนห้องใหญ่ใช้ RAM สูง |
| Open3D tensor `VoxelBlockGrid` (CUDA) | เร็วมาก | API เปลี่ยนบ่อย; ต้องมี CUDA (Colab ได้, Mac ไม่ได้) |
| เขียน TSDF เองด้วย numpy/torch | เข้าใจลึก | 1–2 สัปดาห์ที่ไม่ใช่ contribution |
| PyTorch3D / Kaolin | GPU | ไม่มี TSDF integration สำเร็จรูป |

## Decision
Open3D legacy TSDF ห่อไว้ใน `geometry/tsdf_fusion.py::TSDFFusion` — pipeline ไม่เรียก Open3D ตรง ๆ
เวลาจะย้ายไป GPU: เขียน class ที่สอง interface เดิม (`integrate`, `extract_mesh`)

## Trade-offs
- ยอมช้าเพื่อให้รันได้ทั้ง Mac (dev) และ Colab (full runs)
- Exp2 (voxel 2 cm) อาจต้องรันบน Colab RAM สูง — บันทึก RAM peak ลงตารางอยู่แล้ว

## Revisit trigger
ถ้า Exp3 (ทุกเฟรม, stride=1) ใช้เวลา >1 ชม./ฉาก → ทำ `TSDFFusionGPU`
