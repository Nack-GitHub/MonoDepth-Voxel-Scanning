# ADR-008: ใช้โมเดล depth แบบ pretrained เท่านั้น — ไม่มี training loop ใน repo

## Status
Accepted (2026-09-11)

## Context
Contribution ของงานคือ *การวัดผลกระทบของ depth source ต่อ geometry* ไม่ใช่โมเดล depth
การเทรน/ fine-tune คือ Project 1 (ทีหลัง) และจะกินเวลาหลายสัปดาห์

## Decision
- `models/` = wrapper บาง ๆ รอบ HuggingFace/torch.hub: `predict(rgb) -> np.ndarray` + metadata (`is_metric`, `output_kind`, `param_count`)
- ไม่มี `train.py`, ไม่มี dataloader สำหรับเทรน, ไม่มี loss
- โมเดลใหม่ = ไฟล์ใหม่ใน `models/` + 1 บรรทัดใน registry (Exp4)

## Consequences
- ✅ ขอบเขตชัด ส่งทันเวลา
- ⚠️ ถ้า metric model (DA-v2 metric indoor) generalize ไม่ดีบน ARKitScenes ก็ *รายงานตามนั้น* ไม่ fine-tune
- 🔜 Project 1 (compression/distillation) เริ่มจาก `models/base.py` interface เดิม
