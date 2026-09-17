# Paper outline (System Development track)

**Thesis:** เปลี่ยน depth source (Faro GT / iPad LiDAR / monocular model) โดย pipeline คงเดิม → geometry error เปลี่ยนกี่ซม. และหมายความว่าอะไรต่อ use case

1. Introduction & Business Motivation — RoomPlan จำกัด iPhone Pro; Android ต้องพึ่ง mono
2. Related Work — MiDaS, Depth Anything; Atlas / NeuralRecon / SimpleRecon (baseline, ไม่แข่ง); RoomPlan, Polycam
3. System Design — `docs/architecture/README.md` §2–§5 (DepthSource / ScaleAligner abstraction คือจุดขายเชิงวิศวกรรม)
4. Experimental Setup — ARKitScenes 6 scans, reference mesh จาก Faro (ADR-009), metrics protocol (`metrics_3d.py` docstring)
5. Results — Exp1–5 จาก `experiments/results/*/table.md` (Exp1 มีแถว mono+sparse = VIO proxy, ADR-011; Exp5 = LiDAR confidence, ADR-012)
6. Discussion — error → ซม.บนผนัง 4 ม.; LiDAR vs mono gap = "Android ห่างจาก iPhone Pro เท่าไร"; failure cases
7. Business Implications — cost/scan (Exp3 = เฟรม/วินาทีที่ต้องเก็บ: gt control / oracle / per_scene, Exp4 = เวลาโมเดล), positioning
8. Limitations & Future Work — VIO pose drift, sparse proxy vs จุด VIO จริง, 256×192 depth, Project 1 (compression), semantic layer
9. Conclusion

**ห้ามเขียน:** "แม่นกว่า Atlas" / "จุดเด่นคือไม่ต้องใช้ LiDAR"
**ต้องเขียน:** "เราวัดว่า depth source ต่างกันส่งผลต่อ geometry อย่างไร"

Figures → `figures/` (gitignore `.ply`; commit `.png`/`.pdf` ที่ใช้จริง)
Failure analysis ต่อฉาก → `analysis/<scene>_failure_analysis.md` (วัตถุดิบบท 6 และ 8)
ร่างบท → `draft/` (บทละไฟล์; 00 abstract – 09 conclusion ครบแล้ว 2026-09-17 ด้วยตัวเลข 6 ฉาก + sparse/Exp5)
ต้นฉบับอังกฤษ → `latex/` (IEEEtran; `make paper`; ตัวเลข sparse/Exp5 เป็น macro จาก `scripts/paper_numbers.py` → `numbers.tex`; `refs.bib` 15 รายการ)
รูปวิเคราะห์ → `make paper-figures` (`scale_drift_timeline`, `scale_vs_depth`, `pipeline`)
