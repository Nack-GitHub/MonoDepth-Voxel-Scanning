# experiments/

`results/<experiment>/<scene>_<run>/` — written by `roomscan run` / `roomscan sweep` (ADR-006)

| file | committed? |
|---|---|
| `config.yaml` — resolved config ที่รันจริง | ✅ |
| `metrics.json` — chamfer, precision/recall/F@t, normals, 2D metrics, timing | ✅ |
| `mesh.ply` / `.obj` / `.glb` | ❌ (gitignored) |

`roomscan report experiments/results` → `summary.csv` (ทุก run) + `<experiment>/table.md` (mean ± std ต่อ run ข้ามฉาก) — ตารางในเปเปอร์ copy จากตรงนี้

Experiment definitions live in `configs/experiments/` — not here.
