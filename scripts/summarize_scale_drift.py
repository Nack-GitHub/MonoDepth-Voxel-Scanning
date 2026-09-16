"""Aggregate scripts/analyze_scale_drift.py JSONs across scenes -> markdown tables for paper §6.

    python scripts/summarize_scale_drift.py [paper/analysis]

One table per model. Columns answer the §6 questions per scene: how much does the
oracle scale swing between frames (range, IQR ratio), is it drift (corr with time)
or view-dependent (corr with the frame's median GT depth), and what each aligner
costs in per-frame AbsRel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _col(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def summarize(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = d["rows"]
    s, t = _col(rows, "oracle_s"), _col(rows, "timestamp")
    gm, ps, po = _col(rows, "gt_median"), _col(rows, "abs_rel_per_scene"), _col(rows, "abs_rel_oracle")
    near = gm < 1.0
    out = {
        "scene": d["scene"], "model": d["model"], "n": len(rows),
        "s_range": s.max() / s.min(), "s_iqr": np.percentile(s, 75) / np.percentile(s, 25),
        "corr_time": _corr(s, t), "corr_depth": _corr(s, gm),
        "oracle": po.mean(), "per_scene": ps.mean(), "per_scene_max": ps.max(),
        "per_scene_far": ps[~near].mean() if (~near).any() else float("nan"), "n_near": int(near.sum()),
        "scale_only": _col(rows, "abs_rel_scale_only").mean(), "stride": d["stride"],
    }
    raw = _col(rows, "abs_rel_raw")
    if raw.size:                                  # metric model
        r = _col(rows, "gt_over_pred_median")
        out.update({"raw": raw.mean(), "ratio_med": np.median(r), "ratio_min": r.min(), "ratio_max": r.max(),
                    "t_med": float(np.median(_col(rows, "oracle_t")))})
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/analysis")
    items = [summarize(p) for p in sorted(root.glob("*_scale_drift_*.json"))]
    for model in dict.fromkeys(i["model"] for i in items):
        rows = [i for i in items if i["model"] == model]
        print(f"\n### {model}  (stride {rows[0]['stride']})\n")
        if "raw" in rows[0]:
            print("| scene | n | gt/pred median | range | oracle t | AbsRel raw | scale-only/frame | oracle | "
                  "per_scene |")
            print("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                print(f"| {r['scene']} | {r['n']} | {r['ratio_med']:.2f} | {r['ratio_min']:.2f}–{r['ratio_max']:.2f} | "
                      f"{r['t_med']:+.2f} | {r['raw']:.3f} | {r['scale_only']:.3f} | {r['oracle']:.3f} | "
                      f"{r['per_scene']:.3f} |")
        else:
            print("| scene | n | s max/min | s p75/p25 | corr(s,t) | corr(s,depth) | AbsRel oracle | per_scene | "
                  "per_scene w/o close-up | scale-only(depth) | close-ups |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                print(f"| {r['scene']} | {r['n']} | {r['s_range']:.1f}× | {r['s_iqr']:.2f} | {r['corr_time']:+.2f} | "
                      f"{r['corr_depth']:+.2f} | {r['oracle']:.3f} | {r['per_scene']:.3f} | {r['per_scene_far']:.3f} | "
                      f"{r['scale_only']:.3f} | {r['n_near']}/{r['n']} |")
        if len(rows) > 1:
            keys = ["oracle", "per_scene", "scale_only"] + (["raw"] if "raw" in rows[0] else [])
            print("| **mean** | | | | | | " + " | ".join(f"{np.mean([r[k] for r in rows]):.3f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
