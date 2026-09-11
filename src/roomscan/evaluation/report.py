"""Aggregate experiments/results/**/metrics.json into the paper's tables.

    roomscan report experiments/results
    -> experiments/results/summary.csv           one row per run
    -> experiments/results/<experiment>/table.md  mean ± std over scenes, per run name
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HEADLINE = ["chamfer", "accuracy", "completeness", "fscore@0.05", "precision@0.05", "recall@0.05",
            "normal_consistency", "rmse", "abs_rel", "delta1", "time_total_s", "n_frames"]


def collect_runs(results_root: str | Path) -> pd.DataFrame:
    rows = []
    for mf in sorted(Path(results_root).glob("*/*/metrics.json")):
        d = json.loads(mf.read_text())
        row = {
            "experiment": mf.parent.parent.name,
            "run": mf.parent.name,
            "run_name": mf.parent.name.split("_", 1)[1] if "_" in mf.parent.name else mf.parent.name,
            "scene": d.get("scene"), "depth_source": d.get("depth_source"), "aligner": d.get("aligner"),
            "n_frames": d.get("n_frames"), "voxel_size": d.get("voxel_size"),
            "time_total_s": (d.get("timing") or {}).get("total")
            or sum((d.get("timing") or {}).values()),
        }
        row.update(d.get("metrics_3d") or {})
        row.update({k: v for k, v in (d.get("metrics_2d") or {}).items() if k != "n_valid"})
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(results_root: str | Path) -> None:
    root = Path(results_root)
    df = collect_runs(root)
    if df.empty:
        print(f"no metrics.json under {root}")
        return
    df.to_csv(root / "summary.csv", index=False)
    cols = [c for c in HEADLINE if c in df.columns]
    for exp, g in df.groupby("experiment"):
        agg = g.groupby("run_name")[cols].agg(["mean", "std"])
        lines = [f"# {exp}", "", f"scenes: {sorted(g.scene.unique())}", "",
                 "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for run, r in agg.iterrows():
            cells = []
            for c in cols:
                mu, sd = r[(c, "mean")], r[(c, "std")]
                if c in ("chamfer", "accuracy", "completeness", "rmse"):
                    cells.append(f"{mu*100:.2f} ± {0 if pd.isna(sd) else sd*100:.2f} cm")
                elif c == "n_frames":
                    cells.append(f"{mu:.0f}")
                else:
                    cells.append(f"{mu:.3f} ± {0 if pd.isna(sd) else sd:.3f}")
            lines.append(f"| {run} | " + " | ".join(cells) + " |")
        (root / exp / "table.md").write_text("\n".join(lines) + "\n")
        print(f"wrote {root / exp / 'table.md'}")
    print(f"wrote {root / 'summary.csv'} ({len(df)} runs)")
