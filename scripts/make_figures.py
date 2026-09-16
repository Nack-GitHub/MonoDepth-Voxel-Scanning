"""Phase 4 figures from finished runs (no GUI needed).

    python scripts/make_figures.py experiments/results/exp1_depth_source --out paper/figures [--per-run]

Writes <out>/<experiment>_topdown_grid.png: one row per scene, one column per run,
top-down scatter of prediction points coloured by distance-to-reference (shared
scale 0-10 cm). With --per-run it also writes, under <out>/<experiment>/:
  <scene>_<run>_heatmap.ply   the same points, for MeshLab
  <scene>_<run>_topdown.png   one panel per run
(per-run files live in a sub-folder because run names repeat across exp3 variants.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roomscan.config import load_config
from roomscan.dataio import build_dataset
from roomscan.evaluation.metrics_3d import per_point_error


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--vmax", type=float, default=0.10, help="colour scale max, metres")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--per-run", action="store_true", help="also write per-run heatmap.ply + topdown.png")
    args = ap.parse_args()

    exp_dir, out = Path(args.experiment_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in exp_dir.iterdir() if (p / "mesh.ply").is_file())
    if not runs:
        raise SystemExit(f"no mesh.ply under {exp_dir}")
    per_run_dir = out / exp_dir.name
    if args.per_run:
        per_run_dir.mkdir(exist_ok=True)

    panels: dict[str, dict[str, tuple]] = {}     # scene -> run -> (title, pts, d)
    refs: dict[str, object] = {}
    for run in runs:
        cfg = load_config(run / "config.yaml")
        scene = str(cfg.dataset.scene)
        run_name = run.name.split("_", 1)[1] if run.name.startswith(scene + "_") else run.name
        if scene not in refs:
            refs[scene] = build_dataset(cfg.dataset).gt_mesh()
        pred = o3d.io.read_triangle_mesh(str(run / "mesh.ply"))
        pts, d = per_point_error(pred, refs[scene], n_sample_points=args.n)

        m3 = json.loads((run / "metrics.json").read_text())["metrics_3d"]
        title = f"{run.name}\nchamfer {m3['chamfer']*100:.1f} cm, F@5cm {m3['fscore@0.05']:.2f}"
        panels.setdefault(scene, {})[run_name] = (title, pts, d)
        if args.per_run:
            cmap = plt.get_cmap("turbo")
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pcd.colors = o3d.utility.Vector3dVector(cmap(np.clip(d / args.vmax, 0, 1))[:, :3])
            o3d.io.write_point_cloud(str(per_run_dir / f"{run.name}_heatmap.ply"), pcd)
            fig, ax = plt.subplots(figsize=(5, 5))
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=d * 100, s=0.2, cmap="turbo", vmin=0, vmax=args.vmax * 100)
            _style(ax, title, 9)
            fig.colorbar(sc, ax=ax, label="error (cm)", shrink=0.7)
            fig.savefig(per_run_dir / f"{run.name}_topdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"{run.name}: done")

    scenes = sorted(panels)
    run_names = list(dict.fromkeys(r for sc in scenes for r in panels[sc]))
    fig, axes = plt.subplots(len(scenes), len(run_names), figsize=(3.6 * len(run_names), 3.6 * len(scenes)),
                             squeeze=False)
    for i, scene in enumerate(scenes):
        for j, rn in enumerate(run_names):
            ax = axes[i][j]
            if rn in panels[scene]:
                title, pts, d = panels[scene][rn]
                ax.scatter(pts[:, 0], pts[:, 1], c=d * 100, s=0.1, cmap="turbo", vmin=0, vmax=args.vmax * 100)
                _style(ax, title, 8)
            else:
                ax.axis("off")
    fig.savefig(out / f"{exp_dir.name}_topdown_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / f'{exp_dir.name}_topdown_grid.png'} ({len(scenes)} scenes x {len(run_names)} runs)")


def _style(ax, title: str, fontsize: int) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


if __name__ == "__main__":
    main()
