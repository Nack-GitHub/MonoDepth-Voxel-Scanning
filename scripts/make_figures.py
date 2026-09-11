"""Phase 4 figures from finished runs (no GUI needed).

    python scripts/make_figures.py experiments/results/exp1_depth_source --out paper/figures

For every run dir with mesh.ply it writes:
  <run>_heatmap.ply   prediction points coloured by distance-to-reference (open in MeshLab)
  <run>_topdown.png   top-down scatter of the same, shared colour scale (0-10 cm)
and one <experiment>_topdown_grid.png with all runs side by side.
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
    args = ap.parse_args()

    exp_dir, out = Path(args.experiment_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in exp_dir.iterdir() if (p / "mesh.ply").is_file())
    if not runs:
        raise SystemExit(f"no mesh.ply under {exp_dir}")

    panels = []
    for run in runs:
        cfg = load_config(run / "config.yaml")
        ref = build_dataset(cfg.dataset).gt_mesh()
        pred = o3d.io.read_triangle_mesh(str(run / "mesh.ply"))
        pts, d = per_point_error(pred, ref, n_sample_points=args.n)
        cmap = plt.get_cmap("turbo")
        colors = cmap(np.clip(d / args.vmax, 0, 1))[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out / f"{run.name}_heatmap.ply"), pcd)

        m = json.loads((run / "metrics.json").read_text())
        m3 = m["metrics_3d"]
        title = f"{run.name}\nchamfer {m3['chamfer']*100:.1f} cm, F@5cm {m3['fscore@0.05']:.2f}"
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=d * 100, s=0.2, cmap="turbo", vmin=0, vmax=args.vmax * 100)
        _style(ax, title, 9)
        fig.colorbar(sc, ax=ax, label="error (cm)", shrink=0.7)
        fig.savefig(out / f"{run.name}_topdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        panels.append((title, pts, d))
        print(f"{run.name}: heatmap + topdown written")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (title, pts, d) in zip(axes[0], panels, strict=True):
        ax.scatter(pts[:, 0], pts[:, 1], c=d * 100, s=0.1, cmap="turbo", vmin=0, vmax=args.vmax * 100)
        _style(ax, title, 8)
    fig.savefig(out / f"{exp_dir.name}_topdown_grid.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / f'{exp_dir.name}_topdown_grid.png'}")


def _style(ax, title: str, fontsize: int) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


if __name__ == "__main__":
    main()
