"""Analysis figures for the paper (no runs needed — reads paper/analysis/*.json).

    python scripts/make_paper_figures.py [--analysis paper/analysis] [--out paper/figures]

Writes (PNG + PDF each):
  scale_drift_timeline   6 small multiples: per-frame oracle scale / per-scene scale over the scan
                         (log axis; 1.0 = the one-off calibration was right for that frame);
                         close-up frames (median GT depth < 1 m) marked — §6 "per view, not drift"
  scale_vs_depth         the same ratio against the frame's median GT depth, all scenes pooled
  pipeline               block diagram of the pipeline with the two swap points (§3)
  finetune_absrel        per-scene AbsRel vs the dataset's Faro depth, pretrained DA-v2 Metric (x) against
                         the LiDAR-teacher fine-tune R3 (y), 6 test rooms (Exp6) + 20 Validation-fold rooms
                         (Exp7, 2D only); below the diagonal = the fine-tune is better (§5.1, ADR-014)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
GRID = "#e6e6e3"


def _style(plt):
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })


def _load(analysis: Path, model: str) -> list[dict]:
    out = []
    for p in sorted(analysis.glob(f"*_scale_drift_{model}.json")):
        d = json.loads(p.read_text())
        rows = [r for r in d["rows"] if r.get("oracle_s") and r["oracle_s"] > 0]
        t0 = rows[0]["timestamp"]
        d["t"] = np.array([r["timestamp"] - t0 for r in rows])
        d["ratio"] = np.array([r["oracle_s"] / d["per_scene"]["s"] for r in rows])
        d["gt_median"] = np.array([r["gt_median"] for r in rows])
        d["abs_rel_ps"] = np.array([r["abs_rel_per_scene"] for r in rows])
        out.append(d)
    return out


def fig_timeline(plt, scenes, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.3), sharey=True)
    for ax, d in zip(axes.ravel(), scenes, strict=False):
        near = d["gt_median"] < 1.0
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.plot(d["t"], d["ratio"], color=BLUE, lw=1.2, zorder=2)
        ax.scatter(d["t"][near], d["ratio"][near], s=9, color=ORANGE, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 5.0)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        ax.set_title(f"scene {d['scene']}  (n={len(d['t'])})", loc="left", fontsize=7.5)
        ax.set_xlabel("time in scan (s)")
    axes[0, 0].set_ylabel("oracle s / per-scene s")
    axes[1, 0].set_ylabel("oracle s / per-scene s")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=BLUE, lw=1.2, label="per-frame oracle scale (DA-v2 L, stride 5)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, markersize=4,
                      label="close-up frame (median GT depth < 1 m)"),
               Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)), label="one-off per-scene calibration")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"scale_drift_timeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(plt, scenes, out: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = np.concatenate([d["gt_median"] for d in scenes])
    y = np.concatenate([d["ratio"] for d in scenes])
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.scatter(x, y, s=8, color=BLUE, alpha=0.55, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 6)
    ax.set_ylim(0.2, 5)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlabel("median GT depth of the frame (m)")
    ax.set_ylabel("oracle s / per-scene s")
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    ax.text(0.98, 0.96, f"6 scenes, n={len(x)}\nlog–log r = {r:+.2f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"scale_vs_depth.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline(plt, out: Path):
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.16, 1.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    ax.axis("off")
    boxes = [
        (1, "SceneDataset", "ARKitScenes /\ncustom capture", False),
        (17.5, "DepthSource", "gt · lidar · mono\n(DA-v2, MiDaS)", True),
        (34, "ScaleAligner", "identity · oracle\nsparse · per-scene", True),
        (50.5, "TSDFFusion", "Open3D, 4 cm\n(conf. weights opt.)", False),
        (67, "Mesh", "marching cubes\n+ clean-up", False),
        (83.5, "Metrics", "Chamfer, F@τ\nvs. Faro reference", False),
    ]
    w, h, y = 14.5, 10, 6
    for x, title, sub, swap in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f4f7fb" if swap else "#fcfcfb",
                                    ec=BLUE if swap else MUTED, lw=1.4 if swap else 0.8))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=7,
                fontweight="bold", color=BLUE if swap else INK)
        ax.text(x + w / 2, y + 3.3, sub, ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.15)
        if swap:
            ax.text(x + w / 2, y + h + 3.2, "one config key", ha="center", va="center", fontsize=6.5, color=BLUE)
            ax.annotate("", xy=(x + w / 2, y + h + 0.4), xytext=(x + w / 2, y + h + 2.4),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9))
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + w
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1 - 0.2, y + h / 2), xytext=(x0 + 0.2, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(2, 1.2, "frames: RGB + pose (VIO) + optional sensor depth / confidence / sparse points",
            fontsize=6.5, color=MUTED, va="center")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pipeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_finetune_absrel(plt, results: Path, out: Path) -> bool:
    import pandas as pd

    csv = results / "summary.csv"
    if not csv.is_file():
        return False
    df = pd.read_csv(csv)
    groups = (("exp6_finetune", "6 test rooms (Exp6)", ORANGE, "o", 22),
              ("exp7_valfold_2d", "Validation-fold rooms (Exp7)", BLUE, "s", 14))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    hi, n_total = 0.0, 0
    for exp, label, color, marker, size in groups:
        sub = df[(df.experiment == exp) & df.run_name.isin(["mono_metric", "mono_ft_lidar_all"])]
        if sub.empty:
            continue
        pv = sub.pivot_table(index="scene", columns="run_name", values="abs_rel").dropna()
        n_total += len(pv)
        hi = max(hi, float(pv.values.max()))
        ax.scatter(pv["mono_metric"], pv["mono_ft_lidar_all"], s=size, marker=marker, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3, label=f"{label}, n={len(pv)}")
    if n_total == 0:
        plt.close(fig)
        return False
    lim = (0, hi * 1.08)
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(lim[1] * 0.97, lim[1] * 0.90, "no change", rotation=45, ha="right", va="bottom",
            fontsize=6.5, color=MUTED, rotation_mode="anchor")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlabel("AbsRel, pretrained DA-v2 Metric-Indoor")
    ax.set_ylabel("AbsRel, fine-tuned with iPad LiDAR, 24 rooms")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"finetune_absrel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="paper/analysis")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--model", default="da_large")
    ap.add_argument("--results", default="experiments/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style(plt)
    scenes = _load(Path(args.analysis), args.model)
    if scenes:
        fig_timeline(plt, scenes, out)
        fig_scatter(plt, scenes, out)
    fig_pipeline(plt, out)
    extra = ", finetune_absrel.*" if fig_finetune_absrel(plt, Path(args.results), out) else ""
    print(f"wrote {out}/scale_drift_timeline.*, scale_vs_depth.*, pipeline.*{extra}")


if __name__ == "__main__":
    main()
