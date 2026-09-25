"""2D-only evaluation: the pipeline's per-frame depth metrics without fusion or a reference mesh (ADR-014).

    roomscan eval2d   configs/experiments/exp7_valfold_2d.yaml     # scenes that have Faro depth but no mesh
    roomscan reeval2d experiments/results/exp6_finetune/*/          # recompute metrics_2d of finished runs

Same frames, same aligner fit, same fusion grid (pipeline._to_grid), same mask_to_gt and the same
depth_metrics(depth_min, depth_trunc) call as ReconstructionPipeline.run() — so a number from here and
one from a full run mean the same thing. Nothing here changes pipeline.py.

reeval2d groups runs by (scene, depth source) and predicts each frame once per group: the Exp1/3/4/6
rows that share a model on a scene share its forward passes. metrics_3d in metrics.json is kept as is.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from roomscan.dataio import build_dataset
from roomscan.depth_sources import DepthSource, build_depth_source
from roomscan.evaluation.metrics_2d import PROTOCOL, Metrics2D, depth_metrics, mean_metrics
from roomscan.geometry.scale_align import build_aligner
from roomscan.pipeline import RunResult, _pick_device, _to_grid
from roomscan.types import Timing


class _CachedSource(DepthSource):
    """Wraps a depth source; remembers get_depth per frame timestamp (mono forward passes are the cost)."""

    def __init__(self, inner: DepthSource):
        self.inner = inner
        self.cache: dict[float, np.ndarray] = {}
        self.n_forward = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def is_metric(self) -> bool:
        return self.inner.is_metric

    def get_depth(self, frame) -> np.ndarray:
        ts = float(frame.extra.get("timestamp", frame.idx))
        if ts not in self.cache:
            self.cache[ts] = self.inner.get_depth(frame)
            self.n_forward += 1
        return self.cache[ts]


def evaluate_2d(cfg: DictConfig, depth_source: DepthSource, dataset=None) -> tuple[Metrics2D | None, int, Timing]:
    """-> (mean 2D metrics, frames scored, timing). Mirrors the 2D path of ReconstructionPipeline.run()."""
    t = Timing()
    dataset = dataset if dataset is not None else build_dataset(cfg.dataset)
    aligner = build_aligner(cfg.depth)
    frames = list(dataset.frames(stride=cfg.dataset.frame_stride, max_frames=cfg.dataset.max_frames))
    if hasattr(aligner, "fit_frames"):
        k = max(1, len(frames) // max(1, aligner.fit_frames))
        fit_set = frames[::k][: aligner.fit_frames]
        aligner.fit(fit_set, [depth_source.get_depth(f) for f in fit_set])

    items: list[Metrics2D] = []
    for frame in frames:
        if frame.gt_depth is None:
            continue
        t0 = time.perf_counter()
        pred = depth_source.get_depth(frame)
        t.depth += time.perf_counter() - t0
        t0 = time.perf_counter()
        depth_m = _to_grid(aligner.align(pred, frame), dataset.intrinsics)
        if cfg.eval.get("mask_to_gt", False):
            depth_m = np.where(frame.gt_depth > 0, depth_m, 0.0).astype(np.float32)
        t.align += time.perf_counter() - t0
        items.append(depth_metrics(depth_m, frame.gt_depth, cfg.fusion.depth_min, cfg.fusion.depth_trunc))
    return (mean_metrics(items) if items else None), len(items), t


def _source_key(cfg: DictConfig) -> tuple:
    d = cfg.depth
    return (str(cfg.dataset.name), str(cfg.dataset.root), str(cfg.dataset.scene), str(d.source),
            str(d.get("model")), int(d.get("lidar_min_confidence", 0) or 0))


def reevaluate_runs(run_dirs: Iterable[str | Path], *, force: bool = False, log=print) -> int:
    """Recompute metrics_2d of finished runs in place. Runs already at the current protocol are skipped."""
    groups: dict[tuple, list[tuple[Path, DictConfig]]] = defaultdict(list)
    for rd in map(Path, run_dirs):
        mf, cf = rd / "metrics.json", rd / "config.yaml"
        if not (mf.is_file() and cf.is_file()):
            continue
        m2 = json.loads(mf.read_text()).get("metrics_2d")
        if m2 is None or (not force and m2.get("protocol_2d") == PROTOCOL):
            continue
        cfg = OmegaConf.load(cf)
        groups[_source_key(cfg)].append((rd, cfg))

    n_done = 0
    for key, runs in groups.items():
        cfg0 = runs[0][1]
        t0 = time.perf_counter()
        try:
            source = _CachedSource(build_depth_source(cfg0.depth, device=_pick_device(cfg0.runtime.device)))
        except Exception as e:  # noqa: BLE001 — a long queue must not die on one missing model/scene
            log(f"[group {key}] SKIPPED: {type(e).__name__}: {e}")
            continue
        for rd, cfg in runs:
            try:
                metrics, n, _ = evaluate_2d(cfg, source)
            except Exception as e:  # noqa: BLE001
                log(f"{rd}: SKIPPED: {type(e).__name__}: {e}")
                continue
            payload = json.loads((rd / "metrics.json").read_text())
            old = payload.get("metrics_2d") or {}
            payload["metrics_2d"] = metrics.__dict__ if metrics else None
            (rd / "metrics.json").write_text(json.dumps(payload, indent=2))
            n_done += 1
            log(f"{rd}: abs_rel {old.get('abs_rel', float('nan')):.4f} -> "
                f"{metrics.abs_rel if metrics else float('nan'):.4f}  "
                f"delta1 {old.get('delta1', float('nan')):.3f} -> {metrics.delta1 if metrics else float('nan'):.3f}"
                f"  frames={n}")
        log(f"[group {key[2]} {key[3]}:{key[4]}] {len(runs)} runs, {source.n_forward} forward passes, "
            f"{time.perf_counter() - t0:.0f}s")
        del source
    return n_done


def run_2d_experiment(experiment_file: str | Path, *, overrides: list[str] | None = None,
                      scenes: list[str] | None = None, skip_existing: bool = False, log=print) -> None:
    """Like `roomscan sweep` but 2D only: metrics.json with metrics_3d = null, no mesh."""
    from roomscan.config import apply_overrides, load_config, to_yaml

    exp = OmegaConf.load(experiment_file)
    base = load_config(exp.base, overrides or [])
    for scene in scenes or [str(x) for x in exp.scenes]:
        datasets: dict[str, object] = {}     # one loader per distinct dataset block (runs usually share it)
        for run in exp.runs:
            cfg = apply_overrides(base, dict(run.overrides))
            cfg.dataset.scene = str(scene)
            cfg.output.experiment = exp.experiment
            cfg.output.run_name = f"{scene}_{run.name}"
            out = Path(cfg.output.root) / exp.experiment / cfg.output.run_name
            if skip_existing and (out / "metrics.json").exists():
                log(f"skip {out}")
                continue
            log(f"=== {exp.experiment} / {scene} / {run.name} (2D only) ===")
            dkey = OmegaConf.to_yaml(cfg.dataset)
            if dkey not in datasets:
                datasets[dkey] = build_dataset(cfg.dataset)
            dataset = datasets[dkey]
            source = build_depth_source(cfg.depth, device=_pick_device(cfg.runtime.device))
            metrics, n, timing = evaluate_2d(cfg, source, dataset)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.yaml").write_text(to_yaml(cfg))
            RunResult(run_name=out.name, scene=str(scene), depth_source=source.name,
                      aligner=str(cfg.depth.aligner), n_frames=n, voxel_size=float(cfg.fusion.voxel_size),
                      metrics_3d=None, metrics_2d=metrics, timing=timing, out_dir=out).write()
            if metrics:
                log(f"  abs_rel={metrics.abs_rel:.4f} delta1={metrics.delta1:.3f} frames={n}")
