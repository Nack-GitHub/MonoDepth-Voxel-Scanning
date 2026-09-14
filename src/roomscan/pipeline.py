"""Orchestrator. Reads like the block diagram in docs/architecture/README.md.

    dataset.frames() -> depth_source.get_depth() -> aligner.align() -> fusion.integrate()
                     -> fusion.extract_mesh() -> clean_mesh() -> export + evaluate_mesh()

This file must stay boring: no Open3D calls, no torch, no file-format details.
Anything clever belongs in the stage that owns it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from roomscan.config import to_yaml
from roomscan.dataio import SceneDataset, build_dataset
from roomscan.depth_sources import DepthSource, build_depth_source
from roomscan.evaluation.metrics_2d import Metrics2D, depth_metrics, mean_metrics
from roomscan.evaluation.metrics_3d import Metrics3D, evaluate_mesh
from roomscan.export import export_mesh
from roomscan.geometry.postprocess import clean_mesh
from roomscan.geometry.scale_align import ScaleAligner, build_aligner
from roomscan.geometry.tsdf_fusion import TSDFFusion
from roomscan.types import Intrinsics, Timing


@dataclass
class RunResult:
    run_name: str
    scene: str
    depth_source: str
    aligner: str
    n_frames: int
    voxel_size: float
    metrics_3d: Metrics3D | None
    metrics_2d: Metrics2D | None
    timing: Timing
    out_dir: Path

    def write(self) -> None:
        """metrics.json + config.yaml are the only things a run must leave behind (ADR-006)."""
        payload = {
            "run_name": self.run_name, "scene": self.scene,
            "depth_source": self.depth_source, "aligner": self.aligner,
            "n_frames": self.n_frames, "voxel_size": self.voxel_size,
            "timing": {**asdict(self.timing), "total": self.timing.total},
            "metrics_3d": self.metrics_3d.to_flat_dict() if self.metrics_3d else None,
            "metrics_2d": asdict(self.metrics_2d) if self.metrics_2d else None,
        }
        (self.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


class ReconstructionPipeline:
    def __init__(self, cfg: DictConfig, *, dataset: SceneDataset | None = None,
                 depth_source: DepthSource | None = None, aligner: ScaleAligner | None = None):
        # Injection points exist so tests can pass synthetic scenes / fake sources.
        self.cfg = cfg
        self.device = _pick_device(cfg.runtime.device)
        # `is not None`, not `or`: a SceneDataset with __len__ == 0 is falsy.
        self.dataset = dataset if dataset is not None else build_dataset(cfg.dataset)
        self.depth_source = (depth_source if depth_source is not None
                             else build_depth_source(cfg.depth, device=self.device))
        self.aligner = aligner if aligner is not None else build_aligner(cfg.depth)
        self._check_consistency()

    def _check_consistency(self) -> None:
        if self.depth_source.is_metric and self.aligner.name != "identity":
            raise ValueError(f"{self.depth_source.name} is already metric; use aligner=identity")
        if not self.depth_source.is_metric and self.aligner.name == "identity":
            raise ValueError(f"{self.depth_source.name} is relative; pick a non-identity aligner")

    def run(self) -> RunResult:
        cfg, t = self.cfg, Timing()
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(to_yaml(cfg))

        fusion = TSDFFusion(cfg.fusion.voxel_size, cfg.fusion.sdf_trunc, cfg.fusion.depth_trunc,
                            cfg.fusion.depth_min, self.dataset.intrinsics)
        frames_iter = self.dataset.frames(stride=cfg.dataset.frame_stride,
                                          max_frames=cfg.dataset.max_frames)

        # --- optional one-off calibration (per_scene / sparse_points) ---
        frames = list(frames_iter)
        if hasattr(self.aligner, "fit_frames"):
            k = max(1, len(frames) // max(1, self.aligner.fit_frames))
            fit_set = frames[::k][: self.aligner.fit_frames]   # spread over the scan, not the first N
            self.aligner.fit(fit_set, [self.depth_source.get_depth(f) for f in fit_set])

        # --- main loop: one variable (depth_source), everything else fixed ---
        per_frame_2d: list[Metrics2D] = []
        for frame in frames:
            t0 = time.perf_counter()
            pred = self.depth_source.get_depth(frame)
            t.depth += time.perf_counter() - t0

            t0 = time.perf_counter()
            depth_m = _to_grid(self.aligner.align(pred, frame), self.dataset.intrinsics)
            if cfg.eval.get("mask_to_gt", False) and frame.gt_depth is not None:
                depth_m = np.where(frame.gt_depth > 0, depth_m, 0.0).astype(np.float32)
            t.align += time.perf_counter() - t0

            if cfg.eval.compute_2d_metrics and frame.gt_depth is not None:
                per_frame_2d.append(depth_metrics(depth_m, frame.gt_depth,
                                                  cfg.fusion.depth_min, cfg.fusion.depth_trunc))

            t0 = time.perf_counter()
            fusion.integrate(frame.rgb, depth_m, frame.pose_c2w)
            t.fusion += time.perf_counter() - t0

        t0 = time.perf_counter()
        mesh = fusion.extract_mesh()
        t.extract = time.perf_counter() - t0

        t0 = time.perf_counter()
        mesh = clean_mesh(mesh, remove_small_clusters=cfg.postprocess.remove_small_clusters,
                          min_cluster_triangles=cfg.postprocess.min_cluster_triangles,
                          simplify_target_triangles=cfg.postprocess.simplify_target_triangles)
        t.postprocess = time.perf_counter() - t0

        if cfg.output.save_mesh:
            export_mesh(mesh, out_dir, "mesh", list(cfg.output.export_formats))

        metrics_3d = None
        gt_mesh = self.dataset.gt_mesh()   # may be a *derived* reference mesh (ADR-009)
        if gt_mesh is not None:
            t0 = time.perf_counter()
            metrics_3d = evaluate_mesh(mesh, gt_mesh, thresholds=list(cfg.eval.fscore_thresholds),
                                       n_sample_points=cfg.eval.n_sample_points,
                                       seed=cfg.runtime.seed)
            t.eval = time.perf_counter() - t0

        result = RunResult(
            run_name=out_dir.name, scene=self.dataset.scene_id,
            depth_source=self.depth_source.name, aligner=self.aligner.name,
            n_frames=fusion.n_frames, voxel_size=cfg.fusion.voxel_size,
            metrics_3d=metrics_3d, metrics_2d=mean_metrics(per_frame_2d) if per_frame_2d else None,
            timing=t, out_dir=out_dir,
        )
        result.write()
        return result

    def _out_dir(self) -> Path:
        cfg = self.cfg
        name = cfg.output.run_name or (
            f"{cfg.dataset.scene}_{self.depth_source.name.replace(':', '-')}"
            f"_{self.aligner.name}_v{int(cfg.fusion.voxel_size * 100):02d}_s{cfg.dataset.frame_stride}"
        )
        return Path(cfg.output.root) / cfg.output.experiment / name


def _to_grid(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Every source's depth lands on the fusion grid through this one function (nearest, keeps 0)."""
    if depth.shape == (intr.height, intr.width):
        return depth
    return cv2.resize(depth, (intr.width, intr.height), interpolation=cv2.INTER_NEAREST)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
