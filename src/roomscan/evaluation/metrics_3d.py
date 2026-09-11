"""Mesh-vs-reference geometry metrics. THE numbers of the paper.

Protocol (fixed; state it verbatim in the paper):
  * sample N points uniformly on both meshes (eval.n_sample_points, fixed seed)
  * accuracy     = mean dist pred -> ref     (built things that don't exist?)
  * completeness = mean dist ref  -> pred    (missed things?)
  * chamfer      = (accuracy + completeness) / 2
  * precision@t / recall@t / F@t at each threshold (metres)
  * normal consistency = mean |n_pred . n_ref| over nearest pairs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics3D:
    accuracy: float
    completeness: float
    chamfer: float
    precision: dict[float, float] = field(default_factory=dict)
    recall: dict[float, float] = field(default_factory=dict)
    fscore: dict[float, float] = field(default_factory=dict)
    normal_consistency: float | None = None

    def to_flat_dict(self) -> dict[str, float]:
        out = {"accuracy": self.accuracy, "completeness": self.completeness, "chamfer": self.chamfer}
        for t in self.fscore:
            out[f"precision@{t}"] = self.precision[t]
            out[f"recall@{t}"] = self.recall[t]
            out[f"fscore@{t}"] = self.fscore[t]
        if self.normal_consistency is not None:
            out["normal_consistency"] = self.normal_consistency
        return out


def _sample(mesh, n: int, seed: int):
    import open3d as o3d

    if len(mesh.triangles) == 0:
        raise ValueError("cannot evaluate an empty mesh")
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=int(n), use_triangle_normal=True)
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def _nn(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    d, i = cKDTree(target).query(query, k=1, workers=-1)
    return d, i


def evaluate_mesh(pred_mesh, ref_mesh, *, thresholds: list[float], n_sample_points: int,
                  seed: int = 0) -> Metrics3D:
    p_pts, p_nrm = _sample(pred_mesh, n_sample_points, seed)
    r_pts, r_nrm = _sample(ref_mesh, n_sample_points, seed + 1)
    d_pr, i_pr = _nn(p_pts, r_pts)
    d_rp, _ = _nn(r_pts, p_pts)
    m = Metrics3D(accuracy=float(d_pr.mean()), completeness=float(d_rp.mean()),
                  chamfer=float((d_pr.mean() + d_rp.mean()) / 2))
    for t in thresholds:
        t = float(t)
        prec, rec = float((d_pr < t).mean()), float((d_rp < t).mean())
        m.precision[t], m.recall[t] = prec, rec
        m.fscore[t] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    if p_nrm.size and r_nrm.size:
        m.normal_consistency = float(np.abs((p_nrm * r_nrm[i_pr]).sum(axis=1)).mean())
    return m


def per_point_error(pred_mesh, ref_mesh, *, n_sample_points: int = 200_000, seed: int = 0):
    """(points [N,3], dist_to_ref [N]) sampled on the prediction — for error heatmaps (Phase 4)."""
    p_pts, _ = _sample(pred_mesh, n_sample_points, seed)
    r_pts, _ = _sample(ref_mesh, n_sample_points, seed + 1)
    d, _ = _nn(p_pts, r_pts)
    return p_pts, d
