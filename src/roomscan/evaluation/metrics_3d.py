"""Mesh-vs-GT-mesh geometry metrics (Phase 1). THE numbers of the paper.

Protocol (fix it once, never change mid-project, state it in the paper):
  * sample N points uniformly on both meshes (eval.n_sample_points)
  * accuracy  = mean dist pred->gt        (did we build things that don't exist?)
  * completeness = mean dist gt->pred     (did we miss things?)
  * chamfer = (accuracy + completeness) / 2
  * precision@t = % pred points within t of gt; recall@t likewise; F@t = harmonic mean
  * normal consistency = mean |n_pred . n_gt| over nearest pairs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics3D:
    accuracy: float                  # metres, pred -> gt
    completeness: float              # metres, gt -> pred
    chamfer: float                   # metres
    precision: dict[float, float] = field(default_factory=dict)   # threshold -> value
    recall: dict[float, float] = field(default_factory=dict)
    fscore: dict[float, float] = field(default_factory=dict)
    normal_consistency: float | None = None

    def to_flat_dict(self) -> dict[str, float]:
        """'fscore@0.05' style keys, for CSV/JSON."""
        out = {"accuracy": self.accuracy, "completeness": self.completeness,
               "chamfer": self.chamfer}
        for t in self.fscore:
            out[f"precision@{t}"] = self.precision[t]
            out[f"recall@{t}"] = self.recall[t]
            out[f"fscore@{t}"] = self.fscore[t]
        if self.normal_consistency is not None:
            out["normal_consistency"] = self.normal_consistency
        return out


def evaluate_mesh(pred_mesh, gt_mesh, *, thresholds: list[float], n_sample_points: int,
                  seed: int = 0) -> Metrics3D:
    """TODO(Phase 1): sample_points_uniformly on both; KDTreeFlann nearest neighbours;
    compute the fields above. Also return per-point distances (for error heatmaps in Phase 4)
    via a second function `per_point_error()`.
    """
    raise NotImplementedError("Phase 1")
