"""Mesh cleanup: drop floating noise clusters, optional decimation (Phase 1)."""

from __future__ import annotations


def clean_mesh(mesh, *, remove_small_clusters: bool = True, min_cluster_triangles: int = 1000,
               simplify_target_triangles: int | None = None):
    """open3d TriangleMesh -> open3d TriangleMesh.

    TODO(Phase 1): cluster_connected_triangles(); remove clusters below threshold;
    remove_degenerate/duplicated; optional simplify_quadric_decimation.
    """
    raise NotImplementedError("Phase 1")
