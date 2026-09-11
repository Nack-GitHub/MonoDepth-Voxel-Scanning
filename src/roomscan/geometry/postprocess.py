"""Mesh cleanup: drop floating noise clusters, optional decimation."""

from __future__ import annotations

import numpy as np


def clean_mesh(mesh, *, remove_small_clusters: bool = True, min_cluster_triangles: int = 1000,
               simplify_target_triangles: int | None = None):
    mesh = mesh.remove_degenerate_triangles().remove_duplicated_triangles() \
               .remove_duplicated_vertices().remove_unreferenced_vertices()
    if remove_small_clusters and len(mesh.triangles) > 0:
        cluster_ids, cluster_sizes, _ = mesh.cluster_connected_triangles()
        cluster_ids = np.asarray(cluster_ids)
        cluster_sizes = np.asarray(cluster_sizes)
        small = cluster_sizes[cluster_ids] < min_cluster_triangles
        if small.any() and not small.all():
            mesh.remove_triangles_by_mask(small)
            mesh = mesh.remove_unreferenced_vertices()
    if simplify_target_triangles and len(mesh.triangles) > simplify_target_triangles:
        mesh = mesh.simplify_quadric_decimation(int(simplify_target_triangles))
    mesh.compute_vertex_normals()
    return mesh
