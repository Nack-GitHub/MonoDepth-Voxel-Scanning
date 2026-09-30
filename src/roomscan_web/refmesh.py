"""The reference (Faro) mesh as the browser gets it: a light copy for the wireframe overlay and the framing.

The evaluation reference is fused at 1 cm (ADR-009) — millions of triangles, far more than a wireframe
can show or a page should download. The view copy is vertex-clustered to the pipeline's 4 cm voxel, then
decimated until flat walls are a few large triangles, and keeps positions and faces only.
Error colours are never computed from this copy (see errorcolor.py).
"""

from __future__ import annotations

import threading
from pathlib import Path

from roomscan_web._fs import is_fresh, replace_atomic

VIEW_VOXEL = 0.04           # metres
MAX_TRIANGLES = 30_000      # a reference at or below this is served as it is
_LOCK = threading.Lock()    # endpoints run in a thread pool; build each file once


def write_view(ref_path: Path, out_path: Path) -> Path:
    """Write (or reuse) the display copy of `ref_path` at `out_path`. ValueError if it has no triangles."""
    if is_fresh(out_path, ref_path):
        return out_path
    import open3d as o3d

    with _LOCK:
        if is_fresh(out_path, ref_path):
            return out_path
        mesh = o3d.io.read_triangle_mesh(str(ref_path))
        if not mesh.has_triangles():
            raise ValueError(f"{ref_path} has no triangles")
        if len(mesh.triangles) > MAX_TRIANGLES:
            mesh = mesh.simplify_vertex_clustering(VIEW_VOXEL)      # fast; quadric on the 1 cm mesh takes ~15 s
        if len(mesh.triangles) > MAX_TRIANGLES:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=MAX_TRIANGLES)
        view = o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.ply")
        o3d.io.write_triangle_mesh(str(tmp), view)
        replace_atomic(tmp, out_path)
    return out_path
