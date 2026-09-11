"""Mesh export. ply for evaluation/MeshLab; obj for Blender; glb for a future web viewer."""

from __future__ import annotations

from pathlib import Path


def export_mesh(mesh, out_dir: str | Path, stem: str, formats: list[str]) -> list[Path]:
    import open3d as o3d

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt in ("ply", "obj"):
            o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_normals=True)
        elif fmt == "glb":
            _write_glb(mesh, path)
        else:
            raise ValueError(f"unknown export format {fmt!r} (ply | obj | glb)")
        written.append(path)
    return written


def _write_glb(mesh, path: Path) -> None:
    import numpy as np

    try:
        import trimesh
    except ImportError as e:
        raise ImportError("glb export needs `pip install trimesh`") from e
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         vertex_colors=colors, process=False)
    tm.export(str(path))
