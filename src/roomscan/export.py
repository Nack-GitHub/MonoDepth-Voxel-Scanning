"""Mesh export. ply for evaluation/MeshLab; glb for the future web viewer (three.js)."""

from __future__ import annotations

from pathlib import Path


def export_mesh(mesh, out_dir: str | Path, stem: str, formats: list[str]) -> list[Path]:
    """TODO(Phase 1): o3d.io.write_triangle_mesh for ply/obj; glb via trimesh (add dep when needed)."""
    raise NotImplementedError("Phase 1")
