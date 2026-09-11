"""Phase 1: build the evaluation reference for an ARKitScenes scan (ADR-009).

    python scripts/build_reference_mesh.py --set dataset.scene=41069021 [--voxel 0.01]

Fuses Faro `highres_depth` with the dataset poses over ALL frames at a fine
voxel and writes <scene>/reference_mesh.ply. Every run's metrics_3d is measured
against this file. Build once per scene; never rebuild mid-study.

Tautology guard: the `gt` experiment row runs at the *experiment's* voxel/stride
(4 cm / 10), not this reference's (1 cm / 1) — its non-zero Chamfer is exactly
the pipeline's own discretisation loss.
"""

from __future__ import annotations

import argparse

from roomscan.config import load_config
from roomscan.dataio import build_dataset
from roomscan.depth_sources.ground_truth import GroundTruthDepth
from roomscan.geometry.postprocess import clean_mesh
from roomscan.geometry.tsdf_fusion import TSDFFusion


def main() -> None:
    import open3d as o3d

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides + ["dataset.frame_stride=1", "dataset.max_frames=null"])
    ds = build_dataset(cfg.dataset)
    out = ds.reference_mesh_path
    if out.is_file() and not args.force:
        print(f"exists: {out} (use --force to rebuild)")
        return
    if not ds.has_faro_depth():
        raise SystemExit(f"{ds.scene_id}: no highres_depth — pick a scene with is_in_upsampling=True")

    src = GroundTruthDepth()
    fusion = TSDFFusion(args.voxel, 3 * args.voxel, cfg.fusion.depth_trunc, cfg.fusion.depth_min, ds.intrinsics)
    n = 0
    for f in ds.frames():
        if f.gt_depth is None:
            continue
        fusion.integrate(f.rgb, src.get_depth(f), f.pose_c2w)
        n += 1
    mesh = clean_mesh(fusion.extract_mesh(), min_cluster_triangles=500)
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"{ds.scene_id}: fused {n} Faro frames @ {args.voxel*100:.0f} cm -> {out} "
          f"({len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    main()
