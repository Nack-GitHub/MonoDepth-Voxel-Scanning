"""Phase 1: build the evaluation reference for an ARKitScenes scan (ADR-009).

    python scripts/build_reference_mesh.py --config configs/base.yaml --set dataset.scene=41069021

Fuses Faro `highres_depth` with the dataset poses over ALL frames at a fine voxel
(default 1 cm) and writes <scene>/reference_mesh.ply. Every run's metrics_3d is
measured against this file. Build it once per scene; never rebuild mid-study.

Note the tautology guard: the `gt` experiment row must run at the *experiment's*
voxel size / stride (4 cm / 10), not at this reference's (1 cm / 1) — its
non-zero Chamfer is exactly the pipeline's own discretisation loss.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # TODO(Phase 1):
    #   cfg = load_config(args.config, args.overrides + [
    #       "depth.source=gt", "dataset.frame_stride=1",
    #       f"fusion.voxel_size={args.voxel}", f"fusion.sdf_trunc={3 * args.voxel}"])
    #   ds = build_dataset(cfg.dataset); fusion = TSDFFusion(...); loop ds.frames();
    #   extract; clean; o3d.io.write_triangle_mesh(ds.root / "reference_mesh.ply", mesh)
    raise NotImplementedError(f"Phase 1 (config={args.config}, voxel={args.voxel})")


if __name__ == "__main__":
    main()
