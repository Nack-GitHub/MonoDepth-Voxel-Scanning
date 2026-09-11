"""Phase 0 gate: one frame -> point cloud -> does it look like a room?

    python scripts/sanity_check.py --config configs/base.yaml --frame 0 [--source gt|lidar]

Writes outputs/sanity_<scene>_<frame>_<source>.ply. Open it in MeshLab. If it is
not obviously a room, the loader / intrinsics / pose convention is wrong — fix
that before touching TSDF or any model.

Run it twice with --frame 0 and --frame 50 and overlay both clouds: if they don't
line up, the .traj pose direction (c2w vs w2c) is inverted (ADR-009 VERIFY list).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roomscan.config import load_config
from roomscan.dataio import build_dataset
from roomscan.depth_sources import build_depth_source
from roomscan.geometry.backproject import backproject, to_world


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--source", default="gt", choices=["gt", "lidar"])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides + [f"depth.source={args.source}"])
    ds = build_dataset(cfg.dataset)
    f = ds.frame(args.frame)
    assert f.has_valid_pose, f"frame {args.frame} has no valid pose; try another"
    depth = build_depth_source(cfg.depth).get_depth(f)

    pts_cam = backproject(depth, ds.intrinsics)
    pts_world = to_world(pts_cam, f.pose_c2w)
    print(f"{len(pts_world)} points; depth range "
          f"{np.nanmin(depth[depth > 0]):.2f}-{np.nanmax(depth):.2f} m")

    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
    valid = np.isfinite(depth) & (depth > 0)
    pcd.colors = o3d.utility.Vector3dVector(f.rgb[valid].reshape(-1, 3) / 255.0)
    out = Path(args.out) / f"sanity_{ds.scene_id}_{args.frame:06d}_{args.source}.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
