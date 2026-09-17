"""Export a scene as the app's capture folder (+ zip) for roomscan_web / dataio/custom.

    python scripts/export_capture.py --scene 42444474 --stride 5 --out outputs/captures
    python scripts/export_capture.py --root data/synthetic --scene 90000001 --out outputs/captures

Writes <out>/<scene>/{intrinsics.json, poses.json, rgb/, depth/, confidence/, sparse/, reference.ply}
and <out>/<scene>.zip ready for `POST /scans`. `depth/` is the ARKit LiDAR frame (metres -> uint16 mm),
`sparse/` is the VIO proxy (ADR-011) when --sparse-points > 0, `reference.ply` the Faro reference mesh
(or the ARKit 3DOD mesh with --reference arkit_mesh) so the web job also reports metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from roomscan.dataio.arkitscenes import ARKitScenesScene


def _mm(depth_m: np.ndarray) -> np.ndarray:
    return np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/arkitscenes")
    ap.add_argument("--scene", default="42444474")
    ap.add_argument("--out", default="outputs/captures")
    ap.add_argument("--stride", type=int, default=5, help="keep every k-th Faro frame (5 ≈ 0.8 fps)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sparse-points", type=int, default=200, help="0 = no sparse/ folder")
    ap.add_argument("--reference", default="faro_fused", choices=["faro_fused", "arkit_mesh", "none"])
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    ds = ARKitScenesScene(args.root, args.scene, sparse_points=args.sparse_points,
                          reference="arkit_mesh" if args.reference == "arkit_mesh" else "faro_fused")
    out = Path(args.out) / args.scene
    if out.exists():
        shutil.rmtree(out)
    for sub in ("rgb", "depth", "confidence") + (("sparse",) if args.sparse_points else ()):
        (out / sub).mkdir(parents=True)

    poses, n = {}, 0
    intr = None
    for f in ds.frames(stride=args.stride, max_frames=args.max_frames):
        stem = f"{f.idx:06d}"
        cv2.imwrite(str(out / "rgb" / f"{stem}.jpg"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if "lidar_depth" in f.extra:
            cv2.imwrite(str(out / "depth" / f"{stem}.png"), _mm(f.extra["lidar_depth"]))
        if "lidar_confidence" in f.extra:
            cv2.imwrite(str(out / "confidence" / f"{stem}.png"), f.extra["lidar_confidence"])
        if "sparse_depth" in f.extra:
            cv2.imwrite(str(out / "sparse" / f"{stem}.png"), _mm(f.extra["sparse_depth"]))
        poses[stem] = f.pose_c2w.tolist()
        if intr is None:
            intr = ds.intrinsics.scaled(f.rgb.shape[1], f.rgb.shape[0])   # intrinsics at rgb resolution
        n += 1
    if n == 0:
        raise SystemExit("no frames exported")
    (out / "poses.json").write_text(json.dumps(poses))
    (out / "intrinsics.json").write_text(json.dumps(
        {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "width": intr.width, "height": intr.height}))

    if args.reference != "none":
        src = ds.reference_mesh_path if args.reference == "faro_fused" else ds.scene_dir / f"{args.scene}_3dod_mesh.ply"
        if src.is_file():
            shutil.copy(src, out / "reference.ply")
        else:
            print(f"warning: no reference mesh at {src}; job will run without metrics")

    print(f"wrote {out}: {n} frames, rgb {intr.width}x{intr.height}, sparse={'yes' if args.sparse_points else 'no'}")
    if not args.no_zip:
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
        print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB) -> upload it at http://localhost:8765")


if __name__ == "__main__":
    main()
