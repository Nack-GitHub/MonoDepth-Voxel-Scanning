"""Table 4.1 helper: one row per downloaded ARKitScenes scene.

    python scripts/scene_info.py [--root data/arkitscenes] [--md]

Prints room extent (min-area footprint x height of the Faro reference mesh), scan duration from
lowres_wide.traj, Faro / VGA frame counts and sky_direction. Reads only files
that are already on disk; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def scene_row(scene_dir: Path, meta: dict[str, dict]) -> dict:
    import numpy as np
    import open3d as o3d

    sid = scene_dir.name
    lines = (scene_dir / "lowres_wide.traj").read_text().splitlines()
    traj = [float(ln.split()[0]) for ln in lines if ln.strip()]
    row = {"video_id": sid, "scan_s": round(traj[-1] - traj[0], 1), "traj_rows": len(traj),
           "faro_frames": len(list((scene_dir / "highres_depth").glob("*.png"))),
           "vga_frames": len(list((scene_dir / "vga_wide").glob("*.png"))),
           "sky": meta.get(sid, {}).get("sky_direction", "?")}
    mesh = scene_dir / "reference_mesh.ply"       # Faro-covered region only; the 3DOD mesh leaks into neighbours
    if mesh.is_file():
        m = o3d.io.read_triangle_mesh(str(mesh))
        v = np.asarray(m.vertices)
        rng = v.max(0) - v.min(0)                   # reference is already cluster-filtered
        up = int(np.argmin(rng))                    # gravity axis = smallest extent (room height)
        h = rng[up]
        flat = v[:, [i for i in range(3) if i != up]]
        # min-area rectangle of the footprint: brute-force the yaw (2D PCA is unreliable for L-shaped rooms)
        best = None
        for deg in np.arange(0, 90, 0.5):
            c, s_ = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
            r = flat @ np.array([[c, -s_], [s_, c]])
            ext = r.max(0) - r.min(0)
            if best is None or ext.prod() < best.prod():
                best = ext
        ext = sorted(best, reverse=True)
        row["room_m"] = f"{ext[0]:.1f} × {ext[1]:.1f} × {h:.1f}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/arkitscenes")
    ap.add_argument("--split", default="Training")
    ap.add_argument("--md", action="store_true", help="markdown table instead of csv")
    args = ap.parse_args()
    root = Path(args.root)
    meta = {}
    mp = root / "raw" / "metadata.csv"
    if mp.is_file():
        meta = {r["video_id"]: r for r in csv.DictReader(mp.open())}
    rows = [scene_row(d, meta) for d in sorted((root / "raw" / args.split).iterdir())
            if (d / "lowres_wide.traj").is_file()]
    cols = ["video_id", "room_m", "scan_s", "traj_rows", "faro_frames", "vga_frames", "sky"]
    if args.md:
        print("| " + " | ".join(cols) + " |")
        print("|---" * len(cols) + "|")
        for r in rows:
            print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
