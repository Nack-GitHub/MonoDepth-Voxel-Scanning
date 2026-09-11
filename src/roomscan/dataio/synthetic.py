"""Synthetic room written in the exact ARKitScenes raw/upsampling layout.

Used by tests and by scripts/make_synthetic_scene.py so that every stage —
loader, traj inversion, timestamp matching, TSDF, metrics — can be validated
end-to-end without downloading (or being licensed for) real scans. The
geometry is known exactly, so evaluation numbers have a known answer.

The room: a 5 x 4 x 2.6 m box with two furniture boxes, procedurally
textured. Cameras orbit the centre looking outward and slightly down.
`lowres_depth` gets LiDAR-like noise + dropouts so the `lidar` row differs
from `gt`; `highres_depth` is the exact render.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 256x192 native ARKit low-res grid and a plausible iPad wide-camera focal length at that scale.
LOWRES = (256, 192)
LOWRES_FX = 211.9
DEFAULT_HIRES = (1920, 1440)


def room_mesh(size=(5.0, 4.0, 2.6), furniture: bool = True):
    import open3d as o3d

    w, d, h = size
    room = o3d.geometry.TriangleMesh.create_box(w, d, h)
    room.translate((-w / 2, -d / 2, 0.0))
    mesh = room
    if furniture:
        table = o3d.geometry.TriangleMesh.create_box(1.2, 0.8, 0.75).translate((0.6, -1.6, 0.0))
        cabinet = o3d.geometry.TriangleMesh.create_box(0.6, 1.5, 1.8).translate((-2.5, 0.2, 0.0))
        mesh = room + table + cabinet
    mesh.compute_vertex_normals()
    return mesh


def look_at_c2w(position: np.ndarray, target: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) -> world, z-up world."""
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(world_up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, position
    return c2w


def orbit_poses(n: int, radius: float = 0.8, height: float = 1.4,
                pitches_deg=(-40.0, 0.0, 40.0)) -> list[np.ndarray]:
    """Rings at several pitches (floor / walls / ceiling) — a crude but complete scan path."""
    poses = []
    per = max(1, n // len(pitches_deg))
    for i in range(n):
        ring, k = divmod(i, per)
        pitch = pitches_deg[min(ring, len(pitches_deg) - 1)]
        a = 2 * np.pi * k / per + 0.25 * ring
        pos = np.array([radius * np.cos(a), radius * np.sin(a), height])
        fwd = np.array([np.cos(a), np.sin(a), np.tan(np.deg2rad(pitch))])
        poses.append(look_at_c2w(pos, pos + fwd))
    return poses


def _render(scene, K: np.ndarray, c2w: np.ndarray, width: int, height: int):
    """-> (depth [H,W] float32 metres, 0 = miss), (rgb [H,W,3] uint8)."""
    import open3d.core as o3c

    w2c = np.linalg.inv(c2w)
    rays = scene.create_rays_pinhole(o3c.Tensor(K, dtype=o3c.float64),
                                     o3c.Tensor(w2c, dtype=o3c.float64), width, height)
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    hit = np.isfinite(t)
    r = rays.numpy()
    pts_w = r[..., :3] + r[..., 3:] * t[..., None]
    pts_c = (w2c[:3, :3] @ pts_w.reshape(-1, 3).T).T + w2c[:3, 3]
    depth = np.where(hit, pts_c[:, 2].reshape(height, width), 0.0).astype(np.float32)

    # procedural texture: 0.5 m checkerboard in world space + per-face tint from the normal
    n = ans["primitive_normals"].numpy()
    cells = np.floor(pts_w / 0.5).sum(axis=-1)
    checker = (cells % 2 == 0).astype(np.float32) * 0.35 + 0.45
    tint = 0.5 + 0.5 * np.abs(n)          # walls facing x/y/z get different hues
    rgb = (checker[..., None] * tint * 255).clip(0, 255).astype(np.uint8)
    rgb[~hit] = 0
    return depth, rgb


def _w2c_traj_line(ts: float, c2w: np.ndarray) -> str:
    w2c = np.linalg.inv(c2w)
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    t = w2c[:3, 3]
    return f"{ts:.3f} {rvec[0,0]:.9f} {rvec[1,0]:.9f} {rvec[2,0]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f}"


def write_synthetic_scene(root: str | Path, video_id: str = "90000001", *, split: str = "Training",
                          n_frames: int = 24, hires: tuple[int, int] = DEFAULT_HIRES,
                          write_color: bool = True, lidar_noise: float = 0.01, seed: int = 0,
                          t0: float = 100.0, fps: float = 10.0) -> Path:
    """Write a scene and return its raw scene directory."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    root = Path(root)
    scene_dir = root / "raw" / split / video_id
    color_dir = root / "upsampling" / split / video_id / "color"
    for sub in ("lowres_wide", "lowres_wide_intrinsics", "lowres_depth", "confidence",
                "highres_depth", "vga_wide"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    if write_color:
        color_dir.mkdir(parents=True, exist_ok=True)

    mesh = room_mesh()
    o3d.io.write_triangle_mesh(str(scene_dir / f"{video_id}_3dod_mesh.ply"), mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    lw, lh = LOWRES
    K_low = np.array([[LOWRES_FX, 0, lw / 2 - 0.5], [0, LOWRES_FX, lh / 2 - 0.5], [0, 0, 1]])
    s_hi = hires[0] / lw
    K_hi = K_low.copy()
    K_hi[:2] *= s_hi
    s_vga = 640 / lw
    K_vga = K_low.copy()
    K_vga[:2] *= s_vga

    poses = orbit_poses(n_frames)
    traj_lines = []
    for i, c2w in enumerate(poses):
        ts = t0 + i / fps
        name = f"{video_id}_{ts:.3f}"

        d_hi, rgb_hi = _render(scene, K_hi, c2w, *hires)
        d_lo, rgb_lo = _render(scene, K_low, c2w, lw, lh)
        _, rgb_vga = _render(scene, K_vga, c2w, 640, 480)

        # LiDAR-like: noise grows with range, 2 % dropouts, confidence from noise magnitude
        noise = rng.normal(0.0, lidar_noise * (1.0 + d_lo), d_lo.shape).astype(np.float32)
        lidar = np.where(d_lo > 0, d_lo + noise, 0.0)
        lidar[rng.random(d_lo.shape) < 0.02] = 0.0
        conf = np.where(np.abs(noise) < lidar_noise, 2, np.where(np.abs(noise) < 2 * lidar_noise, 1, 0))
        conf = conf.astype(np.uint8)

        cv2.imwrite(str(scene_dir / "highres_depth" / f"{name}.png"), _mm(d_hi))
        cv2.imwrite(str(scene_dir / "lowres_depth" / f"{name}.png"), _mm(lidar))
        cv2.imwrite(str(scene_dir / "confidence" / f"{name}.png"), conf)
        cv2.imwrite(str(scene_dir / "lowres_wide" / f"{name}.png"), cv2.cvtColor(rgb_lo, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(scene_dir / "vga_wide" / f"{name}.png"), cv2.cvtColor(rgb_vga, cv2.COLOR_RGB2BGR))
        if write_color:
            cv2.imwrite(str(color_dir / f"{name}.png"), cv2.cvtColor(rgb_hi, cv2.COLOR_RGB2BGR))
        (scene_dir / "lowres_wide_intrinsics" / f"{name}.pincam").write_text(
            f"{lw} {lh} {K_low[0,0]:.6f} {K_low[1,1]:.6f} {K_low[0,2]:.6f} {K_low[1,2]:.6f}\n")

        # traj at 60 FPS around each frame, with sub-ms jitter (exercises nearest matching)
        for k in range(6):
            tj = ts + k / 60.0 + rng.uniform(-4e-4, 4e-4)
            traj_lines.append(_w2c_traj_line(tj, c2w if k == 0 else _nudge(c2w, k)))

    (scene_dir / "lowres_wide.traj").write_text("\n".join(traj_lines) + "\n")
    meta = root / "raw" / "metadata.csv"
    if not meta.is_file():
        meta.write_text("video_id,visit_id,sky_direction,fold,has_laser_scanner_point_clouds,"
                        "is_in_upsampling,is_in_threedod\n")
    with meta.open("a") as f:
        f.write(f"{video_id},NA,Up,{split},False,True,False\n")
    return scene_dir


def _nudge(c2w: np.ndarray, k: int) -> np.ndarray:
    """Slightly different pose for the in-between 60 FPS traj rows (they must NOT be picked)."""
    out = c2w.copy()
    out[:3, 3] += np.array([0.05, 0.05, 0.0]) * k
    return out


def _mm(depth_m: np.ndarray) -> np.ndarray:
    return np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
