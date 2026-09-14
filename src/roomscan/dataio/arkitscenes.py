"""ARKitScenes scene loader. Format facts below were read from Apple's own code
(threedod/benchmark_scripts/utils/tenFpsDataLoader.py, DATA.md, raw/README.md):

  * .traj line: "ts rx ry rz tx ty tz" — axis-angle + translation of the
    WORLD->CAMERA transform; Apple inverts it to get camera->world. So do we.
  * .pincam: "width height fx fy cx cy" (one line). Per-frame, but we take the
    median over the scan and treat intrinsics as constant.
  * depth PNG = uint16 millimetres. confidence PNG = uint8 in {0,1,2}.
  * Timestamps live in filenames: <video_id>_<ts>.png with 3 decimals.
  * lowres_wide.traj is only ~10 Hz on real scans (observed: 878 rows / 88 s) while
    highres_depth frames fall between rows -> poses are INTERPOLATED (slerp + lerp)
    between the two bracketing rows when the gap is <= `pose_max_gap`; frames outside
    the trajectory's time span are dropped.
  * Assets and rates (raw dataset):
        lowres_wide / lowres_depth / confidence   256x192   60 FPS  (LiDAR stream)
        highres_depth                              1920x1440 ~10 FPS (Faro, subset of videos)
        vga_wide                                   640x480   30 FPS
    plus the upsampling dataset's  color           1920x1440 at highres_depth timestamps.
  * metadata.csv has `sky_direction` (Up/Down/Left/Right): frames are stored
    landscape even when the iPad was held portrait. Monocular models need
    upright input, so we pass it along in Frame.extra.

Expected layout (see data/README.md):
    <root>/raw/<split>/<video_id>/{lowres_wide,lowres_wide_intrinsics,lowres_depth,confidence,
                                   highres_depth,vga_wide,...}/  + lowres_wide.traj + <video_id>_3dod_mesh.ply
    <root>/upsampling/<split>/<video_id>/color/            (optional, hi-res RGB)
    <root>/raw/metadata.csv                                 (optional, sky_direction)

Frame mapping (ADR-009):
    rgb        = best RGB asset available at its NATIVE resolution
    gt_depth   = highres_depth resampled to fusion_resolution, or None
    pose_c2w   = inv(traj)
    extra      = lidar_depth / lidar_confidence (fusion res), timestamp, sky_direction
    intrinsics = lowres_wide .pincam scaled to fusion_resolution
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from roomscan.dataio.base import SceneDataset
from roomscan.types import Frame, Intrinsics

RGB_ASSET_PRIORITY = ("color", "vga_wide", "lowres_wide")   # best first


def parse_traj_line(line: str) -> tuple[float, np.ndarray]:
    """-> (timestamp, camera-to-world 4x4). Mirrors Apple's TrajStringToMatrix."""
    tok = line.split()
    if len(tok) != 7:
        raise ValueError(f"bad traj line: {line!r}")
    ts = float(tok[0])
    rvec = np.array([float(t) for t in tok[1:4]], dtype=np.float64)
    tvec = np.array([float(t) for t in tok[4:7]], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec
    return ts, np.linalg.inv(w2c)


def read_pincam(path: Path) -> Intrinsics:
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    return Intrinsics(float(fx), float(fy), float(cx), float(cy), int(round(w)), int(round(h)))


def _interp_pose(p0: np.ndarray, p1: np.ndarray, a: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([p0[:3, :3], p1[:3, :3]]))
    out = np.eye(4)
    out[:3, :3] = Slerp([0.0, 1.0], rots)(a).as_matrix()
    out[:3, 3] = (1 - a) * p0[:3, 3] + a * p1[:3, 3]
    return out


def _timestamp_of(path: Path) -> float:
    return float(path.stem.rsplit("_", 1)[1])


class _AssetIndex:
    """Sorted timestamp -> path index for one asset directory, with nearest lookup."""

    def __init__(self, directory: Path, suffix: str):
        files = sorted(directory.glob(f"*{suffix}")) if directory.is_dir() else []
        self.paths = files
        self.ts = np.array([_timestamp_of(p) for p in files], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.paths)

    def nearest(self, ts: float, tol: float) -> Path | None:
        if len(self.ts) == 0:
            return None
        i = int(np.searchsorted(self.ts, ts))
        best, best_dt = None, tol
        for j in (i - 1, i):
            if 0 <= j < len(self.ts):
                dt = abs(self.ts[j] - ts)
                if dt <= best_dt:
                    best, best_dt = self.paths[j], dt
        return best


class ARKitScenesScene(SceneDataset):
    REFERENCES = ("faro_fused", "arkit_mesh")

    def __init__(self, root: str | Path, scene_id: str, *, split: str = "Training",
                 reference: str = "faro_fused", fusion_resolution: tuple[int, int] = (256, 192),
                 frame_source: str = "auto", rgb_asset: str = "auto",
                 match_tolerance: float = 0.02, pose_max_gap: float = 0.25,
                 sky_direction: str | None = None):
        if reference not in self.REFERENCES:
            raise ValueError(f"reference must be one of {self.REFERENCES}")
        self.root = Path(root)
        self.scene_id = str(scene_id)
        self.split = split
        self.scene_dir = self.root / "raw" / split / self.scene_id
        self.upsampling_dir = self.root / "upsampling" / split / self.scene_id
        self.reference = reference
        self.fusion_resolution = (int(fusion_resolution[0]), int(fusion_resolution[1]))
        self.tol = float(match_tolerance)
        self.pose_max_gap = float(pose_max_gap)
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(f"scene dir not found: {self.scene_dir}")

        # --- per-asset indices ---
        self._idx = {
            name: _AssetIndex(self.scene_dir / name, ".png")
            for name in ("lowres_depth", "confidence", "lowres_wide", "highres_depth", "vga_wide")
        }
        self._idx["color"] = _AssetIndex(self.upsampling_dir / "color", ".png")
        self._idx["pincam"] = _AssetIndex(self.scene_dir / "lowres_wide_intrinsics", ".pincam")

        # --- which timestamps define "a frame" ---
        if frame_source == "auto":
            frame_source = "highres_depth" if len(self._idx["highres_depth"]) else "lowres_depth"
        if frame_source not in self._idx or len(self._idx[frame_source]) == 0:
            raise FileNotFoundError(f"frame_source '{frame_source}' has no files in {self.scene_dir}")
        self.frame_source = frame_source

        # --- poses ---
        traj = self.scene_dir / "lowres_wide.traj"
        if not traj.is_file():
            raise FileNotFoundError(f"missing {traj}")
        parsed = [parse_traj_line(ln) for ln in traj.read_text().splitlines() if ln.strip()]
        parsed.sort(key=lambda x: x[0])
        self._pose_ts = np.array([p[0] for p in parsed])
        self._poses = np.stack([p[1] for p in parsed]) if parsed else np.zeros((0, 4, 4))

        # keep frames whose pose can be interpolated (tracking-lost gaps simply aren't in the traj)
        self._timestamps = [
            ts for ts in self._idx[frame_source].ts if self._pose_at(ts) is not None
        ]

        # --- intrinsics: median pincam of the lowres_wide stream, scaled to the fusion grid ---
        pincams = [read_pincam(p) for p in self._idx["pincam"].paths[:: max(1, len(self._idx["pincam"]) // 50)]]
        if not pincams:
            raise FileNotFoundError(f"no .pincam files in {self.scene_dir / 'lowres_wide_intrinsics'}")
        med = lambda k: float(np.median([getattr(p, k) for p in pincams]))  # noqa: E731
        native = Intrinsics(med("fx"), med("fy"), med("cx"), med("cy"), pincams[0].width, pincams[0].height)
        self._intrinsics = native.scaled(*self.fusion_resolution)

        # --- RGB asset ---
        if rgb_asset == "auto":
            rgb_asset = next((a for a in RGB_ASSET_PRIORITY if len(self._idx[a])), None)
            if rgb_asset is None:
                raise FileNotFoundError(f"no RGB asset in {self.scene_dir}")
        self.rgb_asset = rgb_asset

        self.sky_direction = sky_direction or self._sky_direction_from_metadata() or "Up"

    # ------------------------------------------------------------------ SceneDataset
    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics

    def __len__(self) -> int:
        return len(self._timestamps)

    def frame(self, idx: int) -> Frame:
        ts = self._timestamps[idx]
        pose = self._pose_at(ts)
        rgb = self._read_rgb(ts)
        gt = self._read_depth("highres_depth", ts)
        lidar = self._read_depth("lowres_depth", ts)
        conf = self._read_u8("confidence", ts)
        extra = {"timestamp": ts, "sky_direction": self.sky_direction}
        if lidar is not None:
            extra["lidar_depth"] = lidar
        if conf is not None:
            extra["lidar_confidence"] = conf
        return Frame(idx=idx, rgb=rgb, pose_c2w=pose, gt_depth=gt, extra=extra)

    def gt_mesh(self):
        import open3d as o3d

        if self.reference == "faro_fused":
            path = self.reference_mesh_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not built yet — run: python scripts/build_reference_mesh.py "
                    f"--set dataset.scene={self.scene_id}")
        else:
            path = self.scene_dir / f"{self.scene_id}_3dod_mesh.ply"
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing (download raw asset 'mesh')")
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            raise ValueError(f"{path} has no triangles")
        return mesh

    # ------------------------------------------------------------------ extras
    @property
    def reference_mesh_path(self) -> Path:
        return self.scene_dir / "reference_mesh.ply"

    @property
    def timestamps(self) -> list[float]:
        return list(self._timestamps)

    def has_faro_depth(self) -> bool:
        return len(self._idx["highres_depth"]) > 0

    # ------------------------------------------------------------------ internals
    def _pose_at(self, ts: float) -> np.ndarray | None:
        """Camera-to-world at `ts`, interpolated between the bracketing traj rows."""
        n = len(self._pose_ts)
        if n == 0:
            return None
        i = int(np.searchsorted(self._pose_ts, ts))
        if i == 0 or i == n:                       # outside the trajectory: nearest end if very close
            j = 0 if i == 0 else n - 1
            return self._poses[j] if abs(self._pose_ts[j] - ts) <= self.tol else None
        t0, t1 = self._pose_ts[i - 1], self._pose_ts[i]
        if t1 - t0 > self.pose_max_gap:            # tracking gap: don't invent poses across it
            for j in (i - 1, i):
                if abs(self._pose_ts[j] - ts) <= self.tol:
                    return self._poses[j]
            return None
        a = float((ts - t0) / (t1 - t0)) if t1 > t0 else 0.0
        return _interp_pose(self._poses[i - 1], self._poses[i], a)

    def _read_rgb(self, ts: float) -> np.ndarray:
        p = self._idx[self.rgb_asset].nearest(ts, self.tol)
        if p is None:   # fall back down the priority list rather than failing the frame
            for a in RGB_ASSET_PRIORITY:
                p = self._idx[a].nearest(ts, self.tol)
                if p is not None:
                    break
        if p is None:
            raise FileNotFoundError(f"no RGB within {self.tol}s of t={ts:.3f}")
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_depth(self, asset: str, ts: float) -> np.ndarray | None:
        p = self._idx[asset].nearest(ts, self.tol)
        if p is None:
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError(f"{p}: expected uint16 depth PNG, got {None if raw is None else raw.dtype}")
        depth = raw.astype(np.float32) / 1000.0
        return self._to_fusion_grid(depth)

    def _read_u8(self, asset: str, ts: float) -> np.ndarray | None:
        p = self._idx[asset].nearest(ts, self.tol)
        if p is None:
            return None
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return self._to_fusion_grid(raw.astype(np.uint8))

    def _to_fusion_grid(self, img: np.ndarray) -> np.ndarray:
        w, h = self.fusion_resolution
        if img.shape[1] == w and img.shape[0] == h:
            return img
        # nearest: never interpolate across depth discontinuities or into 0 (=invalid)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    def _sky_direction_from_metadata(self) -> str | None:
        meta = self.root / "raw" / "metadata.csv"
        if not meta.is_file():
            return None
        with meta.open() as f:
            for row in csv.DictReader(f):
                if str(row.get("video_id", "")).split(".")[0] == self.scene_id:
                    return row.get("sky_direction") or None
        return None
