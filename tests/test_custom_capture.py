"""dataio/custom.py: a capture folder written by the app runs through the same pipeline."""

import json

import numpy as np
import pytest

pytest.importorskip("open3d")

from roomscan.config import load_config  # noqa: E402
from roomscan.dataio.custom import CustomCaptureScene  # noqa: E402
from roomscan.dataio.synthetic import write_synthetic_scene  # noqa: E402
from roomscan.pipeline import ReconstructionPipeline  # noqa: E402


def _convert_synthetic_to_capture(root, out, video_id="90000001", n=12):
    """Emulate the app: rgb + LiDAR depth + confidence + poses.json + intrinsics.json."""
    import shutil

    from roomscan.dataio.arkitscenes import ARKitScenesScene

    src = ARKitScenesScene(root, video_id, reference="arkit_mesh", fusion_resolution=(256, 192))
    scene = out / "room1"
    for sub in ("rgb", "depth", "confidence", "sparse"):
        (scene / sub).mkdir(parents=True)
    poses = {}
    import cv2

    for i, f in enumerate(src.frames(max_frames=n)):
        stem = f"{i:05d}"
        cv2.imwrite(str(scene / "rgb" / f"{stem}.png"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(scene / "depth" / f"{stem}.png"), (f.extra["lidar_depth"] * 1000).astype(np.uint16))
        cv2.imwrite(str(scene / "confidence" / f"{stem}.png"), f.extra["lidar_confidence"])
        poses[stem] = f.pose_c2w.tolist()
    intr = src.intrinsics.scaled(f.rgb.shape[1], f.rgb.shape[0])
    (scene / "intrinsics.json").write_text(json.dumps(
        {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "width": intr.width, "height": intr.height}))
    (scene / "poses.json").write_text(json.dumps(poses))
    shutil.copy(src.scene_dir / f"{video_id}_3dod_mesh.ply", scene / "reference.ply")
    return out


def test_custom_capture_runs_lidar_row(tmp_path):
    write_synthetic_scene(tmp_path / "syn", n_frames=12, hires=(512, 384))
    root = _convert_synthetic_to_capture(tmp_path / "syn", tmp_path / "captures")
    ds = CustomCaptureScene(root, "room1")
    assert len(ds) == 12 and ds.intrinsics.width == 256
    f = ds.frame(0)
    assert f.gt_depth is None and f.extra["lidar_depth"].shape == (192, 256)

    cfg = load_config("configs/depth/lidar.yaml", [
        "dataset.name=custom", f"dataset.root={root}", "dataset.scene=room1",
        f"output.root={tmp_path}", "output.experiment=t", "eval.n_sample_points=20000",
        "eval.mask_to_gt=false",
    ])
    res = ReconstructionPipeline(cfg).run()
    assert res.n_frames == 12 and res.metrics_2d is None        # no GT depth on a capture
    assert res.metrics_3d is not None and res.metrics_3d.accuracy < 0.08   # reference.ply was given
