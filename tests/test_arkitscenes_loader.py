"""Loader correctness against a synthetic scene with known geometry."""

import numpy as np
import pytest

from roomscan.dataio.arkitscenes import ARKitScenesScene, parse_traj_line
from roomscan.dataio.synthetic import write_synthetic_scene
from roomscan.geometry.backproject import backproject, to_world

pytest.importorskip("open3d")


@pytest.fixture(scope="module")
def scene_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("arkit")
    write_synthetic_scene(root, "90000001", n_frames=6, hires=(512, 384))
    return root


@pytest.fixture(scope="module")
def ds(scene_root):
    return ARKitScenesScene(scene_root, "90000001")


def test_traj_line_roundtrip_is_c2w():
    # w2c = identity rotation, camera at world (1,2,3) => w2c translation = -(1,2,3)
    ts, c2w = parse_traj_line("5.000 0 0 0 -1 -2 -3")
    assert ts == 5.0
    np.testing.assert_allclose(c2w[:3, 3], [1, 2, 3])


def test_frames_have_poses_and_all_assets(ds):
    assert len(ds) == 6
    f = ds.frame(0)
    assert f.has_valid_pose
    assert f.gt_depth is not None and f.gt_depth.shape == (192, 256)
    assert f.extra["lidar_depth"].shape == (192, 256)
    assert f.extra["lidar_confidence"].dtype == np.uint8
    assert f.rgb.shape == (384, 512, 3)          # 'color' asset at native resolution
    assert f.extra["sky_direction"] == "Up"
    assert ds.intrinsics.width == 256 and ds.intrinsics.height == 192


def test_nearest_pose_ignores_jittered_neighbours(ds):
    """The traj has 60 FPS rows; the ones between frames are nudged. We must pick the exact one."""
    f0, f1 = ds.frame(0), ds.frame(1)
    assert not np.allclose(f0.pose_c2w, f1.pose_c2w)
    # orbit radius 0.8 -> camera centre norm in xy == 0.8 exactly for un-nudged poses
    for f in (f0, f1):
        assert abs(np.linalg.norm(f.pose_c2w[:2, 3]) - 0.8) < 1e-6


def test_backprojected_gt_lies_on_room_mesh(ds):
    """The decisive Phase-0 check: depth + intrinsics + pose must land on the true walls."""
    import open3d as o3d

    mesh = ds.gt_mesh() if ds.reference == "arkit_mesh" else o3d.io.read_triangle_mesh(
        str(ds.scene_dir / "90000001_3dod_mesh.ply"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for idx in (0, 3):
        f = ds.frame(idx)
        pts = to_world(backproject(f.gt_depth, ds.intrinsics), f.pose_c2w)
        d = scene.compute_distance(o3d.core.Tensor(pts, dtype=o3d.core.float32)).numpy()
        # nearest-neighbour downsample of a 512x384 render + mm quantisation: a few mm at most
        assert np.percentile(d, 95) < 0.02, f"frame {idx}: p95 dist {np.percentile(d, 95):.4f} m"


def test_lidar_differs_from_gt_but_is_close(ds):
    f = ds.frame(2)
    valid = (f.gt_depth > 0) & (f.extra["lidar_depth"] > 0)
    err = np.abs(f.gt_depth - f.extra["lidar_depth"])[valid]
    assert err.mean() > 0.002 and err.mean() < 0.1
