"""Phase 1 gate on the synthetic room: GT depth -> TSDF -> mesh must match the true room."""

import numpy as np
import pytest

pytest.importorskip("open3d")

from roomscan.config import load_config  # noqa: E402
from roomscan.dataio.synthetic import write_synthetic_scene  # noqa: E402
from roomscan.evaluation.metrics_3d import evaluate_mesh  # noqa: E402
from roomscan.geometry.postprocess import clean_mesh  # noqa: E402
from roomscan.pipeline import ReconstructionPipeline  # noqa: E402


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("p1")
    write_synthetic_scene(r, "90000001", n_frames=18, hires=(512, 384))
    return r


def test_metrics_known_answer():
    import open3d as o3d

    a = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    b = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate((0.0, 0.0, 0.03))
    m = evaluate_mesh(a, b, thresholds=[0.02, 0.05], n_sample_points=20000)
    assert 0.005 < m.chamfer < 0.03            # 3 cm shift -> top/bottom faces off by 3 cm, sides ~0
    assert m.fscore[0.05] > 0.95 and m.fscore[0.02] < m.fscore[0.05]
    assert m.normal_consistency > 0.95


def test_clean_mesh_drops_floaters():
    import open3d as o3d

    big = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=20)
    tiny = o3d.geometry.TriangleMesh.create_sphere(0.05, resolution=3).translate((5, 5, 5))
    out = clean_mesh(big + tiny, min_cluster_triangles=100)
    assert len(out.triangles) == len(big.triangles)


def test_gt_pipeline_reconstructs_room(root, tmp_path):
    cfg = load_config("configs/depth/gt.yaml", [
        f"dataset.root={root}", "dataset.scene=90000001", "dataset.frame_stride=1",
        "dataset.options.reference=arkit_mesh",       # synthetic: the true room mesh
        "fusion.voxel_size=0.04", "fusion.sdf_trunc=0.12",
        f"output.root={tmp_path}", "output.experiment=t", "eval.n_sample_points=50000",
    ])
    res = ReconstructionPipeline(cfg).run()
    assert res.n_frames == 18
    m = res.metrics_3d
    assert m is not None
    assert m.accuracy < 0.05, m                    # what we built is on the walls (4 cm voxels)
    assert m.precision[0.05] > 0.85, m
    assert m.recall[0.10] > 0.6, m                 # hidden furniture faces cap recall at ~0.75
    assert (res.out_dir / "metrics.json").exists() and (res.out_dir / "mesh.ply").exists()
    assert np.isfinite(res.timing.total)
