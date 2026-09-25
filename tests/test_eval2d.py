"""ADR-014: the 2D-only evaluator must give the very numbers a full pipeline run writes."""

import json

import pytest

pytest.importorskip("open3d")

from roomscan.config import load_config  # noqa: E402
from roomscan.dataio.synthetic import write_synthetic_scene  # noqa: E402
from roomscan.eval2d import reevaluate_runs, run_2d_experiment  # noqa: E402
from roomscan.pipeline import ReconstructionPipeline  # noqa: E402


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("e2d")
    write_synthetic_scene(r, "90000002", n_frames=12, hires=(512, 384))
    return r


def _lidar_cfg(root, out):
    return load_config("configs/depth/lidar.yaml", [
        f"dataset.root={root}", "dataset.scene=90000002", "dataset.options.reference=arkit_mesh",
        "depth.lidar_min_confidence=1", f"output.root={out}", "output.experiment=t",
        "eval.n_sample_points=20000"])


def test_reeval_reproduces_pipeline_metrics_2d(root, tmp_path):
    res = ReconstructionPipeline(_lidar_cfg(root, tmp_path)).run()
    mf = res.out_dir / "metrics.json"
    before = json.loads(mf.read_text())
    assert before["metrics_2d"]["protocol_2d"] == 2
    assert reevaluate_runs([res.out_dir]) == 0                 # already at the current protocol
    assert reevaluate_runs([res.out_dir], force=True, log=lambda s: None) == 1
    after = json.loads(mf.read_text())
    assert after["metrics_2d"] == pytest.approx(before["metrics_2d"])
    assert after["metrics_3d"] == before["metrics_3d"]         # 3D untouched


def test_eval2d_experiment_matches_pipeline(root, tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment: t2d\nbase: configs/base.yaml\nscenes: ['90000002']\nruns:\n"
        "  - name: lidar\n    overrides: {depth.source: lidar, depth.aligner: identity, "
        "depth.lidar_min_confidence: 1}\n")
    run_2d_experiment(exp, overrides=[f"dataset.root={root}", f"output.root={tmp_path}"], log=lambda s: None)
    got = json.loads((tmp_path / "t2d" / "90000002_lidar" / "metrics.json").read_text())
    assert got["metrics_3d"] is None and got["n_frames"] == 12
    ref = ReconstructionPipeline(_lidar_cfg(root, tmp_path / "ref")).run()
    assert got["metrics_2d"]["abs_rel"] == pytest.approx(ref.metrics_2d.abs_rel)
    assert got["metrics_2d"]["n_valid"] == ref.metrics_2d.n_valid
