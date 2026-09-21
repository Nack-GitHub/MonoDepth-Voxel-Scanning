"""Fine-tuning data/transform/loss wiring on a synthetic ARKitScenes scene (ADR-013).
Skipped as a whole when torch is missing, so `make test` stays green in a no-torch env."""

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open3d")   # only the synthetic-scene writer needs it; training/ never imports it

from roomscan.dataio.synthetic import write_synthetic_scene  # noqa: E402
from roomscan.training.dataset import DepthPairDataset  # noqa: E402
from roomscan.training.loss import DepthLoss, l1, silog  # noqa: E402
from roomscan.training.pairs import iter_indices, make_pair, open_training_scene  # noqa: E402
from roomscan.training.transforms import (  # noqa: E402
    EvalTransform,
    TrainTransform,
    color_jitter,
    hflip,
    random_crop,
    resize_short,
    upright,
)

N_FRAMES = 6


@pytest.fixture(scope="module")
def scene_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("arkit_train")
    for vid, seed in (("90000001", 0), ("90000002", 1)):
        write_synthetic_scene(root, vid, n_frames=N_FRAMES, hires=(512, 384), write_color=False, seed=seed)
    return root


@pytest.fixture(scope="module")
def faro_scene(scene_root):
    return open_training_scene(scene_root, "90000001", target="faro")


@pytest.fixture(scope="module")
def lidar_scene(scene_root):
    return open_training_scene(scene_root, "90000001", target="lidar")


def _src_depth(rng, h=480, w=640):
    """Depth with few distinct values so 'no new values' is a meaningful check."""
    d = rng.choice(np.array([0.0, 1.25, 2.5, 3.75, 7.0], dtype=np.float32), size=(h, w))
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return rgb, d, d > 0


# ---------------------------------------------------------------- pairs
def test_pair_faro_shapes_and_units(faro_scene):
    p = make_pair(faro_scene, 0, target="faro", max_depth=10.0)
    assert p.rgb.shape == (480, 640, 3) and p.rgb.dtype == np.uint8
    assert p.depth.shape == (480, 640) and p.depth.dtype == np.float32
    assert p.mask.shape == (480, 640) and p.mask.dtype == bool
    assert 1.0 <= float(np.median(p.depth[p.mask])) <= 5.0      # metres, not millimetres
    assert np.all(p.depth[~p.mask] == 0)
    assert p.sky_direction == "Up"


def test_pair_lidar_mask_uses_confidence(lidar_scene):
    f = lidar_scene.frame(0)
    conf = f.extra["lidar_confidence"]
    assert conf.shape == (480, 640)
    p = make_pair(lidar_scene, 0, target="lidar", lidar_min_confidence=2)
    assert not p.mask[conf < 2].any()
    assert np.all(p.depth[conf < 2] == 0)
    assert p.mask.sum() > 0
    loose = make_pair(lidar_scene, 0, target="lidar", lidar_min_confidence=0)
    assert loose.mask.sum() > p.mask.sum()


def test_pair_masks_beyond_max_depth(faro_scene):
    p = make_pair(faro_scene, 0, target="faro", max_depth=2.0)
    assert p.depth.max() <= 2.0
    assert p.mask.sum() > 0


def test_pair_rejects_unknown_target(faro_scene):
    with pytest.raises(ValueError):
        make_pair(faro_scene, 0, target="sonar")


def test_iter_indices_stride(lidar_scene):
    assert iter_indices(lidar_scene, 1) == list(range(len(lidar_scene)))
    assert iter_indices(lidar_scene, 4) == list(range(0, len(lidar_scene), 4))


# ---------------------------------------------------------------- transforms
def test_transform_keeps_depth_nearest():
    rgb, d, m = _src_depth(np.random.default_rng(0))
    _, d2, m2 = resize_short(rgb, d, m, 518)
    assert d2.shape == (518, 686)
    assert d2.shape[0] % 14 == 0 and d2.shape[1] % 14 == 0
    assert set(np.unique(d2)) <= set(np.unique(d))
    assert np.array_equal(m2, d2 > 0)


def test_eval_transform_shape(faro_scene):
    out = EvalTransform(518)(make_pair(faro_scene, 0, target="faro"))
    assert out["image"].shape == (3, 518, 686) and out["image"].dtype == np.float32
    assert out["depth"].shape == (518, 686) and out["mask"].shape == (518, 686)


def test_upright_roundtrip_rotates_all_three():
    rgb, d, m = _src_depth(np.random.default_rng(1), 48, 64)
    r2, d2, m2 = upright(rgb, d, m, "Left")
    assert r2.shape[:2] == d2.shape == m2.shape == (64, 48)
    r3, d3, m3 = upright(r2, d2, m2, 1)       # Left = clockwise (k=3); one more CCW turn undoes it
    assert np.array_equal(r3, rgb) and np.array_equal(d3, d) and np.array_equal(m3, m)


def test_upright_left_is_clockwise():
    """sky=Left: the image's left edge is the sky -> turn clockwise so it ends up on top (checked on
    real scenes against gravity from the poses; see transforms._ROT)."""
    img = np.zeros((2, 3), dtype=np.float32)
    img[:, 0] = 1.0                            # mark the left column
    _, up, _ = upright(np.zeros((2, 3, 3), np.uint8), img, img > 0, "Left")
    assert up.shape == (3, 2) and np.all(up[0] == 1.0) and up[1:].sum() == 0
    _, up_r, _ = upright(np.zeros((2, 3, 3), np.uint8), img[:, ::-1].copy(), img[:, ::-1] > 0, "Right")
    assert np.all(up_r[0] == 1.0)


def test_no_scale_augmentation():
    rng = np.random.default_rng(2)
    rgb, d, m = _src_depth(rng, 518, 686)
    rc, dc, mc = random_crop(rgb, d, m, 518, rng)
    assert dc.shape == (518, 518)
    # a crop is a contiguous window of the source: its values are the source's, unchanged
    found = any(np.array_equal(dc, d[:, x:x + 518]) for x in range(686 - 518 + 1))
    assert found
    rf, df, mf = hflip(rc, dc, mc, p=1.0, rng=rng)
    assert np.array_equal(np.sort(df, axis=None), np.sort(dc, axis=None))
    assert np.array_equal(df[:, ::-1], dc)


def test_train_transform_output_and_depth_values():
    rng = np.random.default_rng(3)
    rgb, d, m = _src_depth(rng)
    from roomscan.training.pairs import DepthPair

    out = TrainTransform(518)(DepthPair(rgb, d, m, "Right"), rng)
    assert out["image"].shape == (3, 518, 518)
    assert out["depth"].shape == out["mask"].shape == (518, 518)
    assert set(np.unique(out["depth"])) <= set(np.unique(d))


def test_color_jitter_changes_rgb_only():
    rng = np.random.default_rng(4)
    rgb, _, _ = _src_depth(rng, 32, 32)
    j = color_jitter(rgb, 0.4, np.random.default_rng(5))
    assert j.shape == rgb.shape and j.dtype == np.uint8
    assert not np.array_equal(j, rgb)


# ---------------------------------------------------------------- loss
def _gt(seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(2, 16, 16, generator=g) * 4 + 0.5
    return gt, torch.ones_like(gt, dtype=torch.bool)


def test_silog_zero_on_perfect_prediction():
    gt, m = _gt()
    assert abs(float(silog(gt, gt, m))) < 1e-6


def test_silog_penalises_scale():
    gt, m = _gt()
    a, b = float(silog(2 * gt, gt, m)), float(silog(1.1 * gt, gt, m))
    assert a > b > 0


def test_silog_gradient_is_finite():
    gt, m = _gt()
    pred = (gt * 1.3).clone().requires_grad_(True)
    silog(pred, gt, m).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    pred2 = gt.clone().requires_grad_(True)          # perfect prediction: sqrt at 0 must not NaN
    silog(pred2, gt, m).backward()
    assert torch.isfinite(pred2.grad).all()


def test_loss_empty_mask_is_zero_not_nan():
    gt, _ = _gt()
    pred = gt.clone().requires_grad_(True)
    empty = torch.zeros_like(gt, dtype=torch.bool)
    for loss in (silog(pred, gt, empty), l1(pred, gt, empty), DepthLoss(lambda_l1=1.0)(pred, gt, empty)):
        assert float(loss) == 0.0 and not torch.isnan(loss)
        loss.backward()


def test_depth_loss_adds_l1():
    gt, m = _gt()
    pred = gt * 1.2
    assert float(DepthLoss(lambda_l1=1.0)(pred, gt, m)) > float(DepthLoss()(pred, gt, m))


# ---------------------------------------------------------------- dataset
def test_dataset_concatenates_scenes(scene_root):
    scenes = [open_training_scene(scene_root, v, target="lidar") for v in ("90000001", "90000002")]
    ds = DepthPairDataset(scenes, target="lidar", transform=TrainTransform(518), stride=2)
    assert len(ds) == sum(len(iter_indices(s, 2)) for s in scenes)
    assert ds.frames_per_scene() == {"90000001": 3, "90000002": 3}
    item = ds[len(ds) - 1]
    assert item["image"].dtype == torch.float32 and item["image"].shape == (3, 518, 518)
    assert item["depth"].shape == (518, 518) and item["mask"].dtype == torch.bool
    assert item["scene"] == "90000002"


def test_training_package_respects_dependency_rule():
    code = ("import sys, roomscan.training.pairs, roomscan.training.transforms, roomscan.training.loss, "
            "roomscan.training.dataset\n"
            "bad = [m for m in ('open3d', 'roomscan.pipeline', 'roomscan.geometry', 'roomscan.depth_sources') "
            "if m in sys.modules]\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------- validate / train / registry
SMALL_METRIC = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class _EchoModel(torch.nn.Module):
    """Returns the Faro label of each validation frame in order -> a perfect prediction."""

    def __init__(self, gts):
        super().__init__()
        self.gts = list(gts)

    def forward(self, pixel_values):
        from types import SimpleNamespace

        return SimpleNamespace(predicted_depth=torch.from_numpy(self.gts.pop(0))[None])


def test_validate_perfect_prediction_scores_zero(faro_scene):
    from roomscan.training.validate import validate

    idx = iter_indices(faro_scene, 5)
    gts = [make_pair(faro_scene, i, target="faro").depth for i in idx]
    res = validate(_EchoModel(gts), [faro_scene], device="cpu", stride=5, size=266)
    assert res["val/n_frames"] == len(idx)
    assert res["val/abs_rel"] == pytest.approx(0.0, abs=1e-7)
    assert res["val/delta1"] == pytest.approx(1.0)
    assert np.isfinite(res["val/rmse"])


def _write_splits(path, root):
    import yaml

    scenes = [{"video_id": v, "visit_id": f"v{v}", "fold": "Training", "has_faro": True, "sky_direction": "Up"}
              for v in ("90000001", "90000002")]
    path.write_text(yaml.safe_dump({"seed": 0, "test": [], "val": scenes[1:], "train_faro": scenes[:1],
                                    "train_lidar": []}))
    return path


def _load_small():
    from transformers import AutoModelForDepthEstimation

    try:
        return AutoModelForDepthEstimation.from_pretrained(SMALL_METRIC)
    except OSError as e:   # offline / HF unreachable
        pytest.skip(f"cannot load {SMALL_METRIC}: {e}")


@pytest.fixture(scope="module")
def two_step_run(scene_root, tmp_path_factory):
    import json

    from roomscan.config import load_config
    from roomscan.training.train import train

    before = _load_small().state_dict()
    tmp = tmp_path_factory.mktemp("run")
    splits = _write_splits(tmp / "splits.yaml", scene_root)
    cfg = load_config("configs/training/r0_smoke.yaml", [
        f"output_dir={tmp}", f"data.root={scene_root}", f"data.splits={splits}", f"model.hf_id={SMALL_METRIC}",
        "train.max_steps=2", "train.batch_size=1", "train.grad_accum=1", "train.device=cpu",
        "data.num_workers=0", "transform.size=266", "val.every=2", "optim.warmup_steps=1", "optim.lr=1e-3"])
    run_dir = train(cfg)
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    return run_dir, rows, before


def test_train_two_steps_tiny_model(two_step_run):
    run_dir, rows, _ = two_step_run
    types = [r["type"] for r in rows]
    assert types[0] == "env" and types[-1] == "summary"
    assert [r["step"] for r in rows if r["type"] == "train"] == [1, 2]
    vals = [r for r in rows if r["type"] == "val"]
    assert vals[0]["step"] == 0 and types.index("val") < types.index("train")      # baseline before step 1
    assert all(np.isfinite(r["loss"]) for r in rows if r["type"] == "train")
    assert rows[0]["n_train_frames"] == N_FRAMES and rows[0]["n_val_scenes"] == 1
    assert (run_dir / "config.yaml").is_file() and (run_dir / "last" / "config.json").is_file()


def test_train_freezes_backbone_and_updates_head(two_step_run):
    from transformers import AutoModelForDepthEstimation

    run_dir, _, before = two_step_run
    after = AutoModelForDepthEstimation.from_pretrained(run_dir / "last").state_dict()
    bb = [k for k in before if k.startswith("backbone.")]
    assert bb and all(torch.equal(before[k], after[k]) for k in bb)
    head = [k for k in before if k.startswith(("head.", "neck."))]
    assert any(not torch.equal(before[k], after[k]) for k in head)


def test_train_refuses_to_reuse_run_dir(two_step_run):
    from roomscan.config import load_config
    from roomscan.training.train import train

    run_dir, _, _ = two_step_run
    cfg = load_config("configs/training/r0_smoke.yaml", [f"output_dir={run_dir.parent}"])
    with pytest.raises(FileExistsError):
        train(cfg)


def test_registry_ft_entries_build(two_step_run, monkeypatch):
    from roomscan.models import _REGISTRY, build_depth_model

    run_dir, _, _ = two_step_run
    for name in ("depth_anything_v2_ft_faro", "depth_anything_v2_ft_lidar", "depth_anything_v2_ft_lidar_all"):
        assert name in _REGISTRY
    cls, kwargs = _REGISTRY["depth_anything_v2_ft_faro"]
    monkeypatch.setitem(_REGISTRY, "depth_anything_v2_ft_faro", (cls, {**kwargs, "hf_id": str(run_dir / "last")}))
    m = build_depth_model("depth_anything_v2_ft_faro", device="cpu")
    assert m.is_metric is True and m.output_kind == "depth"
    out = m.predict(np.full((120, 160, 3), 128, dtype=np.uint8))
    assert out.shape == (120, 160) and out.dtype == np.float32


def test_push_dry_run_lists_files_without_network(two_step_run, capsys, monkeypatch):
    from roomscan.training import push

    run_dir, _, _ = two_step_run
    monkeypatch.setattr(sys, "argv", ["push", "--run", str(run_dir), "--repo", "someone/x", "--which", "last",
                                      "--dry-run"])
    push.main()
    out = capsys.readouterr().out
    assert "model.safetensors" in out and "PRIVATE" in out


def test_lidar_frames_without_depth_are_skipped(tmp_path):
    """Real LiDAR streams have gaps: a vga_wide frame with no lowres_depth within tolerance is dropped."""
    write_synthetic_scene(tmp_path, "90000003", n_frames=N_FRAMES, hires=(256, 192), write_color=False)
    missing = sorted((tmp_path / "raw" / "Training" / "90000003" / "lowres_depth").glob("*.png"))[2]
    missing.unlink()
    scene = open_training_scene(tmp_path, "90000003", target="lidar")
    assert 2 not in iter_indices(scene, 1, "lidar") and len(iter_indices(scene, 1, "lidar")) == N_FRAMES - 1
    ds = DepthPairDataset([scene], target="lidar", transform=TrainTransform(266))
    assert len(ds) == N_FRAMES - 1 and ds.n_skipped == 1
    for i in range(len(ds)):
        ds[i]
