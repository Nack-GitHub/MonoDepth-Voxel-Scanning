from roomscan.config import load_config


def test_base_inheritance_and_override():
    cfg = load_config("configs/depth/mono_oracle.yaml", ["fusion.voxel_size=0.02"])
    assert cfg.depth.source == "mono"            # from child
    assert cfg.depth.aligner == "oracle_frame"   # from child
    assert cfg.fusion.sdf_trunc == 0.12          # inherited from base
    assert cfg.fusion.voxel_size == 0.02         # CLI override wins
    assert "_base_" not in cfg


def test_every_experiment_file_resolves():
    from pathlib import Path

    from omegaconf import OmegaConf

    from roomscan.config import apply_overrides

    for f in Path("configs/experiments").glob("*.yaml"):
        exp = OmegaConf.load(f)
        base = load_config(exp.base)
        for run in exp.runs:
            cfg = apply_overrides(base, dict(run.overrides))
            assert cfg.depth.source in {"gt", "mono", "arcore", "lidar"}, f
