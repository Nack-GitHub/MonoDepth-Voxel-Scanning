"""YAML config loading with `_base_` inheritance and CLI dotlist overrides.

    cfg = load_config("configs/depth/gt.yaml", ["fusion.voxel_size=0.02"])

Deliberately thin: OmegaConf gives us merge + dotlist + interpolation; we
don't need Hydra's launcher/plugins for a 3-scene offline study (ADR-005).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

BASE_KEY = "_base_"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = _load_with_base(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.pop(BASE_KEY, None)
    return cfg


def _load_with_base(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    base = cfg.get(BASE_KEY)
    if base is None:
        return cfg
    base_path = Path(base) if Path(base).is_absolute() else _resolve_relative(path, base)
    return OmegaConf.merge(_load_with_base(base_path), cfg)


def _resolve_relative(child: Path, base: str) -> Path:
    # `_base_` paths are written relative to the repo root (so configs are
    # copy-pasteable); fall back to relative-to-file for ad-hoc configs.
    root_rel = Path(base)
    return root_rel if root_rel.exists() else child.parent / base


def apply_overrides(cfg: DictConfig, overrides: dict[str, object]) -> DictConfig:
    """Merge a flat {'depth.source': 'mono'} dict — the shape used by sweep files."""
    dotlist = [f"{k}={v}" for k, v in overrides.items()]
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))


def to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
