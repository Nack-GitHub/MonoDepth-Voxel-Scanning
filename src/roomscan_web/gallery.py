"""Read-only gallery of finished experiment runs, so the paper's meshes open without re-running anything.

    <root>/<exp>/<scene>_<run>/{config.yaml, metrics.json, mesh.ply}      root = experiments/results

Nothing is ever written under `root`. What the viewer needs beyond the run's own files — the room's
reference mesh, its display copy, the error-coloured mesh — is derived into a separate cache folder
(`outputs/web/cache/<exp>/...`). The reference is rebuilt from the run's own `config.yaml`
(`build_dataset(cfg.dataset).gt_mesh()`, as scripts/make_figures.py does), so no dataset path is guessed.
Delete the cache folder after rebuilding a reference mesh; it is not invalidated on its own.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from roomscan_web import errorcolor, refmesh
from roomscan_web._fs import is_fresh, replace_atomic

log = logging.getLogger(__name__)

DEFAULT_ROOT = "experiments/results"
DEFAULT_EXPS = ("exp1_depth_source", "exp6_finetune")
# Wiring checks on the synthetic room are never listed next to results, whatever the allowlist says.
SYNTHETIC_EXPS = frozenset({"exp0_synthetic_smoke"})

# run name -> the method's name in the paper, in the order the gallery table shows its columns.
RUN_LABELS = {
    "gt": "Faro depth (pipeline ceiling)",
    "lidar": "iPad LiDAR",
    "mono_metric": "DA-v2 Metric-Indoor (pretrained)",
    "mono_ft_lidar_all": "FT-LiDAR-24",
    "mono_ft_lidar": "FT-LiDAR-10",
    "mono_ft_faro": "FT-Faro-10",
    "mono_oracle": "DA-v2 Large + per-frame oracle scale (uses GT)",
    "mono_sparse": "DA-v2 Large + sparse (upper bound)",          # ADR-011: LiDAR-sampled VIO proxy
    "mono_scene": "DA-v2 Large + once-per-room scale",
    "mono_depth_pro": "Depth Pro",
}
_RUN_RE = re.compile(r"^(\d{8})_([a-z0-9_]+)$")
_LOCK = threading.Lock()
_buildable: set[str] = set()      # dataset blocks that were seen to load; failures are retried (data may appear)


def allowed_exps(exps: Sequence[str]) -> tuple[str, ...]:
    return tuple(e for e in (x.strip() for x in exps) if e and e not in SYNTHETIC_EXPS)


def resolve_run(root: Path, exps: Sequence[str], exp: str, run: str) -> Path:
    """The run folder for a gallery URL, or ValueError — never trust path parts from the client."""
    if exp not in allowed_exps(exps) or not _RUN_RE.match(run):
        raise ValueError(f"unknown gallery item {exp}/{run}")
    d = (root / exp / run).resolve()
    if d.parent != (root / exp).resolve() or not (d / "mesh.ply").is_file():
        raise ValueError(f"unknown gallery item {exp}/{run}")
    return d


def scene_of(run: str) -> str:
    """'47429736_mono_ft_lidar_all' -> '47429736' (call after resolve_run has checked the name)."""
    return run.split("_", 1)[0]


def list_runs(root: Path, exps: Sequence[str], cache_dir: Path) -> list[dict[str, Any]]:
    """Every run that has a mesh and metrics, labelled as in the paper, rooms first then table column order.

    A run present in several experiments (mono_metric is in Exp1 and Exp6) is listed once, from the
    experiment named last in `exps`.
    """
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for exp in allowed_exps(exps):
        exp_dir = root / exp
        if not exp_dir.is_dir():
            continue
        for d in sorted(exp_dir.iterdir()):
            m = _RUN_RE.match(d.name)
            if not m or not (d / "mesh.ply").is_file():
                continue
            try:
                metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            scene, run = m.group(1), m.group(2)
            items[(scene, run)] = {
                "id": f"{exp}/{d.name}", "exp": exp, "scene": scene, "run": run,
                "label": RUN_LABELS.get(run, run), "metrics": metrics, "up": "z",    # ARKitScenes worlds are Z-up
                "has_reference": _has_reference(d, cache_dir / exp / scene), "origin": "paper",
            }
    order = {run: i for i, run in enumerate(RUN_LABELS)}
    return sorted(items.values(), key=lambda it: (it["scene"], order.get(it["run"], len(order)), it["run"]))


def reference_for(run_dir: Path, cache_dir: Path) -> Path | None:
    """The reference mesh of the run's room as `cache_dir/reference.ply`, or None when it cannot be had.

    Built once per room from the dataset the run's config.yaml names; the usual reason for None is
    that the dataset (or its reference mesh) is not on this machine.
    """
    out = cache_dir / "reference.ply"
    if out.is_file():
        return out
    import open3d as o3d

    with _LOCK:
        if out.is_file():
            return out
        try:
            mesh = _build_dataset(_dataset_cfg(run_dir)).gt_mesh()
        except Exception as e:  # noqa: BLE001 — whatever stops the dataset from loading means "no reference"
            log.warning("no reference for %s: %s: %s", run_dir.name, type(e).__name__, e)
            return None
        if mesh is None or not mesh.has_triangles():
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / "reference.tmp.ply"
        o3d.io.write_triangle_mesh(str(tmp), mesh)
        replace_atomic(tmp, out)
    return out


def reference_view(run_dir: Path, exp_cache: Path) -> Path | None:
    """Display copy of the room's reference (what GET .../reference.ply serves); `exp_cache` = <cache>/<exp>."""
    ref = reference_for(run_dir, exp_cache / scene_of(run_dir.name))
    return refmesh.write_view(ref, ref.with_name("reference_view.ply")) if ref else None


def error_mesh(run_dir: Path, exp_cache: Path) -> Path | None:
    """The run's mesh coloured by distance to its room's reference (what GET .../error.ply serves)."""
    ref = reference_for(run_dir, exp_cache / scene_of(run_dir.name))
    if ref is None:
        return None
    return errorcolor.write_error_ply(run_dir / "mesh.ply", ref, exp_cache / run_dir.name / "error.ply")


def warm(root: Path, exps: Sequence[str], cache_dir: Path, scenes: Sequence[str] | None = None,
         ) -> Iterator[tuple[str, str]]:
    """Build everything the viewer will ask for, ahead of time: yields (item id, built | cached | no reference).

    Goes through the same functions as the endpoints, so a warmed cache is exactly what a first click makes.
    """
    for item in list_runs(root, exps, cache_dir):
        if scenes and item["scene"] not in scenes:
            continue
        run_dir, exp_cache = root / item["id"], cache_dir / item["exp"]
        ref = exp_cache / item["scene"] / "reference.ply"
        cached = (is_fresh(exp_cache / run_dir.name / "error.ply", run_dir / "mesh.ply", ref)
                  and is_fresh(ref.with_name("reference_view.ply"), ref))
        built = cached or (reference_view(run_dir, exp_cache) is not None
                           and error_mesh(run_dir, exp_cache) is not None)
        yield item["id"], "cached" if cached else "built" if built else "no reference"


def _dataset_cfg(run_dir: Path):
    from roomscan.config import load_config

    return load_config(run_dir / "config.yaml").dataset


def _build_dataset(cfg):
    from roomscan.dataio import build_dataset

    return build_dataset(cfg)


def _has_reference(run_dir: Path, cache_dir: Path) -> bool:
    """Cheap answer for the listing: the reference is cached, or the run's dataset loads on this machine.

    Loading the mesh itself is left to the first request that needs it (or scripts/warm_web_cache.py).
    """
    if (cache_dir / "reference.ply").is_file():
        return True
    try:
        cfg = _dataset_cfg(run_dir)
        key = str(cfg)                       # every run of a room shares one dataset block: build it once
        if key not in _buildable:
            _build_dataset(cfg)
            _buildable.add(key)
    except Exception:  # noqa: BLE001 — see reference_for
        return False
    return True
