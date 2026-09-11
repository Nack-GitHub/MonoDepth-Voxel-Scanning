"""Aggregate experiments/results/**/metrics.json into the paper's tables (Phase 3-4).

    roomscan report experiments/results
    -> experiments/results/summary.csv            (one row per run)
    -> experiments/results/<experiment>/table.md  (mean +/- std over scenes)
"""

from __future__ import annotations

from pathlib import Path


def collect_runs(results_root: str | Path):
    """Walk <root>/<experiment>/<run>/metrics.json -> pandas.DataFrame."""
    raise NotImplementedError("Phase 3")


def write_tables(results_root: str | Path) -> None:
    raise NotImplementedError("Phase 3")
