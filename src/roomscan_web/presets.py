"""Upload presets: the one place that maps a dropdown entry to config overrides and a paper label.

Labels are the method names used in the paper; internal names (run ids, registry keys) never reach the page.
`eta_s_per_frame` is a rough laptop figure (Exp1/Exp6 timing on 47429736: 504 frames, ~440 s with DA-v2 Large,
~3 s with LiDAR) — it only feeds the "expected" hint next to the running timer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    note: str
    eta_s_per_frame: float
    overrides: dict[str, str] = field(default_factory=dict)


def _mono(model: str, aligner: str) -> dict[str, str]:
    return {"depth.source": "mono", "depth.model": model, "depth.aligner": aligner}


# Dropdown order.
PRESETS: dict[str, Preset] = {p.key: p for p in (
    Preset("lidar", "iPad LiDAR (sensor)", "needs depth/ in zip", 0.01),
    Preset("ft_lidar_24", "FT-LiDAR-24 (ours, RGB only)", "~3–7 min/room", 0.9,
           _mono("depth_anything_v2_ft_lidar_all", "identity")),
    Preset("mono_metric", "DA-v2 Metric-Indoor (pretrained)", "~3–7 min/room", 0.9,
           _mono("depth_anything_v2_metric_indoor", "identity")),
    # ADR-011: the sparse points are sampled from LiDAR, so this row is an upper bound of real VIO.
    Preset("mono_sparse", "DA-v2 Large + sparse points (upper bound)",
           "LiDAR-sampled VIO proxy — upper bound; needs sparse/ in zip", 0.9,
           _mono("depth_anything_v2_large", "sparse_points")),
)}


def public_presets() -> list[dict]:
    return [asdict(p) for p in PRESETS.values()]


def resolve(key: str) -> Preset:
    """The preset for a client-supplied key, or ValueError."""
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(f"unknown preset {key!r}; known: {list(PRESETS)}") from None
