"""Error colours for the viewer: each vertex of a reconstruction painted by its distance to the Faro reference.

Same scale as the error figures of the paper (scripts/make_figures.py): turbo, 0 cm = dark blue,
>= 10 cm = dark red. Distances are vertex -> nearest of 200k points sampled on the reference, with the
seed `per_point_error` uses for the reference, so the page and the figures colour the same error.
The result is cached as a PLY next to where the caller keeps the run's derived files.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from roomscan.evaluation.metrics_3d import _nn, _sample
from roomscan_web._fs import is_fresh, replace_atomic

VMAX = 0.10                 # metres: the top of the colour scale
N_REF_POINTS = 200_000
REF_SEED = 1                # per_point_error(seed=0) samples the reference with seed + 1
_LOCK = threading.Lock()    # endpoints run in a thread pool; compute each file once

# matplotlib's "turbo", 256 x RGB uint8. `.[web]` has no matplotlib, so the table lives here
# (tests/test_web.py checks it against matplotlib when that is installed).
TURBO = np.frombuffer(bytes.fromhex(
    "30123b32154333184a341b51351e5836215f37246638276d392a733a2d793b2f803c32863d358b3e38913f3b973f3e9c"
    "4040a24143a74146ac4249b1424bb5434eba4451bf4454c34456c74559cb455ccf455ed34661d64664da4666dd4669e0"
    "466be3476ee64771e94773eb4776ee4778f0477bf2467df44680f64682f84685fa4687fb458afc458cfd448ffe4391fe"
    "4294ff4196ff4099ff3e9bfe3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf731aff52fb2f42eb4f22cb7f02ab9ee"
    "28bceb27bee925c0e723c3e422c5e220c7df1fc9dd1ecbda1ccdd81bd0d51ad2d21ad4d019d5cd18d7ca18d9c818dbc5"
    "18ddc218dec018e0bd19e2bb19e3b91ae4b61ce6b41de7b21fe9af20eaac22ebaa25eca727eea42aefa12cf09e2ff19b"
    "32f29835f39438f4913cf58e3ff68a43f78746f8844af8804ef97d52fa7a55fa7659fb735dfc6f61fc6c65fd6969fd66"
    "6dfe6271fe5f75fe5c79fe597dff5680ff5384ff5188ff4e8bff4b8fff4992ff4796fe4499fe429cfe409ffd3fa1fd3d"
    "a4fc3ca7fc3aa9fb39acfb38affa37b1f936b4f836b7f735b9f635bcf534bef434c1f334c3f134c6f034c8ef34cbed34"
    "cdec34d0ea34d2e935d4e735d7e535d9e436dbe236dde037dfdf37e1dd37e3db38e5d938e7d739e9d539ebd339ecd13a"
    "eecf3aefcd3af1cb3af2c93af4c73af5c53af6c33af7c13af8be39f9bc39faba39fbb838fbb637fcb336fcb136fdae35"
    "fdac34fea933fea732fea431fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fd8d27fd8a26fc8725fc8423fb8122"
    "fb7e21fa7b1ff9781ef9751df8721cf76f1af66c19f56918f46617f36315f26014f15d13f05b12ef5811ed5510ec530f"
    "eb500eea4e0de84b0ce7490ce5470be4450ae2430ae14109df3f08dd3d08dc3b07da3907d83706d63506d43305d23105"
    "d02f05ce2d04cc2b04ca2a04c82803c52603c32503c12302be2102bc2002b91e02b71d02b41b01b21a01af1801ac1701"
    "a91601a71401a41301a112019e10019b0f01980e01950d01920b018e0a018b09028808028507028106027e05027a0403"
), dtype=np.uint8).reshape(256, 3)


def turbo(x: np.ndarray) -> np.ndarray:
    """[N] values in 0..1 (clipped) -> [N, 3] RGB floats in 0..1, indexed the way matplotlib does."""
    idx = np.minimum((np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * 256).astype(np.int64), 255)
    return TURBO[idx] / 255.0


def vertex_error(pred_mesh, ref_mesh, *, n_ref_points: int = N_REF_POINTS, seed: int = REF_SEED) -> np.ndarray:
    """Distance (metres) from every vertex of `pred_mesh` to the nearest sampled point of `ref_mesh`."""
    r_pts, _ = _sample(ref_mesh, n_ref_points, seed)
    d, _ = _nn(np.asarray(pred_mesh.vertices), r_pts)
    return d


def colorize(pred_mesh, ref_mesh, *, vmax: float = VMAX):
    """A copy of `pred_mesh` (same vertices, same faces) whose vertex colours are turbo(error / vmax)."""
    import open3d as o3d

    out = o3d.geometry.TriangleMesh(pred_mesh.vertices, pred_mesh.triangles)
    out.vertex_colors = o3d.utility.Vector3dVector(turbo(vertex_error(pred_mesh, ref_mesh) / vmax))
    return out


def write_error_ply(mesh_path: Path, ref_path: Path, out_path: Path) -> Path:
    """Write (or reuse) the error-coloured copy of `mesh_path`; recomputed when either source is newer."""
    if is_fresh(out_path, mesh_path, ref_path):
        return out_path
    import open3d as o3d

    with _LOCK:
        if is_fresh(out_path, mesh_path, ref_path):
            return out_path
        colored = colorize(o3d.io.read_triangle_mesh(str(mesh_path)), o3d.io.read_triangle_mesh(str(ref_path)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.ply")
        o3d.io.write_triangle_mesh(str(tmp), colored)
        replace_atomic(tmp, out_path)
    return out_path
