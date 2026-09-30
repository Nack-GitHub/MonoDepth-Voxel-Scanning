"""roomscan_web: upload a zipped capture -> job -> mesh, all through the HTTP surface."""

import io
import json
import shutil
import zipfile

import numpy as np
import pytest

pytest.importorskip("open3d")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from roomscan.dataio.synthetic import write_synthetic_scene  # noqa: E402
from roomscan_web.app import create_app  # noqa: E402
from roomscan_web.jobs import ALLOWED_OVERRIDES  # noqa: E402
from roomscan_web.presets import PRESETS  # noqa: E402
from tests.test_custom_capture import _convert_synthetic_to_capture  # noqa: E402

FAST = {"fusion.voxel_size": 0.05, "fusion.sdf_trunc": 0.15}      # coarse grid: the wiring is what is tested


def _zip_dir(d):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in d.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(d.parent))       # one top-level folder inside the zip
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    """One synthetic capture folder (rgb + LiDAR depth + reference.ply) shared by the read-only tests."""
    d = tmp_path_factory.mktemp("capture")
    write_synthetic_scene(d / "syn", n_frames=10, hires=(512, 384))
    return _convert_synthetic_to_capture(d / "syn", d / "captures", n=10) / "room1"


def _upload(client, capture_dir, overrides=FAST, **form):
    return client.post("/scans", files={"file": ("room.zip", _zip_dir(capture_dir), "application/zip")},
                       data={"overrides": json.dumps(overrides), **form})


def test_upload_queue_and_fetch_mesh(tmp_path):
    write_synthetic_scene(tmp_path / "syn", n_frames=10, hires=(512, 384))
    root = _convert_synthetic_to_capture(tmp_path / "syn", tmp_path / "captures", n=10)
    app = create_app(tmp_path / "web", autostart=False)      # run the worker by hand
    client = TestClient(app)

    assert client.get("/").status_code == 200
    r = client.post("/scans", files={"file": ("room.zip", _zip_dir(root / "room1"), "application/zip")},
                    data={"overrides": json.dumps({"fusion.voxel_size": 0.05, "fusion.sdf_trunc": 0.15})})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued"
    assert client.get(f"/scans/{job['id']}/mesh.ply").status_code == 409

    app.state.runner.run_pending_sync()
    j = client.get(f"/scans/{job['id']}").json()
    assert j["status"] == "done", j.get("error")
    assert j["result"]["n_frames"] == 10
    # reference.ply was in the zip -> metrics exist; 10 frames leave the room incomplete, so check accuracy not chamfer
    assert j["result"]["metrics_3d"]["accuracy"] < 0.1
    ply = client.get(f"/scans/{job['id']}/mesh.ply")
    assert ply.status_code == 200 and ply.content[:3] == b"ply"
    assert [x["id"] for x in client.get("/scans").json()] == [job["id"]]


def test_rejects_bad_uploads(tmp_path):
    client = TestClient(create_app(tmp_path / "web", autostart=False))
    bad = io.BytesIO(b"not a zip")
    assert client.post("/scans", files={"file": ("x.zip", bad, "application/zip")}).status_code == 400
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "no capture here")
    empty.seek(0)
    assert client.post("/scans", files={"file": ("x.zip", empty, "application/zip")}).status_code == 400
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escape.txt", "zip-slip")
    evil.seek(0)
    assert client.post("/scans", files={"file": ("x.zip", evil, "application/zip")}).status_code == 400
    assert client.get("/scans/nope").status_code == 404


def test_presets_listed_in_dropdown_order(tmp_path):
    client = TestClient(create_app(tmp_path / "web", autostart=False))
    presets = client.get("/presets").json()
    assert [p["key"] for p in presets] == ["lidar", "ft_lidar_24", "mono_metric", "mono_sparse"]
    by_key = {p["key"]: p for p in presets}
    assert by_key["ft_lidar_24"]["overrides"]["depth.model"] == "depth_anything_v2_ft_lidar_all"
    assert by_key["ft_lidar_24"]["label"].startswith("FT-LiDAR-24")
    assert "upper bound" in by_key["mono_sparse"]["label"]          # ADR-011: never shown without it
    assert by_key["mono_sparse"]["overrides"]["depth.model"] == "depth_anything_v2_large"
    for p in presets:
        assert set(p["overrides"]) <= ALLOWED_OVERRIDES
        assert p["eta_s_per_frame"] > 0 and "ft_lidar_all" not in p["label"]
    assert list(PRESETS) == [p["key"] for p in presets]


def test_presets_on_upload(tmp_path, capture):
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)
    assert _upload(client, capture, preset="no_such_preset").status_code == 400
    assert not any((tmp_path / "web" / "captures").iterdir())       # rejected before extraction

    job = _upload(client, capture, preset="ft_lidar_24").json()
    assert job["preset"] == "ft_lidar_24" and job["label"] == PRESETS["ft_lidar_24"].label
    assert job["overrides"]["depth.model"] == "depth_anything_v2_ft_lidar_all"    # merged server-side
    assert job["overrides"]["fusion.voxel_size"] == "0.05"                         # explicit overrides kept

    job = _upload(client, capture, preset="lidar").json()
    app.state.runner._q.queue.clear()                                # only the lidar job runs (no torch in tests)
    app.state.runner._q.put(job["id"])
    app.state.runner.run_pending_sync()
    j = client.get(f"/scans/{job['id']}").json()
    assert j["status"] == "done" and j["label"] == "iPad LiDAR (sensor)", j.get("error")


def test_persist_across_restart(tmp_path, capture):
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)
    job = _upload(client, capture, preset="lidar").json()
    assert job["created_at"] and job["started_at"] is None and job["n_frames"] == 10
    app.state.runner.run_pending_sync()
    before = client.get(f"/scans/{job['id']}").json()
    assert before["status"] == "done", before.get("error")
    assert before["created_at"] <= before["started_at"] <= before["finished_at"]
    assert before["elapsed_s"] == pytest.approx(before["finished_at"] - before["started_at"])
    rec = json.loads((tmp_path / "web" / "results" / "web" / job["id"] / "job.json").read_text())
    assert set(rec) == {"id", "scene_dir", "overrides", "preset", "status", "error",
                        "created_at", "started_at", "finished_at"}

    client2 = TestClient(create_app(tmp_path / "web", autostart=False))      # "restart" on the same work dir
    after = client2.get(f"/scans/{job['id']}").json()
    assert after["status"] == "done" and after["preset"] == "lidar" and after["label"] == before["label"]
    assert after["result"]["metrics_3d"] == before["result"]["metrics_3d"]
    assert (after["started_at"], after["finished_at"]) == (before["started_at"], before["finished_at"])
    assert client2.get(f"/scans/{job['id']}/mesh.ply").status_code == 200
    assert [x["id"] for x in client2.get("/scans").json()] == [job["id"]]


def test_interrupted_job(tmp_path, capture):
    app = create_app(tmp_path / "web", autostart=False)
    job = _upload(TestClient(app), capture, preset="lidar").json()            # stays queued: no worker

    app2 = create_app(tmp_path / "web", autostart=False)
    j = TestClient(app2).get(f"/scans/{job['id']}").json()
    assert j["status"] == "failed" and j["error"] == "interrupted by server restart"
    assert app2.state.runner._q.empty()                                        # never requeued
    app2.state.runner.run_pending_sync()
    assert TestClient(app2).get(f"/scans/{job['id']}").json()["status"] == "failed"
    # and the failure is what a third start reads back, not "queued" again
    app3 = create_app(tmp_path / "web", autostart=False)
    assert TestClient(app3).get(f"/scans/{job['id']}").json()["error"] == "interrupted by server restart"


def test_static_modules_served(tmp_path):
    client = TestClient(create_app(tmp_path / "web", autostart=False))
    index = client.get("/").text
    assert '<script type="module" src="/static/js/main.js">' in index and "/static/app.css" in index
    for name in ("api", "viewer", "panels", "format", "main"):
        r = client.get(f"/static/js/{name}.js")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"], name   # module scripts need it
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/../app.py").status_code == 404


def test_meta_json_up_and_scene(tmp_path, capture):
    with_meta = tmp_path / "with_meta" / "room1"
    shutil.copytree(capture, with_meta)
    (with_meta / "meta.json").write_text(json.dumps(
        {"source": "arkitscenes", "scene": "47429736", "up": "z", "stride": 5}))
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)

    plain = _upload(client, capture, preset="lidar").json()
    assert (plain["up"], plain["scene"], plain["capture_stride"], plain["origin"]) == ("y", None, None, "web")
    job = _upload(client, with_meta, preset="lidar").json()
    assert (job["up"], job["scene"], job["capture_stride"]) == ("z", "47429736", 5)

    app.state.runner.run_pending_sync()                     # the loader must not trip over the extra file
    for j in (plain, job):
        assert client.get(f"/scans/{j['id']}").json()["status"] == "done"
    restored = TestClient(create_app(tmp_path / "web", autostart=False)).get(f"/scans/{job['id']}").json()
    assert (restored["up"], restored["scene"]) == ("z", "47429736")

    for bad in ('["z"]', "not json", json.dumps({"up": "sideways", "scene": {"a": 1}, "stride": "5"})):
        (with_meta / "meta.json").write_text(bad)           # comes from an upload: anything odd falls back
        j = _upload(client, with_meta, preset="lidar").json()
        assert (j["up"], j["scene"], j["capture_stride"]) == ("y", None, None)


def test_job_id_never_parses_as_a_number(tmp_path, capture, monkeypatch):
    import uuid

    class NumericLooking:
        hex = "1234e5678901" + "0" * 20                     # float("1234e5678901") parses: inf

    monkeypatch.setattr(uuid, "uuid4", lambda: NumericLooking)
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)
    job = _upload(client, capture, preset="lidar").json()
    assert job["id"] == "s1234e567890"
    app.state.runner.run_pending_sync()
    j = client.get(f"/scans/{job['id']}").json()
    assert j["status"] == "done", j.get("error")            # run_name stayed a string all the way to the out dir


def _n_vertices(ply_bytes):
    header = ply_bytes[:ply_bytes.index(b"end_header")].decode("ascii")
    return int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])


def test_reference_endpoint(tmp_path, capture):
    bare = tmp_path / "bare" / "room1"                      # the same capture, uploaded without a reference
    shutil.copytree(capture, bare)
    (bare / "reference.ply").unlink()
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)
    job, no_ref = _upload(client, capture, preset="lidar").json(), _upload(client, bare, preset="lidar").json()
    assert job["has_reference"] is True and no_ref["has_reference"] is False

    r = client.get(f"/scans/{job['id']}/reference.ply")     # there before the job runs: it came with the upload
    assert r.status_code == 200 and r.content[:3] == b"ply" and _n_vertices(r.content) > 0
    view = tmp_path / "web" / "results" / "web" / job["id"] / "reference_view.ply"
    mtime = view.stat().st_mtime_ns
    assert client.get(f"/scans/{job['id']}/reference.ply").status_code == 200
    assert view.stat().st_mtime_ns == mtime                 # second call is served from the cache

    assert client.get(f"/scans/{no_ref['id']}/reference.ply").status_code == 404
    assert client.get("/scans/nope/reference.ply").status_code == 404
    assert client.get("/scans/..%2F..%2Fapp/reference.ply").status_code == 404


def test_reference_view_thins_a_dense_reference(tmp_path):
    import open3d as o3d

    from roomscan_web import refmesh

    dense = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=400)     # ~640k triangles
    assert len(dense.triangles) > refmesh.MAX_TRIANGLES
    o3d.io.write_triangle_mesh(str(tmp_path / "ref.ply"), dense)
    view = o3d.io.read_triangle_mesh(str(refmesh.write_view(tmp_path / "ref.ply", tmp_path / "out" / "view.ply")))
    assert 0 < len(view.triangles) < len(dense.triangles) / 4
    assert not view.has_vertex_colors() and not view.has_vertex_normals()
    size = view.get_max_bound() - view.get_min_bound()
    assert abs(size - 2.0).max() < 0.1                                              # same sphere, fewer triangles


def _plane(z=0.0, size=0.5, n=40):
    """A size x size square at height z, as an n x n grid of vertices."""
    import open3d as o3d

    g = np.linspace(0.0, size, n)
    xs, ys = np.meshgrid(g, g)
    verts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)
    i = np.arange((n - 1) * n).reshape(n - 1, n)[:, :-1].ravel()
    tris = np.concatenate([np.stack([i, i + 1, i + n], 1), np.stack([i + 1, i + n + 1, i + n], 1)])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))


def test_errorcolor_zero_and_5cm():
    from roomscan_web import errorcolor

    def packed(rgb01):
        return (np.round(np.asarray(rgb01) * 255).astype(np.int64) * [65536, 256, 1]).sum(axis=1)

    ref = _plane()
    same = errorcolor.colorize(_plane(), ref)
    d0 = errorcolor.vertex_error(_plane(), ref)
    assert d0.max() < 0.003                                  # only the spacing of the 200k reference samples
    # error ~ 0 -> the dark-blue end of the scale (turbo(0) and its first few neighbours)
    assert np.isin(packed(same.vertex_colors), packed(errorcolor.TURBO[:8] / 255.0)).all()

    shifted = errorcolor.colorize(_plane(z=0.05), ref)       # 5 cm off on a 0-10 cm scale -> the middle colour
    assert np.abs(errorcolor.vertex_error(_plane(z=0.05), ref) - 0.05).max() < 0.002
    green = errorcolor.turbo(np.array([0.5]))[0]
    assert np.abs(np.asarray(shifted.vertex_colors) - green).max() < 0.03
    assert len(shifted.vertices) == len(ref.vertices) and len(shifted.triangles) == len(ref.triangles)

    far = errorcolor.colorize(_plane(z=0.5), ref)            # beyond the scale: clipped to the last colour
    assert np.allclose(np.asarray(far.vertex_colors), errorcolor.turbo(np.ones(1))[0])


def test_errorcolor_turbo_table():
    from roomscan_web import errorcolor

    assert errorcolor.TURBO.shape == (256, 3)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    assert np.allclose(errorcolor.turbo(x), errorcolor.TURBO[[0, 0, 128, 255, 255]] / 255.0)
    cm = pytest.importorskip("matplotlib").colormaps["turbo"]          # the figures use matplotlib's turbo
    grid = np.linspace(0.0, 1.0, 1001)
    assert np.abs(errorcolor.turbo(grid) - cm(grid)[:, :3]).max() <= 0.5 / 255 + 1e-9


def test_error_endpoint(tmp_path, capture):
    bare = tmp_path / "bare" / "room1"
    shutil.copytree(capture, bare)
    (bare / "reference.ply").unlink()
    app = create_app(tmp_path / "web", autostart=False)
    client = TestClient(app)
    job, no_ref = _upload(client, capture, preset="lidar").json(), _upload(client, bare, preset="lidar").json()
    assert client.get(f"/scans/{job['id']}/error.ply").status_code == 409        # not done yet
    app.state.runner.run_pending_sync()

    r = client.get(f"/scans/{job['id']}/error.ply")
    assert r.status_code == 200 and r.content[:3] == b"ply"
    header = r.content[:r.content.index(b"end_header")]
    assert b"property uchar red" in header                                        # vertex colours
    assert _n_vertices(r.content) == _n_vertices(client.get(f"/scans/{job['id']}/mesh.ply").content)
    cached = tmp_path / "web" / "results" / "web" / job["id"] / "error.ply"
    mtime = cached.stat().st_mtime_ns
    assert client.get(f"/scans/{job['id']}/error.ply").content == r.content
    assert cached.stat().st_mtime_ns == mtime                                     # not recomputed

    mesh = tmp_path / "web" / "results" / "web" / job["id"] / "mesh.ply"
    import os
    os.utime(mesh, ns=(mtime + 5_000_000_000, mtime + 5_000_000_000))             # a newer mesh drops the cache
    assert client.get(f"/scans/{job['id']}/error.ply").status_code == 200
    assert cached.stat().st_mtime_ns != mtime

    assert client.get(f"/scans/{no_ref['id']}/error.ply").status_code == 404      # done, but nothing to compare to
    assert client.get("/scans/nope/error.ply").status_code == 404


def test_legend_stops_match_turbo():
    import re

    from roomscan_web import errorcolor
    from roomscan_web.app import STATIC

    js = (STATIC / "js" / "panels.js").read_text(encoding="utf-8")
    stops = re.findall(r"#[0-9a-f]{6}", re.search(r"TURBO_STOPS = \[(.*?)\]", js, re.S).group(1))
    assert stops == ["#{:02x}{:02x}{:02x}".format(*errorcolor.TURBO[round(i * 25.5)]) for i in range(11)]
