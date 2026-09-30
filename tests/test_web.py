"""roomscan_web: upload a zipped capture -> job -> mesh, all through the HTTP surface."""

import io
import json
import zipfile

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
