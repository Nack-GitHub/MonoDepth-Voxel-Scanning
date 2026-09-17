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
from tests.test_custom_capture import _convert_synthetic_to_capture  # noqa: E402


def _zip_dir(d):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in d.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(d.parent))       # one top-level folder inside the zip
    buf.seek(0)
    return buf


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
