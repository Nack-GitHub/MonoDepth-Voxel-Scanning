"""FastAPI app: upload a capture, poll the job, fetch the mesh, view it in the browser.

    GET  /presets          upload presets (key, paper label, overrides) in dropdown order
    POST /scans            multipart: file=<capture.zip> [, preset=<key>] [, overrides=<json dict>]
    GET  /scans            all jobs
    GET  /scans/{id}       job status + metrics.json when done
    GET  /scans/{id}/mesh.ply
    GET  /                 three.js viewer (static/index.html)

The zip must contain the folder format of `roomscan.dataio.custom` (rgb/, poses.json,
intrinsics.json, optional depth/ confidence/ sparse/), either at the root or in one
top-level folder.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from roomscan_web.jobs import JobRunner
from roomscan_web.presets import public_presets, resolve

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("ROOMSCAN_MAX_UPLOAD_MB", "512")) * 1024 * 1024


def create_app(work_dir: str | Path | None = None, preset: str | None = None, *, autostart: bool = True) -> FastAPI:
    runner = JobRunner(work_dir or os.environ.get("ROOMSCAN_WORK_DIR", "outputs/web"),
                       preset or os.environ.get("ROOMSCAN_PRESET", "configs/depth/lidar.yaml"),
                       autostart=autostart)
    app = FastAPI(title="roomscan", version="0.1.0")
    app.state.runner = runner

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/presets")
    def list_presets() -> list[dict]:
        return public_presets()

    @app.get("/scans")
    def list_scans() -> list[dict]:
        return [j.public() for j in runner.list()]

    @app.post("/scans", status_code=202)
    async def create_scan(file: UploadFile = File(...), overrides: str = Form("{}"),  # noqa: B008
                          preset: str | None = Form(None)) -> dict:  # noqa: B008
        try:
            ov = json.loads(overrides or "{}")
            if not isinstance(ov, dict):
                raise ValueError("overrides must be a JSON object")
        except ValueError as e:
            raise HTTPException(400, f"bad overrides: {e}") from None
        try:
            preset = resolve(preset).key if preset else None       # reject before anything touches the disk
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        job_id, capture_dir = runner.new_capture_dir()
        zip_path = capture_dir.with_suffix(".zip")
        size = 0
        with zip_path.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    zip_path.unlink(missing_ok=True)
                    shutil.rmtree(capture_dir, ignore_errors=True)
                    raise HTTPException(413, f"upload larger than {MAX_UPLOAD_BYTES >> 20} MB")
                out.write(chunk)
        try:
            _safe_extract(zip_path, capture_dir)
        except (zipfile.BadZipFile, ValueError) as e:
            shutil.rmtree(capture_dir, ignore_errors=True)
            raise HTTPException(400, f"bad zip: {e}") from None
        finally:
            zip_path.unlink(missing_ok=True)
        scene_dir = _find_capture_root(capture_dir)
        if scene_dir is None:
            shutil.rmtree(capture_dir, ignore_errors=True)
            raise HTTPException(400, "zip does not contain rgb/ + poses.json + intrinsics.json")
        try:
            job = runner.submit(job_id, scene_dir, {str(k): str(v) for k, v in ov.items()}, preset)
        except ValueError as e:
            shutil.rmtree(capture_dir, ignore_errors=True)
            raise HTTPException(400, str(e)) from None
        return job.public()

    @app.get("/scans/{job_id}")
    def get_scan(job_id: str) -> dict:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, "no such scan")
        return job.public()

    @app.get("/scans/{job_id}/mesh.ply")
    def get_mesh(job_id: str) -> FileResponse:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, "no such scan")
        if job.status != "done" or job.mesh_path is None or not job.mesh_path.is_file():
            raise HTTPException(409, f"scan is {job.status}")
        return FileResponse(job.mesh_path, media_type="application/octet-stream", filename=f"{job_id}.ply")

    return app


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract, refusing paths that would escape `dest` (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise ValueError(f"unsafe path in zip: {member.filename}")
        zf.extractall(dest)


def _find_capture_root(d: Path) -> Path | None:
    """The capture folder is `d` itself or its single top-level directory."""
    def ok(p: Path) -> bool:
        return (p / "rgb").is_dir() and (p / "poses.json").is_file() and (p / "intrinsics.json").is_file()
    if ok(d):
        return d
    subs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith("__")]
    if len(subs) == 1 and ok(subs[0]):
        return subs[0]
    return None

