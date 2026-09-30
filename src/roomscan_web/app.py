"""FastAPI app: upload a capture, poll the job, fetch the mesh, view it in the browser.

    GET  /presets          upload presets (key, paper label, overrides) in dropdown order
    POST /scans            multipart: file=<capture.zip> [, preset=<key>] [, overrides=<json dict>]
    GET  /scans            all jobs
    GET  /scans/{id}       job status + metrics.json when done
    GET  /scans/{id}/mesh.ply
    GET  /scans/{id}/reference.ply   the capture's Faro reference, thinned for display (404 without one)
    GET  /scans/{id}/error.ply       the mesh coloured by distance to the reference, turbo 0-10 cm
    GET  /gallery                                  finished experiment runs (read-only, see gallery.py)
    GET  /gallery/{exp}/{run}/mesh.ply | reference.ply | error.ply
    GET  /                 three.js viewer (static/index.html)
    GET  /static/...       css / js modules of the viewer

The zip must contain the folder format of `roomscan.dataio.custom` (rgb/, poses.json,
intrinsics.json, optional depth/ confidence/ sparse/), either at the root or in one
top-level folder.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from roomscan_web import errorcolor, gallery, refmesh
from roomscan_web.jobs import Job, JobRunner, derived_files, reference_path
from roomscan_web.presets import public_presets, resolve

STATIC = Path(__file__).parent / "static"
# Windows may map .js to text/plain in the registry; browsers refuse module scripts served that way.
mimetypes.add_type("text/javascript", ".js")
MAX_UPLOAD_BYTES = int(os.environ.get("ROOMSCAN_MAX_UPLOAD_MB", "512")) * 1024 * 1024


class _Static(StaticFiles):
    """Static files the browser must revalidate: one stale cached module next to fresh ones breaks the page."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(work_dir: str | Path | None = None, preset: str | None = None, *, autostart: bool = True,
               gallery_root: str | Path | None = None, gallery_exps: Sequence[str] | None = None) -> FastAPI:
    runner = JobRunner(work_dir or os.environ.get("ROOMSCAN_WORK_DIR", "outputs/web"),
                       preset or os.environ.get("ROOMSCAN_PRESET", "configs/depth/lidar.yaml"),
                       autostart=autostart)
    g_root = Path(gallery_root or os.environ.get("ROOMSCAN_GALLERY_ROOT", gallery.DEFAULT_ROOT))
    g_exps = gallery.allowed_exps(
        gallery_exps or os.environ.get("ROOMSCAN_GALLERY_EXPS", ",".join(gallery.DEFAULT_EXPS)).split(","))
    g_cache = runner.work_dir / "cache"          # derived gallery files live here, never under g_root
    app = FastAPI(title="roomscan", version="0.1.0")
    app.state.runner = runner
    app.mount("/static", _Static(directory=STATIC), name="static")

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

    def job_or_404(job_id: str) -> Job:
        job = runner.get(job_id)                 # a dict lookup: the id never becomes a path
        if job is None:
            raise HTTPException(404, "no such scan")
        return job

    @app.get("/scans/{job_id}")
    def get_scan(job_id: str) -> dict:
        return job_or_404(job_id).public()

    @app.get("/scans/{job_id}/mesh.ply")
    def get_mesh(job_id: str) -> FileResponse:
        job = job_or_404(job_id)
        if job.status != "done" or job.mesh_path is None or not job.mesh_path.is_file():
            raise HTTPException(409, f"scan is {job.status}")
        return _ply(job.mesh_path, f"{job_id}.ply")

    @app.get("/scans/{job_id}/reference.ply")
    def get_reference(job_id: str) -> FileResponse:
        job = job_or_404(job_id)
        ref = reference_path(job)
        if ref is None:
            raise HTTPException(404, "no Faro reference")
        try:
            view = refmesh.write_view(ref, derived_files(runner.work_dir, job)["reference_view"])
        except ValueError as e:
            raise HTTPException(404, f"no Faro reference: {e}") from None
        return _ply(view, f"{job_id}_reference.ply")

    @app.get("/scans/{job_id}/error.ply")
    def get_error(job_id: str) -> FileResponse:
        job = job_or_404(job_id)
        if job.status != "done" or job.mesh_path is None or not job.mesh_path.is_file():
            raise HTTPException(409, f"scan is {job.status}")
        ref = reference_path(job)
        if ref is None:
            raise HTTPException(404, "no Faro reference")
        try:
            out = errorcolor.write_error_ply(job.mesh_path, ref, derived_files(runner.work_dir, job)["error"])
        except ValueError as e:                   # an empty mesh on either side
            raise HTTPException(404, f"no error colours: {e}") from None
        return _ply(out, f"{job_id}_error.ply")

    def run_or_404(exp: str, run: str) -> Path:
        try:
            return gallery.resolve_run(g_root, g_exps, exp, run)
        except ValueError:
            raise HTTPException(404, "no such gallery item") from None

    @app.get("/gallery")
    def list_gallery() -> list[dict]:
        return gallery.list_runs(g_root, g_exps, g_cache)

    @app.get("/gallery/{exp}/{run}/mesh.ply")
    def gallery_mesh(exp: str, run: str) -> FileResponse:
        return _ply(run_or_404(exp, run) / "mesh.ply", f"{run}.ply")

    def derived_or_404(build, exp: str, run: str, filename: str) -> FileResponse:
        try:
            out = build(run_or_404(exp, run), g_cache / exp)
        except ValueError:                          # an empty mesh on either side
            out = None
        if out is None:
            raise HTTPException(404, "no Faro reference")
        return _ply(out, filename)

    @app.get("/gallery/{exp}/{run}/reference.ply")
    def gallery_reference(exp: str, run: str) -> FileResponse:
        return derived_or_404(gallery.reference_view, exp, run, f"{run}_reference.ply")

    @app.get("/gallery/{exp}/{run}/error.ply")
    def gallery_error(exp: str, run: str) -> FileResponse:
        return derived_or_404(gallery.error_mesh, exp, run, f"{run}_error.ply")

    return app


def _ply(path: Path, filename: str) -> FileResponse:
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


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

