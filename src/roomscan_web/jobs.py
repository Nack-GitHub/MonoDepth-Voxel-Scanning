"""One-worker job queue running ReconstructionPipeline on uploaded captures.

Deliberately simple (threads + dict, no broker): an MVP serves one phone at a time,
and results are files anyway (ADR-006). Swap for a real queue behind the same
`submit / get / list` surface when there is more than one worker.

Jobs survive a server restart through `results/web/<id>/job.json`, written next to the
pipeline's own config.yaml / metrics.json on every status change. A job found `queued` or
`running` at start-up is marked failed instead of requeued: a 7-minute mono job must not
start by itself when the server is reopened on stage.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roomscan.config import load_config
from roomscan_web._fs import replace_atomic
from roomscan_web.presets import PRESETS, resolve

DEFAULT_PRESET = "configs/depth/lidar.yaml"
INTERRUPTED = "interrupted by server restart"
RGB_SUFFIXES = (".jpg", ".jpeg", ".png")

# What a client may override per scan. Everything else is pinned by the preset.
ALLOWED_OVERRIDES = {
    "depth.source", "depth.model", "depth.aligner",
    "fusion.voxel_size", "fusion.sdf_trunc", "fusion.confidence_weights",
    "dataset.frame_stride", "dataset.max_frames",
}
# What job.json holds; result / mesh_path are re-read from the run folder, the rest is derived.
PERSISTED = ("id", "scene_dir", "overrides", "preset", "status", "error", "created_at", "started_at", "finished_at")


@dataclass
class Job:
    id: str
    scene_dir: Path
    overrides: dict[str, str]
    preset: str | None = None         # key in presets.PRESETS the scan was submitted with (None = raw overrides)
    status: str = "queued"            # queued | running | done | failed
    error: str | None = None
    result: dict[str, Any] | None = None
    mesh_path: Path | None = None
    log: list[str] = field(default_factory=list)
    created_at: float | None = None   # unix seconds, server clock
    started_at: float | None = None
    finished_at: float | None = None
    n_frames: int | None = None       # frames the run will fuse (rgb count after stride / max_frames)
    scene: str | None = None          # from the capture's meta.json: which room this is (None = unknown)
    up: str = "y"                     # "y" | "z": world up-axis of the capture
    capture_stride: int | None = None  # frame stride the capture was exported with
    has_reference: bool = False       # the capture came with a reference.ply (Faro mesh)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["scene_dir"] = str(self.scene_dir)
        d["mesh_path"] = str(self.mesh_path) if self.mesh_path else None
        d["mesh_url"] = f"/scans/{self.id}/mesh.ply" if self.status == "done" else None
        d["label"] = PRESETS[self.preset].label if self.preset in PRESETS else "custom overrides"
        d["origin"] = "web"               # a demo run, as opposed to a gallery item ("paper")
        # elapsed on the server clock, so a timer in the page keeps counting right across reloads and devices
        if self.started_at is not None:
            d["elapsed_s"] = (self.finished_at or time.time()) - self.started_at
        else:
            d["elapsed_s"] = None
        eta = PRESETS[self.preset].eta_s_per_frame if self.preset in PRESETS else None
        d["eta_s"] = eta * self.n_frames if eta is not None and self.n_frames else None
        return d


class JobRunner:
    def __init__(self, work_dir: str | Path, preset: str = DEFAULT_PRESET, *, autostart: bool = True):
        self.work_dir = Path(work_dir)
        self.captures_dir = self.work_dir / "captures"
        self.results_dir = self.work_dir / "results"         # the pipeline's output.root; jobs land in results/web/
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.preset = preset
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._restore()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="roomscan-worker")
        if autostart:
            self._thread.start()

    # ------------------------------------------------------------------ public
    def new_capture_dir(self) -> tuple[str, Path]:
        # "s" first: a bare hex id such as 123456789012 or 1234e5678901 is a number to the config parser,
        # and the pipeline then fails joining a path with an int/float (about one upload in a hundred)
        job_id = "s" + uuid.uuid4().hex[:11]
        d = self.captures_dir / job_id
        d.mkdir(parents=True)
        return job_id, d

    def submit(self, job_id: str, scene_dir: Path, overrides: dict[str, str] | None = None,
               preset: str | None = None) -> Job:
        """Queue a scan. `preset` (a presets.PRESETS key) supplies overrides; explicit `overrides` win over it."""
        merged = {**(resolve(preset).overrides if preset else {}), **(overrides or {})}
        bad = set(merged) - ALLOWED_OVERRIDES
        if bad:
            raise ValueError(f"overrides not allowed: {sorted(bad)}; allowed: {sorted(ALLOWED_OVERRIDES)}")
        job = Job(id=job_id, scene_dir=scene_dir, overrides=merged, preset=preset, created_at=time.time(),
                  n_frames=_count_frames(scene_dir, merged), **read_meta(scene_dir))
        job.has_reference = reference_path(job) is not None
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        self._q.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def run_dir(self, job_id: str) -> Path:
        return run_dir(self.work_dir, job_id)

    def run_pending_sync(self) -> None:
        """Drain the queue on the calling thread (tests; also handy for CLI batch use)."""
        while not self._q.empty():
            self._run(self._q.get())

    # ------------------------------------------------------------------ persistence
    def _save(self, job: Job) -> None:
        d = self.run_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        rec = {k: getattr(job, k) for k in PERSISTED}
        rec["scene_dir"] = _portable(job.scene_dir, self.work_dir)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        replace_atomic(tmp, d / "job.json")              # a crash never leaves half a record

    def _restore(self) -> None:
        for job in read_jobs(self.work_dir):
            if job.status in ("queued", "running"):       # it died with the old server; say so, do not rerun it
                job.status, job.error = "failed", INTERRUPTED
                job.finished_at = job.finished_at or time.time()
                self._save(job)
            self._jobs[job.id] = job

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            self._run(self._q.get())

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status, job.started_at = "running", time.time()
        self._save(job)
        try:
            from roomscan.pipeline import ReconstructionPipeline

            cfg = load_config(self.preset, [
                "dataset.name=custom", f"dataset.root={job.scene_dir.parent}", f"dataset.scene={job.scene_dir.name}",
                "eval.mask_to_gt=false", "eval.compute_2d_metrics=false",
                f"output.root={self.results_dir}", "output.experiment=web", f"output.run_name={job.id}",
                *[f"{k}={v}" for k, v in job.overrides.items()],
            ])
            res = ReconstructionPipeline(cfg).run()
            job.result = json.loads((res.out_dir / "metrics.json").read_text())
            job.mesh_path = res.out_dir / "mesh.ply"
            job.n_frames = res.n_frames
            job.status = "done"
        except Exception as e:  # noqa: BLE001 — the job record is the error channel
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc())
        job.finished_at = time.time()
        self._save(job)


def run_dir(work_dir: Path, job_id: str) -> Path:
    """Where the pipeline writes a job's config.yaml / metrics.json / mesh.ply (and we keep job.json)."""
    return Path(work_dir) / "results" / "web" / job_id


def reference_path(job: Job) -> Path | None:
    """The capture's own reference.ply, or None when it was uploaded without one."""
    path = job.scene_dir / "reference.ply"
    return path if path.is_file() else None


def read_jobs(work_dir: str | Path) -> list[Job]:
    """The jobs recorded under a work dir, oldest first, exactly as their job.json left them (nothing is written).

    Finished jobs come back with their metrics and mesh path; a record that cannot be read is skipped.
    """
    work_dir, jobs = Path(work_dir), []
    for path in run_dir(work_dir, "*").parent.glob("*/job.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            scene_dir = Path(rec["scene_dir"])
            job = Job(id=str(rec["id"]), scene_dir=scene_dir if scene_dir.is_absolute() else work_dir / scene_dir,
                      overrides=dict(rec.get("overrides") or {}), preset=rec.get("preset"),
                      status=str(rec["status"]), error=rec.get("error"), created_at=rec.get("created_at"),
                      started_at=rec.get("started_at"), finished_at=rec.get("finished_at"))
            job.scene, job.up, job.capture_stride = read_meta(job.scene_dir).values()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if job.id != path.parent.name:
            continue
        job.has_reference = reference_path(job) is not None
        job.n_frames = _count_frames(job.scene_dir, job.overrides)
        if job.status == "done":
            try:
                job.result = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
                job.mesh_path = path.parent / "mesh.ply"
                job.n_frames = job.result.get("n_frames", job.n_frames)
            except (OSError, ValueError):
                job.status, job.error = "failed", "result files missing after restart"
        jobs.append(job)
    return sorted(jobs, key=lambda j: j.created_at or 0.0)


def _portable(path: Path, base: Path) -> str:
    """`path` relative to the work dir when it lives inside it, so a moved work dir still restores."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_meta(scene_dir: Path) -> dict[str, Any]:
    """`meta.json` of a capture -> {scene, up, capture_stride}. Missing or malformed: unknown room, Y-up.

    The file comes out of an uploaded zip, so nothing in it is trusted beyond these three checked fields.
    """
    meta: dict[str, Any] = {"scene": None, "up": "y", "capture_stride": None}
    try:
        raw = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return meta
    if not isinstance(raw, dict):
        return meta
    if isinstance(raw.get("scene"), (str, int)) and not isinstance(raw["scene"], bool) and str(raw["scene"]):
        meta["scene"] = str(raw["scene"])[:64]
    if raw.get("up") in ("y", "z"):
        meta["up"] = raw["up"]
    if isinstance(raw.get("stride"), int) and not isinstance(raw["stride"], bool) and raw["stride"] > 0:
        meta["capture_stride"] = raw["stride"]
    return meta


def derived_files(work_dir: Path, job: Job) -> dict[str, Path]:
    """Cache paths of what the viewer derives from a finished job: the reference's display copy, the error mesh."""
    d = run_dir(work_dir, job.id)
    return {"reference_view": d / "reference_view.ply", "error": d / "error.ply"}


def _count_frames(scene_dir: Path, overrides: dict[str, str]) -> int | None:
    """How many frames the run will fuse — only for the ETA hint, so any surprise is just `None`."""
    try:
        n = sum(1 for p in (scene_dir / "rgb").iterdir() if p.suffix.lower() in RGB_SUFFIXES)
        stride = max(1, int(overrides.get("dataset.frame_stride", 1)))
        n = -(-n // stride)
        max_frames = overrides.get("dataset.max_frames")
        return min(n, int(max_frames)) if max_frames not in (None, "", "null", "None") else n
    except (OSError, ValueError):
        return None
