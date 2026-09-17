"""One-worker job queue running ReconstructionPipeline on uploaded captures.

Deliberately simple (threads + dict, no broker): an MVP serves one phone at a time,
and results are files anyway (ADR-006). Swap for a real queue behind the same
`submit / get / list` surface when there is more than one worker.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roomscan.config import load_config

DEFAULT_PRESET = "configs/depth/lidar.yaml"

# What a client may override per scan. Everything else is pinned by the preset.
ALLOWED_OVERRIDES = {
    "depth.source", "depth.model", "depth.aligner",
    "fusion.voxel_size", "fusion.sdf_trunc", "fusion.confidence_weights",
    "dataset.frame_stride", "dataset.max_frames",
}


@dataclass
class Job:
    id: str
    scene_dir: Path
    overrides: dict[str, str]
    status: str = "queued"            # queued | running | done | failed
    error: str | None = None
    result: dict[str, Any] | None = None
    mesh_path: Path | None = None
    log: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["scene_dir"] = str(self.scene_dir)
        d["mesh_path"] = str(self.mesh_path) if self.mesh_path else None
        d["mesh_url"] = f"/scans/{self.id}/mesh.ply" if self.status == "done" else None
        return d


class JobRunner:
    def __init__(self, work_dir: str | Path, preset: str = DEFAULT_PRESET, *, autostart: bool = True):
        self.work_dir = Path(work_dir)
        self.captures_dir = self.work_dir / "captures"
        self.results_dir = self.work_dir / "results"
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.preset = preset
        self._jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="roomscan-worker")
        if autostart:
            self._thread.start()

    # ------------------------------------------------------------------ public
    def new_capture_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        d = self.captures_dir / job_id
        d.mkdir(parents=True)
        return job_id, d

    def submit(self, job_id: str, scene_dir: Path, overrides: dict[str, str] | None = None) -> Job:
        bad = set(overrides or {}) - ALLOWED_OVERRIDES
        if bad:
            raise ValueError(f"overrides not allowed: {sorted(bad)}; allowed: {sorted(ALLOWED_OVERRIDES)}")
        job = Job(id=job_id, scene_dir=scene_dir, overrides=dict(overrides or {}))
        with self._lock:
            self._jobs[job_id] = job
        self._q.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def run_pending_sync(self) -> None:
        """Drain the queue on the calling thread (tests; also handy for CLI batch use)."""
        while not self._q.empty():
            self._run(self._q.get())

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            self._run(self._q.get())

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = "running"
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
            job.status = "done"
        except Exception as e:  # noqa: BLE001 — the job record is the error channel
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc())
