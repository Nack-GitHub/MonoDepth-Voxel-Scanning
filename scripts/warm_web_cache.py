"""Build the web viewer's derived files ahead of the demo, so the first click on stage is not the slow one.

    python scripts/warm_web_cache.py --scenes 47429736 47333774     # two rooms of the gallery + all web jobs
    python scripts/warm_web_cache.py                                 # every gallery room + all web jobs

For each gallery run of the chosen rooms and for each finished web job it makes what the page asks for
on "Faro overlay" and "Color: error": the room's reference mesh, its display copy, and the error-coloured
mesh. It calls the same functions as the HTTP endpoints (no server needed) and skips whatever is already
cached and still newer than its sources, so it is safe to run again. Nothing is written under
experiments/results — only under <work dir>/cache and <work dir>/results/web/<job>/.

It ends with a table of every finished web job next to the paper run of the same room and method: a
demo run (thinned capture, no GT mask) is not the paper's number, and a gap beyond +-20 % of Chamfer is
flagged so it can be looked into before anyone asks about it on stage.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from roomscan_web import errorcolor, gallery, refmesh
from roomscan_web._fs import is_fresh
from roomscan_web.jobs import derived_files, read_jobs, reference_path

# upload preset -> the run of Exp1/Exp6 that is the same method
PAPER_RUN = {"lidar": "lidar", "ft_lidar_24": "mono_ft_lidar_all", "mono_metric": "mono_metric",
             "mono_sparse": "mono_sparse"}
TOLERANCE = 0.20        # relative Chamfer gap a demo run may have to its paper run before it is flagged


def warm_jobs(work_dir: Path):
    """Finished web jobs that came with a reference -> (job id, built | cached)."""
    for job in read_jobs(work_dir):
        ref = reference_path(job)
        if job.status != "done" or job.mesh_path is None or ref is None:
            continue
        out = derived_files(work_dir, job)
        cached = is_fresh(out["reference_view"], ref) and is_fresh(out["error"], job.mesh_path, ref)
        refmesh.write_view(ref, out["reference_view"])
        errorcolor.write_error_ply(job.mesh_path, ref, out["error"])
        yield job.id, "cached" if cached else "built"


def compare_with_paper(work_dir: Path, paper_runs: list[dict]):
    """Finished web jobs that have a paper run of the same room and preset -> one table row each."""
    paper = {(r["scene"], r["run"]): r for r in paper_runs}
    for job in read_jobs(work_dir):
        run = paper.get((job.scene, PAPER_RUN.get(job.preset)))
        web = (job.result or {}).get("metrics_3d")
        if job.status != "done" or run is None or not web or not run["metrics"].get("metrics_3d"):
            continue
        ref = run["metrics"]["metrics_3d"]["chamfer"]
        gap = web["chamfer"] / ref - 1.0
        yield (f"{job.scene}  {run['label']:<34} demo {web['chamfer'] * 100:5.1f} cm   paper {ref * 100:5.1f} cm   "
               f"{gap:+5.0%}  {'ok' if abs(gap) <= TOLERANCE else 'LOOK INTO THIS'}   ({job.id})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", nargs="*", default=None, help="gallery rooms to warm (default: all)")
    ap.add_argument("--work-dir", default=os.environ.get("ROOMSCAN_WORK_DIR", "outputs/web"))
    ap.add_argument("--gallery-root", default=os.environ.get("ROOMSCAN_GALLERY_ROOT", gallery.DEFAULT_ROOT))
    ap.add_argument("--exps", nargs="*",
                    default=os.environ.get("ROOMSCAN_GALLERY_EXPS", ",".join(gallery.DEFAULT_EXPS)).split(","))
    ap.add_argument("--no-web-jobs", action="store_true", help="only the gallery")
    args = ap.parse_args()

    work_dir, counts, t0 = Path(args.work_dir), {}, time.perf_counter()
    root, cache = Path(args.gallery_root), work_dir / "cache"

    def report(what: str, steps) -> None:
        t = time.perf_counter()
        for name, state in steps:
            counts[state] = counts.get(state, 0) + 1
            print(f"{state:>12}  {what}  {name}  ({time.perf_counter() - t:.1f} s)")
            t = time.perf_counter()

    report("gallery", gallery.warm(root, args.exps, cache, args.scenes))
    if not args.no_web_jobs:
        report("web job", warm_jobs(work_dir))
    summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items())) or "nothing to warm"
    print(f"{summary} in {time.perf_counter() - t0:.1f} s")

    rows = list(compare_with_paper(work_dir, gallery.list_runs(root, args.exps, cache)))
    if rows:
        print(f"\ndemo run vs paper run (Chamfer; flagged beyond +-{TOLERANCE:.0%}):")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
