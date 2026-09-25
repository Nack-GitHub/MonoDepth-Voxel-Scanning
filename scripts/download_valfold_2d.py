"""Download the 2D-only Validation-fold test scenes (configs/eval/valfold_2d.yaml, ADR-014), keeping only
the frames the 2D evaluation reads.

    python scripts/download_valfold_2d.py --list configs/eval/valfold_2d.yaml --root data/arkitscenes

Per scene: lowres_wide.traj and lowres_wide_intrinsics in full; highres_depth (Faro) thinned to every
`--faro-every`-th frame (sorted by timestamp) — these ARE the evaluated frames, so the experiment runs
with frame_stride 1; from vga_wide, lowres_depth and confidence only the one image nearest to each kept
Faro timestamp (within `--tol` s, the loader's own 0.02 s match tolerance) is extracted from the zip.
~250 MB per scene on disk instead of ~1.5 GB. Idempotent (`.<asset>.ok` markers), one scene at a
time, zips deleted after extraction.
Refuses any video whose visit_id is in configs/training/splits.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_training_scenes import BASE, _curl, _unzip  # noqa: E402

FULL = ("lowres_wide_intrinsics", "highres_depth")
SUBSET = ("vga_wide", "lowres_depth", "confidence")


def _ts(name: str) -> float:
    return float(Path(name).stem.rsplit("_", 1)[1])


def _thin_faro(scene_dir: Path, every: int) -> int:
    """Keep every `every`-th highres_depth frame (idempotent via a marker naming the factor)."""
    marker = scene_dir / f".highres_depth.every{every}"
    files = sorted((scene_dir / "highres_depth").glob("*.png"), key=lambda p: _ts(p.name))
    if every > 1 and not marker.is_file():
        for k, f in enumerate(files):
            if k % every:
                f.unlink()
        marker.touch()
        files = files[::every]
    return len(files)


def _extract_near(zpath: Path, scene_dir: Path, asset: str, keep_ts: np.ndarray, tol: float) -> int:
    """Extract, for every kept timestamp, the single nearest image of the zip within tol."""
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.endswith(".png")]
        ts = np.array([_ts(i.filename) for i in infos])
        order = np.argsort(ts)
        infos, ts = [infos[k] for k in order], ts[order]
        chosen = set()
        for t in keep_ts:
            i = int(np.searchsorted(ts, t))
            cand = [j for j in (i - 1, i) if 0 <= j < len(ts) and abs(ts[j] - t) <= tol]
            if cand:
                chosen.add(min(cand, key=lambda j: abs(ts[j] - t)))
        for j in sorted(chosen):
            info = infos[j]
            dst = scene_dir / asset / Path(info.filename).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
    return len(chosen)


def _prune_to_nearest(scene_dir: Path, asset: str, keep_ts: np.ndarray, tol: float) -> None:
    """Scenes fetched by an earlier version of this script: drop images not nearest to a kept Faro frame."""
    files = sorted((scene_dir / asset).glob("*.png"), key=lambda p: _ts(p.name))
    if not files:
        return
    ts = np.array([_ts(p.name) for p in files])
    keep = set()
    for t in keep_ts:
        i = int(np.searchsorted(ts, t))
        cand = [j for j in (i - 1, i) if 0 <= j < len(ts) and abs(ts[j] - t) <= tol]
        if cand:
            keep.add(min(cand, key=lambda j: abs(ts[j] - t)))
    for j, f in enumerate(files):
        if j not in keep:
            f.unlink()


def download(root: Path, e: dict, tol: float, faro_every: int) -> None:
    vid, fold = str(e["video_id"]), e["fold"]
    scene_dir = root / "raw" / fold / vid
    scene_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{fold}/{vid}"
    if not (scene_dir / "lowres_wide.traj").is_file():
        _curl(f"{url}/lowres_wide.traj", scene_dir / "lowres_wide.traj")
    for asset in FULL + SUBSET:
        marker = scene_dir / f".{asset}.ok"
        if marker.is_file():
            if asset == "highres_depth":
                _thin_faro(scene_dir, faro_every)
            elif asset in SUBSET:
                keep = np.array(sorted(_ts(p.name) for p in (scene_dir / "highres_depth").glob("*.png")))
                _prune_to_nearest(scene_dir, asset, keep, tol)
            continue
        zpath = scene_dir / f"{asset}.zip"
        if not zpath.is_file():
            _curl(f"{url}/{asset}.zip", zpath)
        shutil.rmtree(scene_dir / asset, ignore_errors=True)
        if asset in FULL:
            _unzip(zpath, scene_dir, asset)
            msg = "all"
            if asset == "highres_depth":
                msg = f"{_thin_faro(scene_dir, faro_every)} frames kept (every {faro_every})"
        else:
            keep = np.array(sorted(_ts(p.name) for p in (scene_dir / "highres_depth").glob("*.png")))
            msg = f"{_extract_near(zpath, scene_dir, asset, keep, tol)} nearest to {len(keep)} Faro frames"
        zpath.unlink()
        marker.touch()
        print(f"  {vid} {asset}: {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/eval/valfold_2d.yaml")
    ap.add_argument("--root", default="data/arkitscenes")
    ap.add_argument("--splits", default="configs/training/splits.yaml")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--faro-every", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root)
    splits = yaml.safe_load(Path(a.splits).read_text())
    used = {str(e["visit_id"]) for k in ("test", "val", "train_faro", "train_lidar") for e in splits[k]}
    scenes = yaml.safe_load(Path(a.list).read_text())["scenes"]
    bad = [s["video_id"] for s in scenes if str(s["visit_id"]) in used]
    if bad:
        sys.exit(f"REFUSED: videos {bad} share a visit with the fine-tuning splits")
    failed = []
    for i, e in enumerate(scenes, 1):
        free_gb = shutil.disk_usage(root).free / 2**30
        if free_gb < 4:
            sys.exit(f"stopping: only {free_gb:.1f} GB free")
        print(f"[{i}/{len(scenes)}] {e['video_id']} (free {free_gb:.1f} GB)", flush=True)
        try:
            download(root, e, a.tol, a.faro_every)
        except Exception as ex:  # noqa: BLE001
            print(f"  FAILED {e['video_id']}: {ex}", flush=True)
            failed.append(e["video_id"])
    print(f"done, failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
