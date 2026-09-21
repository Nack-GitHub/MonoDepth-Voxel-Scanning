"""Download the fine-tuning scenes listed in configs/training/splits.yaml (ADR-013), or verify them.

    python scripts/download_training_scenes.py --splits configs/training/splits.yaml --root data/arkitscenes
    python scripts/download_training_scenes.py --splits configs/training/splits.yaml --root data/arkitscenes \
        --verify --md experiments/training/DATA.md

Same CDN URLs as Apple's download_data.py (raw/<fold>/<video_id>/<asset>.zip). Per scene it fetches
vga_wide, lowres_depth, confidence, lowres_wide_intrinsics (+ highres_depth when has_faro) and
lowres_wide.traj; never `color` 1920x1440. Each zip is unpacked into raw/<fold>/<video_id>/ and deleted;
a `.<asset>.ok` marker makes re-runs skip finished assets (idempotent). Order: val, train_faro,
train_lidar — the first two are all R0-R2 need.

The `test` set is NEVER downloaded; both modes refuse any video whose visit_id belongs to it, and
--verify fails if a folder of such a visit exists anywhere under raw/.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ZIP_ASSETS = ("vga_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics")
SETS = ("val", "train_faro", "train_lidar")      # download order


def load_splits(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def training_entries(splits: dict) -> list[tuple[str, dict]]:
    test_visits = {e["visit_id"] for e in splits["test"]}
    out = []
    for name in SETS:
        for e in splits.get(name) or []:
            if e["visit_id"] in test_visits:
                sys.exit(f"REFUSED: {name} video {e['video_id']} has test visit {e['visit_id']}")
            out.append((name, e))
    return out


def _curl(url: str, dst: Path) -> None:
    part = dst.with_name(dst.name + ".part")
    subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(part), url],
                   check=True)
    part.rename(dst)


def _unzip(zpath: Path, scene_dir: Path, asset: str) -> None:
    """Unpack into scene_dir/<asset>/ whether or not the archive has the asset folder at its root."""
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        prefixed = all(n.startswith(f"{asset}/") for n in names)
        z.extractall(scene_dir if prefixed else scene_dir / asset)


def download_scene(root: Path, e: dict, log=print) -> None:
    vid, fold = str(e["video_id"]), e["fold"]
    scene_dir = root / "raw" / fold / vid
    scene_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{fold}/{vid}"
    traj = scene_dir / "lowres_wide.traj"
    if not traj.is_file():
        _curl(f"{url}/lowres_wide.traj", traj)
    assets = list(ZIP_ASSETS) + (["highres_depth"] if e["has_faro"] else [])
    for asset in assets:
        marker = scene_dir / f".{asset}.ok"
        if marker.is_file():
            continue
        zpath = scene_dir / f"{asset}.zip"
        if not zpath.is_file():
            _curl(f"{url}/{asset}.zip", zpath)
        shutil.rmtree(scene_dir / asset, ignore_errors=True)   # a half-unpacked previous attempt
        _unzip(zpath, scene_dir, asset)
        zpath.unlink()
        marker.touch()
    log(f"ok {fold}/{vid} ({', '.join(assets)})")


def _du_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20


def _metadata_visits(root: Path) -> dict[str, str]:
    meta = root / "raw" / "metadata.csv"
    with meta.open(newline="") as f:
        return {r["video_id"]: r["visit_id"] for r in csv.DictReader(f)}


def upright_gravity_score(scene) -> float:
    """Mean world-z of the image-down axis after the training upright rotation; about -1 when the frames
    the model sees are upright (world is z-up), > 0 when they are upside-down."""
    import numpy as np

    from roomscan.training.transforms import upright_k

    # image-down direction in camera coords after np.rot90(k): k=0 +y, k=1 -x, k=2 -y, k=3 +x
    down = {0: (0, 1, 0), 1: (-1, 0, 0), 2: (0, -1, 0), 3: (1, 0, 0)}[upright_k(scene.sky_direction) % 4]
    ts = scene.timestamps[:: max(1, len(scene) // 40)]
    rots = np.stack([scene._pose_at(t)[:3, :3] for t in ts])
    return float(np.mean((rots @ np.array(down, dtype=np.float64))[:, 2]))


def verify(root: Path, splits: dict, md_out: Path | None) -> int:
    from roomscan.training.pairs import open_training_scene

    errors: list[str] = []
    test_visits = {e["visit_id"] for e in splits["test"]}
    visits = _metadata_visits(root)
    for d in sorted((root / "raw").glob("*/*")):
        if d.is_dir() and visits.get(d.name) in test_visits:
            errors.append(f"TEST SCENE ON DISK: {d} (visit {visits[d.name]})")
    leftovers = sorted(str(p) for p in (root / "raw").rglob("*.zip*"))
    if leftovers:
        errors.append(f"{len(leftovers)} zip files left over, e.g. {leftovers[0]}")

    rows = []
    for name, e in training_entries(splits):
        vid, fold = str(e["video_id"]), e["fold"]
        scene_dir = root / "raw" / fold / vid
        try:
            vga = open_training_scene(root, vid, fold=fold, target="lidar")
            n_vga, n_faro, unmatched = len(vga), 0, 0
            g = upright_gravity_score(vga)
            sky = f"{vga.sky_direction} (g {g:+.2f})"
            if g > -0.3:
                errors.append(f"{vid}: frames not upright after sky={vga.sky_direction} rotation (g {g:+.2f})")
            if e["has_faro"]:
                faro = open_training_scene(root, vid, fold=fold, target="faro")
                n_faro = len(faro)
                vga_idx = faro._idx["vga_wide"]
                unmatched = sum(vga_idx.nearest(ts, faro.tol) is None for ts in faro.timestamps)
                if n_faro == 0:
                    errors.append(f"{vid}: 0 Faro frames")
                elif unmatched / n_faro > 0.05:
                    errors.append(f"{vid}: {unmatched}/{n_faro} Faro frames have no vga_wide match")
            if n_vga == 0:
                errors.append(f"{vid}: 0 vga_wide frames with a pose")
        except (FileNotFoundError, ValueError) as ex:
            errors.append(f"{vid}: {ex}")
            n_vga = n_faro = unmatched = 0
            sky = "?"
        rows.append((name, vid, e["visit_id"], fold, n_faro, unmatched, n_vga, sky,
                     _du_mb(scene_dir) if scene_dir.is_dir() else 0.0))

    total_gb = _du_mb(root) / 1024
    lines = ["| set | video_id | visit_id | fold | Faro frames | Faro w/o vga | vga frames | sky (upright g) | MB |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]:.0f} |" for r in rows]
    for name in SETS:
        sel = [r for r in rows if r[0] == name]
        lines.append(f"| **{name}** | {len(sel)} scenes | | | {sum(r[4] for r in sel)} | "
                     f"{sum(r[5] for r in sel)} | {sum(r[6] for r in sel)} | | {sum(r[8] for r in sel):.0f} |")
    table = "\n".join(lines)
    print(table)
    print(f"\ntotal on disk under {root}: {total_gb:.1f} GB")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text("# Fine-tuning data on the training machine (generated by "
                          "`scripts/download_training_scenes.py --verify`)\n\n"
                          f"Splits: `configs/training/splits.yaml` (seed {splits.get('seed')}). "
                          "Test scenes are not on this machine (checked by visit_id). "
                          "`g` = world-z of the image-down axis after the training upright rotation "
                          "(≈ -1 upright, > 0 upside-down; fails below -0.3).\n\n"
                          f"{table}\n\nTotal on disk: **{total_gb:.1f} GB**\n")
    for err in errors:
        print("ERROR", err, file=sys.stderr)
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="configs/training/splits.yaml")
    ap.add_argument("--root", default="data/arkitscenes")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--md", default=None, help="with --verify: also write the table to this markdown file")
    ap.add_argument("--sets", nargs="*", default=list(SETS), choices=SETS)
    ap.add_argument("--jobs", type=int, default=3, help="scenes downloaded in parallel")
    a = ap.parse_args()

    root = Path(a.root)
    splits = load_splits(a.splits)
    if a.verify:
        sys.exit(verify(root, splits, Path(a.md) if a.md else None))

    entries = [e for name, e in training_entries(splits) if name in a.sets]
    failed = []

    def job(e: dict) -> None:
        try:
            download_scene(root, e, log=lambda m: print(m, flush=True))
        except (subprocess.CalledProcessError, zipfile.BadZipFile, OSError) as ex:
            print(f"FAILED {e['video_id']}: {ex}", flush=True)
            failed.append(e["video_id"])

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        list(pool.map(job, entries))
    print(f"done: {len(entries) - len(failed)}/{len(entries)} scenes" + (f"; failed {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
