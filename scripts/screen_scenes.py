"""Screen ARKitScenes candidates WITHOUT downloading them.

For each eligible video (Training, highres_depth + laser scan + 3DOD) it HEADs the
asset zips for their sizes and fetches only the tiny lowres_wide.traj (scan duration)
and 3DOD annotation (object labels -> room type hint). ~6 requests per video.

    python scripts/screen_scenes.py --n 40 --seed 0 > candidates.csv

Reading the output:
  * hi_MB ~= 0.8 MB per highres (Faro) frame -> under ~80 MB the reference mesh will be
    hollow; prefer >= 100 MB.
  * total_MB is what `download_data.py raw` with the assets in data/README.md will use.
  * one video per visit_id so candidates are different homes/rooms.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
import urllib.request

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training"
ZIPS = ("highres_depth", "lowres_depth", "vga_wide", "lowres_wide", "confidence")


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as r:
        return int(r.headers["Content-Length"])


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/arkitscenes/raw/metadata.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video_id", nargs="*", help="screen exactly these instead of sampling")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.metadata))
            if r["fold"] == "Training" and r["is_in_upsampling"] == "True"
            and r["has_laser_scanner_point_clouds"] == "True" and r["is_in_threedod"] == "True"]
    if a.video_id:
        cands = [r for r in rows if r["video_id"] in set(a.video_id)]
    else:
        by_visit: dict[str, dict] = {}
        for r in rows:
            by_visit.setdefault(r["visit_id"], r)
        random.seed(a.seed)
        cands = random.sample(list(by_visit.values()), min(a.n, len(by_visit)))
    print(f"{len(rows)} eligible videos, screening {len(cands)}", file=sys.stderr)

    out = csv.writer(sys.stdout)
    out.writerow(["video_id", "visit_id", "traj_s", "traj_rows", "hi_MB", "total_MB", "objects"])
    for r in cands:
        v = r["video_id"]
        u = f"{BASE}/{v}"
        try:
            traj = _get(f"{u}/lowres_wide.traj").decode().strip().splitlines()
            ts = [float(line.split()[0]) for line in traj]
            ann = json.loads(_get(f"{u}/{v}_3dod_annotation.json"))
            labels = collections.Counter(o["label"] for o in ann.get("data", []))
            sizes = {z: _head_size(f"{u}/{z}.zip") for z in ZIPS}
            sizes["mesh"] = _head_size(f"{u}/{v}_3dod_mesh.ply")
            out.writerow([v, r["visit_id"], round(ts[-1] - ts[0], 1), len(traj),
                          sizes["highres_depth"] >> 20, sum(sizes.values()) >> 20,
                          " ".join(f"{k}:{n}" for k, n in labels.most_common())])
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001 — keep screening the rest
            print(v, "ERR", e, file=sys.stderr)


if __name__ == "__main__":
    main()
