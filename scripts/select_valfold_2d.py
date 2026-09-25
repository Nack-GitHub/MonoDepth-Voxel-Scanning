"""Pick the extra 2D-only test scenes from the ARKitScenes Validation fold (ADR-014) -> configs/eval/valfold_2d.yaml.

    python scripts/select_valfold_2d.py --candidates data/arkitscenes/candidates_valfold2d_seed2.csv \
        --n 20 --seed 0 --out configs/eval/valfold_2d.yaml

Same scan criteria as the fine-tuning val set V (scan 60-120 s, hi_MB >= 100), one video per visit.
Refuses any visit that is in configs/training/splits.yaml (test / val / train) or that also appears in
the Training fold of metadata.csv. These scenes have Faro depth but no reference mesh: 2D only.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--metadata", default="data/arkitscenes/raw/metadata.csv")
    ap.add_argument("--splits", default="configs/training/splits.yaml")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=60.0)
    ap.add_argument("--max-s", type=float, default=120.0)
    ap.add_argument("--min-hi-mb", type=float, default=100.0)
    ap.add_argument("--out", default="configs/eval/valfold_2d.yaml")
    a = ap.parse_args()

    with open(a.metadata, newline="") as f:
        meta = list(csv.DictReader(f))
    fold_of = {r["video_id"]: r["fold"] for r in meta}
    sky_of = {r["video_id"]: r["sky_direction"] for r in meta}
    training_visits = {r["visit_id"] for r in meta if r["fold"] == "Training"}
    splits = yaml.safe_load(Path(a.splits).read_text())
    used = {str(e["visit_id"]) for k in ("test", "val", "train_faro", "train_lidar") for e in splits[k]}

    with open(a.candidates, newline="") as f:
        cands = list(csv.DictReader(f))
    ok, seen = [], set()
    for c in cands:
        v = str(c["visit_id"])
        if fold_of.get(c["video_id"]) != "Validation" or v in used or v in training_visits or v in seen:
            continue
        try:
            if not (a.min_s <= float(c["traj_s"]) <= a.max_s and float(c["hi_MB"]) >= a.min_hi_mb):
                continue
        except (KeyError, ValueError):
            continue
        ok.append(c)
        seen.add(v)
    if len(ok) < a.n:
        sys.exit(f"only {len(ok)} eligible candidates for n={a.n}")
    pick = sorted(random.Random(a.seed).sample(ok, a.n), key=lambda c: c["video_id"])

    out = {
        "generated_by": " ".join(["scripts/select_valfold_2d.py"] + sys.argv[1:]),
        "seed": a.seed,
        "criteria": {"fold": "Validation", "scan_s": [a.min_s, a.max_s], "min_hi_MB": a.min_hi_mb,
                     "disjoint_on": "visit_id", "excluded": "every visit in configs/training/splits.yaml "
                     "and every visit that also appears in the Training fold"},
        "scenes": [{"video_id": c["video_id"], "visit_id": str(c["visit_id"]), "fold": "Validation",
                    "has_faro": True, "sky_direction": sky_of.get(c["video_id"]),
                    "traj_s": float(c["traj_s"]), "hi_MB": float(c["hi_MB"]), "vga_MB": float(c["vga_MB"])}
                   for c in pick],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"{len(ok)} eligible -> {a.n} picked -> {a.out}")


if __name__ == "__main__":
    main()
