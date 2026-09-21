"""Pick the visit-disjoint fine-tuning splits (ADR-013) -> configs/training/splits.yaml.

    python scripts/select_training_scenes.py --metadata data/arkitscenes/raw/metadata.csv \
        --train-candidates data/arkitscenes/candidates_train_seed1.csv \
        --val-candidates data/arkitscenes/candidates_val_seed1.csv \
        --exclude-test configs/experiments/exp1_depth_source.yaml \
        --n-faro 10 --n-lidar-only 14 --n-val 3 --seed 0 --out configs/training/splits.yaml

  T test        = the Exp1 scenes (from --exclude-test), mapped to visit_id through metadata.csv
  A train_faro  = Training fold, Faro used      (scan 60-120 s, hi_MB >= 100)
  B train_lidar = Training fold, LiDAR only     (scan 60-120 s; highres_depth is not downloaded)
  V val         = Validation fold, Faro used    (scan 60-120 s, hi_MB >= 100)

Candidates come from scripts/screen_scenes.py (one video per visit). Any candidate whose visit_id
belongs to T is dropped; no visit may appear in two sets. Exits non-zero if a set cannot be filled.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import yaml

# visit_ids of the six Exp1 test scenes (SPEC §5.1) — cross-checked against metadata.csv at run time
EXPECTED_TEST_VISITS = {"421069", "467370", "470046", "423306", "471364", "467324"}


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _eligible(row: dict, *, min_s: float, max_s: float, min_hi_mb: int | None) -> bool:
    try:
        ok = min_s <= float(row["traj_s"]) <= max_s
        if min_hi_mb is not None:
            ok &= int(row["hi_MB"]) >= min_hi_mb
        return ok
    except (KeyError, ValueError):
        return False


def _entry(row: dict, meta: dict[str, dict], fold: str, has_faro: bool) -> dict:
    m = meta.get(row["video_id"], {})
    return {"video_id": str(row["video_id"]), "visit_id": str(row["visit_id"]), "fold": fold,
            "has_faro": has_faro, "sky_direction": m.get("sky_direction") or "Up",
            "traj_s": float(row["traj_s"]), "hi_MB": int(row["hi_MB"]),
            "vga_MB": int(row.get("vga_MB") or 0), "ld_MB": int(row.get("ld_MB") or 0),
            "conf_MB": int(row.get("conf_MB") or 0)}


def download_mb(e: dict) -> int:
    """MB fetched for one scene by download_training_scenes.py (zips; unpacked size is similar)."""
    return e["vga_MB"] + e["ld_MB"] + e["conf_MB"] + (e["hi_MB"] if e["has_faro"] else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/arkitscenes/raw/metadata.csv")
    ap.add_argument("--train-candidates", required=True)
    ap.add_argument("--val-candidates", required=True)
    ap.add_argument("--exclude-test", default="configs/experiments/exp1_depth_source.yaml")
    ap.add_argument("--n-faro", type=int, default=10)
    ap.add_argument("--n-lidar-only", type=int, default=14)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=60.0)      # seconds of scan
    ap.add_argument("--max-s", type=float, default=120.0)
    ap.add_argument("--min-hi-mb", type=int, default=100)     # MB of highres_depth.zip
    ap.add_argument("--out", default="configs/training/splits.yaml")
    a = ap.parse_args()

    meta = {r["video_id"]: r for r in _read_csv(a.metadata)}
    test_ids = [str(s) for s in yaml.safe_load(Path(a.exclude_test).read_text())["scenes"]]
    missing = [v for v in test_ids if v not in meta]
    if missing:
        sys.exit(f"test scenes not in metadata: {missing}")
    test = [{"video_id": v, "visit_id": meta[v]["visit_id"], "fold": meta[v]["fold"], "has_faro": True,
             "sky_direction": meta[v].get("sky_direction") or "Up"} for v in test_ids]
    test_visits = {t["visit_id"] for t in test}
    if test_visits != EXPECTED_TEST_VISITS:
        sys.exit(f"test visit_ids {sorted(test_visits)} != SPEC §5.1 {sorted(EXPECTED_TEST_VISITS)}")

    rng = random.Random(a.seed)
    used = set(test_visits)

    def pick(rows: list[dict], n: int, min_hi_mb: int | None, fold: str, has_faro: bool, name: str) -> list[dict]:
        rows = sorted(rows, key=lambda r: r["video_id"])       # deterministic before shuffling
        rng.shuffle(rows)
        out = []
        for r in rows:
            if len(out) == n:
                break
            if r["visit_id"] in used or meta.get(r["video_id"], {}).get("fold") != fold:
                continue
            if not _eligible(r, min_s=a.min_s, max_s=a.max_s, min_hi_mb=min_hi_mb):
                continue
            used.add(r["visit_id"])
            out.append(_entry(r, meta, fold, has_faro))
        if len(out) < n:
            sys.exit(f"{name}: only {len(out)} eligible candidates, need {n} — screen more (--n)")
        return out

    train_rows = _read_csv(a.train_candidates)
    val_rows = _read_csv(a.val_candidates)
    train_faro = pick(train_rows, a.n_faro, a.min_hi_mb, "Training", True, "train_faro (A)")
    train_lidar = pick(train_rows, a.n_lidar_only, None, "Training", False, "train_lidar (B)")
    val = pick(val_rows, a.n_val, a.min_hi_mb, "Validation", True, "val (V)")

    visits = [e["visit_id"] for s in (test, val, train_faro, train_lidar) for e in s]
    assert len(visits) == len(set(visits)), "visit_id overlap between sets"

    splits = {
        "generated_by": "scripts/select_training_scenes.py " + " ".join(sys.argv[1:]),
        "seed": a.seed,
        "criteria": {"scan_s": [a.min_s, a.max_s], "min_hi_MB_faro": a.min_hi_mb,
                     "disjoint_on": "visit_id", "train_fold": "Training", "val_fold": "Validation"},
        "test": test,
        "val": val,
        "train_faro": train_faro,
        "train_lidar": train_lidar,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Fine-tuning splits (ADR-013). Frozen once R1 starts — changing it needs a new ADR.\n"
              "# test = Exp1 scenes: evaluated on the Mac only, never downloaded to the training machine.\n")
    out.write_text(header + yaml.safe_dump(splits, sort_keys=False))

    gb = {k: sum(download_mb(e) for e in v) / 1024 for k, v in
          (("val", val), ("train_faro", train_faro), ("train_lidar", train_lidar))}
    print(f"wrote {out}: test {len(test)} / val {len(val)} / train_faro {len(train_faro)} / "
          f"train_lidar {len(train_lidar)}; download ≈ " +
          ", ".join(f"{k} {v:.1f} GB" for k, v in gb.items()) + f" (total {sum(gb.values()):.1f} GB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
