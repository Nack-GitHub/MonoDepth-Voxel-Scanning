"""Summarise fine-tuning runs from their log.jsonl -> markdown (validation set V only, ADR-013).

    python scripts/summarize_training.py experiments/training/r1_ft_faro experiments/training/r2_ft_lidar \
        experiments/training/r3_ft_lidar_all --out experiments/training/RESULTS.md

Every number here is on V (ARKitScenes Validation fold, Faro labels) and was used to pick checkpoints —
it is NOT a test result. Test numbers come from the Mac-side evaluation on the six Exp1 scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if line.strip()]


def summarise(run: Path) -> dict:
    rows = _rows(run)
    env = next(r for r in rows if r["type"] == "env")
    vals = [r for r in rows if r["type"] == "val"]
    summary = next((r for r in rows if r["type"] == "summary"), {})
    base = next(r for r in vals if r["step"] == 0)
    best = min(vals, key=lambda r: r["val/abs_rel"])
    return {"run": run.name, "target": env["target"], "train_sets": "+".join(env["train_sets"]),
            "scenes": env["n_train_scenes"], "frames": env["n_train_frames"], "steps": summary.get("steps"),
            "base": base, "best": best, "last": vals[-1], "vals": vals,
            "sec_per_step": summary.get("sec_per_step_mean"), "vram": summary.get("vram_reserved_peak_gb")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [summarise(Path(r)) for r in a.runs]
    ref = next((r for r in runs if r["target"] == "faro"), None)

    lines = ["# Fine-tuning results on V (validation, not test)", "",
             "All numbers: ARKitScenes **Validation** fold, 3 scenes, every 5th Faro frame (214 frames), "
             "`metrics_2d.depth_metrics` (0.1–5 m), pixel-weighted mean. V picked the checkpoints, so these are "
             "optimistic; the paper's numbers come from the six held-out Exp1 test scenes (evaluated on the Mac).", "",
             "| run | teacher | train sets | scenes / frames | steps | pretrained abs_rel | best abs_rel (step) "
             "| δ1 @best | RMSE m @best | last abs_rel | best / R1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        b = r["best"]
        ratio = f"{b['val/abs_rel'] / ref['best']['val/abs_rel']:.2f}" if ref else "—"
        lines.append(f"| {r['run']} | {r['target']} | {r['train_sets']} | {r['scenes']} / {r['frames']} | {r['steps']} "
                     f"| {r['base']['val/abs_rel']:.3f} | **{b['val/abs_rel']:.3f}** ({b['step']}) "
                     f"| {b['val/delta1']:.3f} | {b['val/rmse']:.3f} | {r['last']['val/abs_rel']:.3f} | {ratio} |")

    scenes = list(runs[0]["base"]["val/abs_rel_per_scene"])
    lines += ["", "Per V scene, abs_rel at the chosen (best) checkpoint:", "",
              "| run | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
    lines.append("| pretrained | " + " | ".join(f"{runs[0]['base']['val/abs_rel_per_scene'][s]:.3f}"
                                               for s in scenes) + " |")
    for r in runs:
        lines.append(f"| {r['run']} | " + " | ".join(f"{r['best']['val/abs_rel_per_scene'][s]:.3f}"
                                                   for s in scenes) + " |")

    lines += ["", "Validation curve (abs_rel every 500 steps):", "",
              "| step | " + " | ".join(r["run"] for r in runs) + " |", "|---|" + "---|" * len(runs)]
    steps = sorted({v["step"] for r in runs for v in r["vals"]})
    for s in steps:
        cells = []
        for r in runs:
            v = next((x for x in r["vals"] if x["step"] == s), None)
            cells.append(f"{v['val/abs_rel']:.3f}" if v else "")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "Cost: " + ", ".join(f"{r['run']} {r['sec_per_step']} s/step, {r['vram']} GB reserved"
                                      for r in runs) + " (RTX 3070 Ti, bf16, batch 2×4)."]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
