"""`roomscan` command line.

    roomscan run   --config configs/depth/gt.yaml [--set k=v ...]
    roomscan sweep configs/experiments/exp1_depth_source.yaml
    roomscan report experiments/results
    roomscan eval2d   configs/experiments/exp7_valfold_2d.yaml   # 2D only, no mesh (ADR-014)
    roomscan reeval2d experiments/results/exp6_finetune/*/       # recompute metrics_2d in place
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="roomscan")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one scene, one config")
    r.add_argument("--config", required=True)
    r.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE")

    s = sub.add_parser("sweep", help="all runs x all scenes in an experiment file")
    s.add_argument("experiment_file")
    s.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    s.add_argument("--skip-existing", action="store_true")
    s.add_argument("--scenes", nargs="+", default=None, metavar="SCENE",
                   help="subset of the experiment's scenes to run (default: all)")

    e2 = sub.add_parser("eval2d", help="2D depth metrics only (no fusion / reference mesh), ADR-014")
    e2.add_argument("experiment_file")
    e2.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE")
    e2.add_argument("--skip-existing", action="store_true")
    e2.add_argument("--scenes", nargs="+", default=None, metavar="SCENE")

    r2 = sub.add_parser("reeval2d", help="recompute metrics_2d of finished runs with the current protocol")
    r2.add_argument("run_dirs", nargs="+")
    r2.add_argument("--force", action="store_true", help="also runs already at the current protocol")

    rp = sub.add_parser("report", help="aggregate metrics.json files into tables")
    rp.add_argument("results_root", nargs="?", default="experiments/results")

    args = p.parse_args(argv)
    return {"run": _run, "sweep": _sweep, "report": _report, "eval2d": _eval2d,
            "reeval2d": _reeval2d}[args.cmd](args)


def _run(args) -> int:
    from roomscan.config import load_config
    from roomscan.pipeline import ReconstructionPipeline

    cfg = load_config(args.config, args.overrides)
    result = ReconstructionPipeline(cfg).run()
    print(f"[{result.run_name}] frames={result.n_frames} total={result.timing.total:.1f}s "
          f"-> {result.out_dir}")
    if result.metrics_3d:
        m = result.metrics_3d.to_flat_dict()
        print(f"  chamfer={m['chamfer']*100:.2f}cm  f@0.05={m.get('fscore@0.05', float('nan')):.3f}")
    return 0


def _sweep(args) -> int:
    from roomscan.config import apply_overrides, load_config
    from roomscan.pipeline import ReconstructionPipeline

    exp = OmegaConf.load(args.experiment_file)
    base = load_config(exp.base, args.overrides)
    scenes = [str(x) for x in exp.scenes]
    if args.scenes:
        unknown = set(args.scenes) - set(scenes)
        if unknown:
            raise SystemExit(f"--scenes not in {args.experiment_file}: {sorted(unknown)}")
        scenes = args.scenes
    for scene in scenes:
        for run in exp.runs:
            cfg = apply_overrides(base, dict(run.overrides))
            cfg.dataset.scene = scene
            cfg.output.experiment = exp.experiment
            cfg.output.run_name = f"{scene}_{run.name}"
            out = Path(cfg.output.root) / exp.experiment / cfg.output.run_name
            if args.skip_existing and (out / "metrics.json").exists():
                print(f"skip {out}")
                continue
            print(f"=== {exp.experiment} / {scene} / {run.name} ===")
            ReconstructionPipeline(cfg).run()
    return 0


def _eval2d(args) -> int:
    from roomscan.eval2d import run_2d_experiment

    run_2d_experiment(args.experiment_file, overrides=args.overrides, scenes=args.scenes,
                      skip_existing=args.skip_existing)
    return 0


def _reeval2d(args) -> int:
    from roomscan.eval2d import reevaluate_runs

    n = reevaluate_runs(args.run_dirs, force=args.force, log=lambda s: print(s, flush=True))
    print(f"re-evaluated {n} runs")
    return 0


def _report(args) -> int:
    from roomscan.evaluation.report import write_tables

    write_tables(args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
