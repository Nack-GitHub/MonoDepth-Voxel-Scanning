"""Upload a run's checkpoint (best/ by default) to a PRIVATE HuggingFace Hub repo (ADR-013).

    python -m roomscan.training.push --run experiments/training/r1_ft_faro \
        --repo <hf_user>/roomscan-dav2-metric-large-ft-faro --private [--which best|last] [--dry-run]

Auth comes from `hf auth login` (token in ~/.cache/huggingface) — never from a file in this repo.
Writes a short model card (run, split seed, val metrics, config hash) next to the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_log(run: Path) -> list[dict]:
    path = run / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def model_card(run: Path, which: str, repo: str) -> str:
    rows = _read_log(run)
    env = next((r for r in rows if r.get("type") == "env"), {})
    vals = [r for r in rows if r.get("type") == "val"]
    base = next((r for r in vals if r.get("step") == 0), {})
    ckpt_path = run / which / "checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.is_file() else {}
    cfg = (run / "config.yaml").read_text() if (run / "config.yaml").is_file() else ""
    lines = [
        "---", "library_name: transformers", "pipeline_tag: depth-estimation",
        "base_model: " + str(env.get("hf_id", "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")), "---", "",
        f"# {repo.split('/')[-1]}", "",
        "Depth Anything V2 Metric-Indoor fine-tuned on ARKitScenes (roomscan, ADR-013): frozen DINOv2 backbone, "
        f"DPT neck + head trained with SiLog on the **{env.get('target', '?')}** teacher.", "",
        "| | |", "|---|---|",
        f"| run | `{run.name}` ({which}/, step {ckpt.get('step', '?')}) |",
        f"| train sets | {env.get('train_sets')} — {env.get('n_train_scenes')} scenes, "
        f"{env.get('n_train_frames')} frames |",
        f"| val (ARKitScenes Validation fold, Faro) | abs_rel {ckpt.get('val/abs_rel', '?')} "
        f"(pretrained step 0: {base.get('val/abs_rel', '?')}) |",
        f"| config sha1 | `{hashlib.sha1(cfg.encode()).hexdigest()[:12]}` |", "",
        "Test scenes (the six Exp1 scenes) were never seen during training, validation or checkpoint choice.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiments/training/<run>")
    ap.add_argument("--repo", required=True, help="<hf_user>/<name>")
    ap.add_argument("--which", choices=("best", "last"), default="best")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no network")
    a = ap.parse_args()

    run = Path(a.run)
    folder = run / a.which
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
    if not files:
        raise SystemExit(f"nothing to upload: {folder} is missing or empty")
    card = model_card(run, a.which, a.repo)
    if a.dry_run:
        print(f"would create PRIVATE repo {a.repo} and upload {folder}:")
        for p in files:
            print(f"  {p.relative_to(folder)}  {p.stat().st_size / 2**20:.1f} MB")
        print("  README.md (model card)\n" + card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, private=True, exist_ok=True, repo_type="model")
    (folder / "README.md").write_text(card)
    api.upload_folder(repo_id=a.repo, folder_path=str(folder), repo_type="model",
                      commit_message=f"{run.name}: {a.which}/ checkpoint")
    print(f"uploaded {folder} -> https://huggingface.co/{a.repo} (private)")


if __name__ == "__main__":
    main()
