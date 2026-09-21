"""Fine-tune DA-v2 metric (frozen DINOv2 backbone; DPT neck + head trained) on ARKitScenes (ADR-013).

    python -m roomscan.training.train --config configs/training/r1_ft_faro.yaml [--set train.max_steps=30 ...]

Writes experiments/training/<run_name>/:
    config.yaml   resolved config (commit)
    log.jsonl     {"type": "env"} first, then "val" (step 0 = pretrained baseline), "train" per optimizer
                  step, "val" every val_every steps, and a final "summary" (peak VRAM, s/step) (commit)
    best/         save_pretrained + image processor at the lowest val/abs_rel (never committed)
    last/         same, every save_every steps and at the end (never committed)
One step = one optimizer update = batch_size x grad_accum samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from roomscan.config import load_config, to_yaml
from roomscan.dataio.arkitscenes import ARKitScenesScene
from roomscan.training.dataset import DepthPairDataset
from roomscan.training.loss import DepthLoss
from roomscan.training.pairs import open_training_scene
from roomscan.training.transforms import TrainTransform
from roomscan.training.validate import predict_depth, validate


# ---------------------------------------------------------------- setup helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_splits(path: str | Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def scene_entries(splits: dict, set_names: list[str]) -> list[dict]:
    test_visits = {str(e["visit_id"]) for e in splits.get("test") or []}
    out = []
    for name in set_names:
        if name == "test":
            raise ValueError("the test set is never used for training or validation (ADR-013)")
        for e in splits.get(name) or []:
            if str(e["visit_id"]) in test_visits:
                raise ValueError(f"{name} video {e['video_id']} shares visit {e['visit_id']} with test")
            out.append(e)
    return out


def open_scenes(root: str, entries: list[dict], target: str, limit: int | None = None) -> list[ARKitScenesScene]:
    entries = entries[:limit] if limit else entries
    return [open_training_scene(root, str(e["video_id"]), fold=e["fold"], target=target) for e in entries]


def freeze_backbone(model) -> None:
    model.backbone.requires_grad_(False)
    model.backbone.eval()


def set_train_mode(model, frozen_backbone: bool) -> None:
    model.train()
    if frozen_backbone:
        model.backbone.eval()   # frozen DINOv2 stays in eval mode (no dropout / drop-path)


def enable_neck_checkpointing(model) -> None:
    """Recompute DPT neck activations in backward (OOM fallback #2, plan T10)."""
    from torch.utils.checkpoint import checkpoint

    neck_forward = model.neck.forward

    def forward(*args, **kwargs):
        return checkpoint(neck_forward, *args, use_reentrant=False, **kwargs)

    model.neck.forward = forward


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine to 0; `step` counts optimizer updates from 0."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def infinite(loader: DataLoader):
    while True:
        yield from loader


def save_checkpoint(model, processor, path: Path, info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if processor is not None:
        processor.save_pretrained(path)
    (path / "checkpoint.json").write_text(json.dumps(info, indent=2) + "\n")


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------- main loop
def train(cfg: DictConfig, *, overwrite: bool = False) -> Path:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    run_dir = Path(cfg.output_dir) / cfg.run_name
    log_path = run_dir / "log.jsonl"
    if log_path.exists() and not overwrite:
        raise FileExistsError(f"{log_path} exists — refusing to write a second run into the same folder "
                              "(pass --overwrite to replace it)")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    cfg_yaml = to_yaml(cfg)
    (run_dir / "config.yaml").write_text(cfg_yaml)
    log = JsonlLog(log_path)

    seed_everything(int(cfg.train.seed))
    device = pick_device(cfg.train.device)
    bf16 = device == "cuda" and cfg.train.precision == "bf16"

    # ---- data
    d = cfg.data
    splits = load_splits(d.splits)
    target = d.target
    train_scenes = open_scenes(d.root, scene_entries(splits, list(d.train_sets)), target, d.max_train_scenes)
    val_scenes = open_scenes(d.root, scene_entries(splits, [d.val_set]), "faro", d.max_val_scenes) \
        if d.val_set else []
    stride = int(d.stride_faro if target == "faro" else d.stride_lidar)
    t = cfg.transform
    dataset = DepthPairDataset(train_scenes, target=target, stride=stride, max_depth=d.max_depth,
                               lidar_min_confidence=d.lidar_min_confidence,
                               transform=TrainTransform(t.size, t.hflip_p, t.jitter))
    tr = cfg.train
    loader = DataLoader(dataset, batch_size=int(tr.batch_size), shuffle=True, drop_last=True,
                        num_workers=int(d.num_workers), pin_memory=device == "cuda",
                        persistent_workers=int(d.num_workers) > 0,
                        generator=torch.Generator().manual_seed(int(tr.seed)))
    if len(loader) == 0:
        raise ValueError(f"not enough training frames ({len(dataset)}) for batch_size {tr.batch_size}")

    # ---- model
    m = cfg.model
    model = AutoModelForDepthEstimation.from_pretrained(m.hf_id)
    try:
        processor = AutoImageProcessor.from_pretrained(m.hf_id)
    except OSError:
        processor = None
    if m.freeze_backbone:
        freeze_backbone(model)
    if m.grad_checkpointing:
        enable_neck_checkpointing(model)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    o = cfg.optim
    optimizer = torch.optim.AdamW(params, lr=float(o.lr), weight_decay=float(o.weight_decay),
                                  betas=tuple(o.betas))
    max_steps = int(tr.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, int(o.warmup_steps), max_steps))
    criterion = DepthLoss(beta=cfg.loss.beta, scale=cfg.loss.scale, lambda_l1=cfg.loss.lambda_l1)

    log.write({
        "type": "env", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": cfg.run_name, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "python": platform.python_version(), "config_sha1": hashlib.sha1(cfg_yaml.encode()).hexdigest()[:12],
        "hf_id": m.hf_id, "target": target, "bf16": bf16,
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in params),
        "train_sets": list(d.train_sets), "n_train_scenes": len(train_scenes), "n_train_frames": len(dataset),
        "train_frames_per_scene": dataset.frames_per_scene(),
        "n_train_frames_skipped_no_label": dataset.n_skipped,
        "n_val_scenes": len(val_scenes), "val_scenes": [s.scene_id for s in val_scenes],
        "effective_batch": int(tr.batch_size) * int(tr.grad_accum),
    })

    v = cfg.val
    frozen = bool(m.freeze_backbone)

    def run_val(step: int) -> dict | None:
        if not val_scenes:
            return None
        row = validate(model, val_scenes, device=device, stride=int(v.stride), size=int(t.size),
                       label_max_depth=float(d.max_depth), metric_max_depth=float(v.max_depth), bf16=bf16)
        set_train_mode(model, frozen)
        if device == "cuda":
            row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log.write({"type": "val", "step": step, **row})
        return row

    # ---- step 0: pretrained baseline on V
    best = math.inf
    base = run_val(0)
    if base is not None and math.isfinite(base["val/abs_rel"]):
        best = base["val/abs_rel"]      # best/ must beat the pretrained model to be written at all

    set_train_mode(model, frozen)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batches = infinite(loader)
    accum = int(tr.grad_accum)
    step_times: list[float] = []
    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            batch = next(batches)
            image = batch["image"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                pred = predict_depth(model, image, depth.shape[-2:])
            loss = criterion(pred, depth, mask) / accum
            loss.backward()
            loss_sum += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(tr.grad_clip))
        optimizer.step()
        scheduler.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        row = {"type": "train", "step": step, "loss": round(loss_sum, 6),
               "lr": optimizer.param_groups[0]["lr"], "grad_norm": round(float(grad_norm), 4),
               "sec_per_step": round(dt, 4)}
        if device == "cuda":
            row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log.write(row)

        last_step = step == max_steps
        if int(v.every) > 0 and (step % int(v.every) == 0 or last_step):
            res = run_val(step)
            if res is not None and res["val/abs_rel"] < best:
                best = res["val/abs_rel"]
                save_checkpoint(model, processor, run_dir / "best",
                                {"step": step, **{k: res[k] for k in ("val/abs_rel", "val/delta1", "val/rmse")}})
        if last_step or (int(tr.save_every) > 0 and step % int(tr.save_every) == 0):
            save_checkpoint(model, processor, run_dir / "last", {"step": step})

    warm = step_times[min(len(step_times) - 1, 5):] or step_times
    summary = {"type": "summary", "steps": max_steps, "sec_per_step_mean": round(float(np.mean(warm)), 4),
               "best_val_abs_rel": best if math.isfinite(best) else None,
               "baseline_val_abs_rel": base["val/abs_rel"] if base else None}
    if device == "cuda":
        summary["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        summary["vram_reserved_peak_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    log.write(summary)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. train.max_steps=30")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    run_dir = train(cfg, overwrite=a.overwrite)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
