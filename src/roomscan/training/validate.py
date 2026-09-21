"""Validation on the V scenes with the paper's own 2D metrics (evaluation/metrics_2d.py, not re-derived).

Whole frame (no crop), upright, short side = 518; the prediction is resized back to the label grid
with bilinear interpolation — the same way models/depth_anything.py does at inference — and scored
against Faro depth with depth_metrics(max_depth=5.0 m by default, as in the paper tables).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from roomscan.dataio.arkitscenes import ARKitScenesScene
from roomscan.evaluation.metrics_2d import depth_metrics, mean_metrics
from roomscan.training.pairs import iter_indices, make_pair
from roomscan.training.transforms import EvalTransform, upright


def predict_depth(model, image: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """(B, 3, h, w) normalised -> (B, H, W) float32 metres at out_hw (bilinear)."""
    pred = model(pixel_values=image).predicted_depth
    if pred.dim() == 4:
        pred = pred[:, 0]
    if tuple(pred.shape[-2:]) != tuple(out_hw):
        pred = F.interpolate(pred[:, None].float(), size=out_hw, mode="bilinear", align_corners=False)[:, 0]
    return pred.float()


@torch.no_grad()
def validate(model, scenes: Sequence[ARKitScenesScene], *, device: str, stride: int = 5, size: int = 518,
             label_max_depth: float = 10.0, metric_max_depth: float = 5.0, bf16: bool = False) -> dict:
    was_training = model.training
    model.eval()
    transform = EvalTransform(size)
    t0 = time.perf_counter()
    items, per_scene = [], {}
    for scene in scenes:
        scene_items = []
        for idx in iter_indices(scene, stride):
            pair = make_pair(scene, idx, target="faro", max_depth=label_max_depth)
            out = transform(pair)
            gt = upright(pair.rgb, pair.depth, pair.mask, pair.sky_direction)[1]
            image = torch.from_numpy(out["image"])[None].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                pred = predict_depth(model, image, gt.shape)
            scene_items.append(depth_metrics(pred[0].cpu().numpy().astype(np.float32), gt,
                                             max_depth=metric_max_depth))
        items += scene_items
        per_scene[scene.scene_id] = mean_metrics(scene_items).abs_rel
    if was_training:
        model.train()
    m = mean_metrics(items)
    return {"val/abs_rel": m.abs_rel, "val/delta1": m.delta1, "val/rmse": m.rmse, "val/n_frames": len(items),
            "val/n_pixels": m.n_valid, "val/sec": round(time.perf_counter() - t0, 2),
            "val/abs_rel_per_scene": per_scene}
