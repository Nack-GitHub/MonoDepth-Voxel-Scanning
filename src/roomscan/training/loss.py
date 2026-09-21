"""Scale-invariant log loss (ZoeDepth / DA-v2 metric form) + optional L1, computed per image.

    g = log(pred) - log(gt) over valid pixels;  silog = scale * sqrt(var(g) + beta * mean(g)^2)

beta < 1 keeps a penalty on the global log-scale error mean(g) — exactly the per-frame scale
wobble Exp1 found — while var(g) scores shape. Computed per image, then averaged over images
that have any valid pixel (so one frame's scale error cannot be cancelled by another's).
Always evaluated in float32, even under bf16 autocast.
"""

from __future__ import annotations

import torch

MIN_PRED = 1e-3    # metres; clamp before log
_EPS = 1e-12       # keeps sqrt differentiable at a perfect prediction


def _per_image(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pred, target = pred.float(), target.float()
    if pred.dim() == 2:
        pred, target, mask = pred[None], target[None], mask[None]
    return pred, target, mask & (target > 0)


def silog(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.15,
          scale: float = 10.0) -> torch.Tensor:
    pred, target, mask = _per_image(pred, target, mask)
    losses = []
    for p, t, m in zip(pred, target, mask, strict=True):
        n = m.sum()
        if n == 0:
            continue
        g = torch.log(p[m].clamp_min(MIN_PRED)) - torch.log(t[m])
        mean = g.mean()
        var = (g - mean).pow(2).mean()
        losses.append(scale * torch.sqrt(var + beta * mean.pow(2) + _EPS) - scale * _EPS ** 0.5)
    if not losses:
        return pred.sum() * 0.0     # keeps the graph; no NaN on an all-invalid batch
    return torch.stack(losses).mean()


def l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred, target, mask = _per_image(pred, target, mask)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return (pred[mask] - target[mask]).abs().mean()


class DepthLoss:
    """silog (+ lambda_l1 * l1). Defaults match configs/training/base.yaml."""

    def __init__(self, beta: float = 0.15, scale: float = 10.0, lambda_l1: float = 0.0):
        self.beta, self.scale, self.lambda_l1 = float(beta), float(scale), float(lambda_l1)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = silog(pred, target, mask, self.beta, self.scale)
        if self.lambda_l1 > 0:
            loss = loss + self.lambda_l1 * l1(pred, target, mask)
        return loss
