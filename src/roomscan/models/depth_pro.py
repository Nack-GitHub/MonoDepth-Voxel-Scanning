"""Depth Pro (Apple) — the intrinsics-conditioned metric baseline (Level 0 of ADR-013).

Unlike DA-v2 Metric it does not just regress metres: it predicts the field of view as well and
divides by the resulting focal length, which is the standard answer to ``why not use a model that
knows the camera?''. Inference only; RGB in, metres out, so it drops into MonocularDepth like any
other model. A scene's real focal length can be passed in (`focal_px`) to use the camera's own
intrinsics instead of the predicted ones.
"""

from __future__ import annotations

import numpy as np

from roomscan.models.base import DepthModel, pick_torch_device


class DepthPro(DepthModel):
    is_metric = True
    output_kind = "depth"

    def __init__(self, hf_id: str = "apple/DepthPro-hf", device: str = "auto", focal_px: float | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.name = "depth_pro"
        self.device = pick_torch_device(device)
        self.focal_px = focal_px
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        # fp16 on MPS/CUDA: the 1536x1536 backbone needs ~4 GB in fp32 and the head is scale-invariant
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id, dtype=dtype).to(self.device).eval()
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self._torch = torch

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.model.dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with torch.no_grad():
            out = self.model(**inputs)
        kwargs = {"target_sizes": [(h, w)]}
        if self.focal_px is not None:
            kwargs["target_sizes"] = [(h, w)]
            out.field_of_view = torch.full_like(
                out.field_of_view, float(np.degrees(2 * np.arctan(w / (2 * self.focal_px)))))
        depth = self.processor.post_process_depth_estimation(out, **kwargs)[0]["predicted_depth"]
        depth = depth.float().cpu().numpy().astype(np.float32)
        return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
