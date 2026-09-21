"""Train/eval transforms for DA-v2 fine-tuning (numpy only).

Depth and mask are resampled with NEAREST only and the image scale is never augmented (no zoom,
no random-resized-crop): the target is metric, so a zoom would teach the wrong metres. Allowed
augmentations: random crop at fixed scale, horizontal flip, colour jitter on RGB.
The HF image processor is NOT used here — it resizes keep-aspect on its own and the depth/mask
would no longer line up with the prediction; normalisation is done with the same ImageNet stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roomscan.training.pairs import DepthPair

PATCH = 14   # DINOv2 patch size: every side fed to the model must be a multiple of it
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# k for np.rot90 (counter-clockwise quarter turns) that makes the image upright. Verified against gravity
# from the poses on all 27 fine-tuning scenes (2026-09-21, world is z-up): after rot90(k) the image-down
# axis must point to -z (Up/Down scenes: -0.7..-0.9). "Left" needs a CLOCKWISE turn (k=3; k=1 scored
# +0.7..+0.9 = upside-down on all 12 Left scenes). depth_sources/monocular.py still maps Left->1 —
# reported, not changed here (outside ADR-013 scope). download_training_scenes.py --verify re-checks this.
_ROT = {"Up": 0, "Left": 3, "Down": 2, "Right": 1}

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]   # rgb (H,W,3) uint8, depth (H,W) f32, mask (H,W) bool


def upright_k(sky_direction: str) -> int:
    return _ROT.get(str(sky_direction), 0)


def upright(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, sky_direction: str | int) -> Arrays:
    """Rotate all three arrays so the image is upright. Pass an int k (e.g. -k) to rotate explicitly."""
    k = sky_direction if isinstance(sky_direction, int) else upright_k(sky_direction)
    if k % 4 == 0:
        return rgb, depth, mask
    return (np.ascontiguousarray(np.rot90(rgb, k)), np.ascontiguousarray(np.rot90(depth, k)),
            np.ascontiguousarray(np.rot90(mask, k)))


def resized_shape(h: int, w: int, short: int = 518) -> tuple[int, int]:
    """(H, W) with the short side == `short` and the long side rounded to a multiple of PATCH."""
    if h <= w:
        return short, max(PATCH, int(round(w * short / h / PATCH)) * PATCH)
    return max(PATCH, int(round(h * short / w / PATCH)) * PATCH), short


def resize_short(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, short: int = 518) -> Arrays:
    h, w = depth.shape
    nh, nw = resized_shape(h, w, short)
    if (nh, nw) == (h, w):
        return rgb, depth, mask
    interp = cv2.INTER_LINEAR if nh * nw >= h * w else cv2.INTER_AREA
    rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
    depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
    return rgb, depth, mask


def random_crop(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, size: int,
                rng: np.random.Generator) -> Arrays:
    h, w = depth.shape
    if h < size or w < size:
        raise ValueError(f"crop {size} larger than image {h}x{w}")
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return rgb[y:y + size, x:x + size], depth[y:y + size, x:x + size], mask[y:y + size, x:x + size]


def hflip(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, p: float, rng: np.random.Generator) -> Arrays:
    if rng.random() >= p:
        return rgb, depth, mask
    return rgb[:, ::-1], depth[:, ::-1], mask[:, ::-1]


def color_jitter(rgb: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Brightness / contrast / saturation, each by a factor in [1-strength, 1+strength]. RGB only."""
    if strength <= 0:
        return rgb
    x = rgb.astype(np.float32)
    b, c, s = rng.uniform(1 - strength, 1 + strength, size=3)
    x = x * b
    x = (x - x.mean()) * c + x.mean()
    gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    x = (x - gray[..., None]) * s + gray[..., None]
    return x.clip(0, 255).astype(np.uint8)


def normalize_imagenet(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, H, W) float32 normalised with ImageNet mean/std."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _pack(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {"image": normalize_imagenet(np.ascontiguousarray(rgb)),
            "depth": np.ascontiguousarray(depth, dtype=np.float32),
            "mask": np.ascontiguousarray(mask, dtype=bool)}


@dataclass(frozen=True)
class TrainTransform:
    size: int = 518                # crop side (pixels), multiple of PATCH
    hflip_p: float = 0.5
    jitter: float = 0.2

    def __call__(self, pair: DepthPair, rng: np.random.Generator) -> dict[str, np.ndarray]:
        rgb, depth, mask = upright(pair.rgb, pair.depth, pair.mask, pair.sky_direction)
        rgb, depth, mask = resize_short(rgb, depth, mask, self.size)
        rgb, depth, mask = random_crop(rgb, depth, mask, self.size, rng)
        rgb, depth, mask = hflip(rgb, depth, mask, self.hflip_p, rng)
        return _pack(color_jitter(rgb, self.jitter, rng), depth, mask)


@dataclass(frozen=True)
class EvalTransform:
    """Whole frame, upright, short side = size (no crop, no augmentation)."""
    size: int = 518

    def __call__(self, pair: DepthPair, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        rgb, depth, mask = upright(pair.rgb, pair.depth, pair.mask, pair.sky_direction)
        return _pack(*resize_short(rgb, depth, mask, self.size))
