"""torch Dataset over several ARKitScenes scenes: index -> {image, depth, mask, scene, idx}.

Augmentation randomness is drawn from torch's RNG (seeded per DataLoader worker), so runs are
reproducible under torch.manual_seed and differ between epochs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from roomscan.dataio.arkitscenes import ARKitScenesScene
from roomscan.training.pairs import iter_indices, make_pair


class DepthPairDataset(Dataset):
    def __init__(self, scenes: Sequence[ARKitScenesScene], *, target: str, transform: Callable,
                 stride: int = 1, max_depth: float = 10.0, lidar_min_confidence: int = 1):
        self.scenes = list(scenes)
        self.target = target
        self.transform = transform
        self.max_depth = float(max_depth)                  # metres
        self.lidar_min_confidence = int(lidar_min_confidence)
        self.items = [(s, i) for s, scene in enumerate(self.scenes) for i in iter_indices(scene, stride)]

    def __len__(self) -> int:
        return len(self.items)

    def frames_per_scene(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s, _ in self.items:
            key = self.scenes[s].scene_id
            out[key] = out.get(key, 0) + 1
        return out

    def __getitem__(self, i: int) -> dict:
        s, idx = self.items[i]
        scene = self.scenes[s]
        pair = make_pair(scene, idx, target=self.target, max_depth=self.max_depth,
                         lidar_min_confidence=self.lidar_min_confidence)
        rng = np.random.default_rng([int(torch.randint(0, 2**31 - 1, (1,)).item()), i])
        out = self.transform(pair, rng)
        return {"image": torch.from_numpy(out["image"]), "depth": torch.from_numpy(out["depth"]),
                "mask": torch.from_numpy(out["mask"]), "scene": scene.scene_id, "idx": idx}
