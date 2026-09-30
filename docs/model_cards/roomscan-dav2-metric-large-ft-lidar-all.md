---
library_name: transformers
pipeline_tag: depth-estimation
base_model: depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
license: cc-by-nc-4.0
tags:
  - depth-estimation
  - metric-depth
  - monocular-depth
  - indoor
  - arkitscenes
  - 3d-reconstruction
  - depth-anything-v2
---

<h1 align="center">roomscan · DA-v2 Metric Large · FT-LiDAR-24</h1>

<p align="center">
  <b>Metric depth in metres from a single RGB photo of an indoor room.</b><br>
  Taught by iPad LiDAR, measured against a Faro laser scanner. At run time it needs only the RGB image.
</p>

<p align="center">
  <img alt="AbsRel 0.41 → 0.13" src="https://img.shields.io/badge/AbsRel-0.41%20%E2%86%92%200.13-2E7D32">
  <img alt="Chamfer 54 → 16 cm" src="https://img.shields.io/badge/Chamfer-54%20%E2%86%92%2016%20cm-1565C0">
  <img alt="19 of 20 rooms improved" src="https://img.shields.io/badge/unseen%20rooms%20improved-19%2F20-6A1B9A">
  <img alt="Base model Depth Anything V2" src="https://img.shields.io/badge/base-Depth%20Anything%20V2-7B61FF">
  <img alt="Dataset ARKitScenes" src="https://img.shields.io/badge/data-ARKitScenes-555555?logo=apple">
  <img alt="License CC-BY-NC-4.0" src="https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7753049/144107932-39b010fc-6111-4b13-9c68-57dd903d78c5.png" width="100%" alt="ARKitScenes sample: 3D boxes, RGB frame, upsampled depth and laser-scanner depth of a living room">
  <img src="https://user-images.githubusercontent.com/7753049/144108052-6a1d3a67-3948-4ded-bd08-6f1572fdf97a.png" width="100%" alt="Example rooms reconstructed in 3D in ARKitScenes">
  <br>
  <sub>The domain this model was trained on: real homes captured with an iPad Pro in
  <a href="https://github.com/apple-aiml-research/ARKitScenes">ARKitScenes</a>.
  The images are from the ARKitScenes repo (Apple); they are <b>not</b> output of this model.</sub>
</p>

This is [Depth Anything V2 Metric-Indoor Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf)
fine-tuned on real iPad captures from [ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes).
The training labels came only from the **iPad's own LiDAR**; no laser scanner was used to make them.
We measured the result against a **Faro laser scanner**. On rooms the model never saw, the error falls
from **AbsRel 0.41 to 0.13**, and 19 of 20 rooms improve.

It comes from the *roomscan* project, which studies room-scale 3D reconstruction (TSDF fusion) from phone video.
In the paper it is the **FT-LiDAR-24** model (run `r3_ft_lidar_all`).

---

## What problem it solves

The original DA-v2 Metric-Indoor was trained on Hypersim, a set of synthetic indoor scenes. On real phone video
the shapes it predicts are good, but the **scale is off**. It usually predicts depth **1.18–1.45× too far**, so a
room reconstructed from its depth is misplaced by about half a metre (54 cm Chamfer).

Phones with LiDAR can measure depth directly, and we used that as a free teacher.
We fine-tuned the model on LiDAR depth recorded alongside the RGB video from 24 rooms. The fine-tuned model
predicts depth at the right scale from RGB alone, so phones **without** LiDAR can use it too.

## Results

We scored all numbers against Faro laser depth (ARKitScenes `highres_depth`). None of these rooms were used for
training, validation or checkpoint choice.

### 2D depth: 20 unseen rooms (ARKitScenes Validation fold, 2,090 frames)

| Model | AbsRel ↓ | δ<1.25 ↑ | RMSE (cm) ↓ |
|---|---:|---:|---:|
| DA-v2 Metric-Indoor Large (pretrained) | 0.410 ± 0.164 | 0.333 ± 0.216 | 50.9 |
| **This model** | **0.130 ± 0.046** | **0.827 ± 0.135** | **18.4** |
| *iPad LiDAR (reference sensor, not a model)* | *0.019* | *0.997* | *3.8* |

AbsRel improves in 19 of 20 rooms (Wilcoxon signed-rank p = 3.8×10⁻⁶). The one exception is room 48018730, where
AbsRel goes from 0.275 to 0.281. Mean ± std are taken across rooms.

### 2D depth: 6 test rooms (ARKitScenes Training fold, held out from training)

| Model | AbsRel ↓ | δ<1.25 ↑ | RMSE (cm) ↓ |
|---|---:|---:|---:|
| DA-v2 Metric-Indoor Large (pretrained) | 0.382 | 0.277 | 49.3 |
| **This model** | **0.134** | **0.812** | **20.2** |

### 3D room reconstruction: same 6 test rooms

Each depth source is fused into a TSDF with 4 cm voxels. All sources use the same camera poses. The resulting mesh is
compared with a 1 cm reference mesh fused from Faro depth.

| Depth source | Needed at run time | Chamfer (cm) ↓ | F@5cm ↑ |
|---|---|---:|---:|
| DA-v2 Metric-Indoor Large (pretrained) | RGB | 54.3 ± 10.6 | 0.03 |
| **This model** | **RGB** | **15.7 ± 4.0** | **0.24** |
| iPad LiDAR | LiDAR sensor | 3.0 ± 0.6 | 0.93 |
| Faro depth (upper bound) | laser scanner | 2.1 ± 0.5 | 0.97 |

Chamfer is 3.5× better, and every one of the six rooms improves.

**How the 2D metrics are computed (protocol 2):** a pixel counts if its Faro depth lies in 0.1–5 m and the
prediction is finite and > 0. Predictions are clipped to [0.1, 5] m before scoring. The prediction is used exactly as
the model outputs it, in metres, with **no** per-image scale or shift alignment.

## How to use

The model loads like any Depth Anything V2 checkpoint in 🤗 Transformers, and its output is **depth in metres**.

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

repo = "NackPanupong/roomscan-dav2-metric-large-ft-lidar-all"
processor = AutoImageProcessor.from_pretrained(repo)
model = AutoModelForDepthEstimation.from_pretrained(repo).eval()

image = Image.open("room.jpg").convert("RGB")      # upright (not rotated) indoor photo
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

depth = processor.post_process_depth_estimation(
    out, target_sizes=[(image.height, image.width)]
)[0]["predicted_depth"]                              # (H, W) tensor, metres
print(depth.min().item(), depth.max().item())
```

You can also use the pipeline:

```python
from transformers import pipeline

pipe = pipeline("depth-estimation", model="NackPanupong/roomscan-dav2-metric-large-ft-lidar-all")
result = pipe("room.jpg")
depth_m = result["predicted_depth"]   # metres
vis = result["depth"]                 # PIL image normalised for display only; do not use it for measurement
```

Tips:
- **The image must be upright.** ARKitScenes and many phone captures store frames in landscape even when the device
  was held in portrait. Rotate them upright before inference, using the `sky_direction` field in ARKitScenes.
- The model was trained on 640×480 iPad frames with the processor's default 518-px short side.
  Other resolutions work, but we have not measured them.
- To turn depth into a point cloud you need the camera intrinsics, as usual. The model itself does not use them.

## Intended use

- Getting metric depth indoors from RGB-only cameras: phones without LiDAR, webcams, archived video.
- Rough room-scale 3D reconstruction or measurement, where tens of centimetres of error are acceptable.
- Research on monocular metric depth, domain adaptation, and using sensor depth as a teacher.

**Not suitable for:**
- Accurate measurement. The error is still about 5× worse than real LiDAR (see the table above).
- Outdoor scenes, or depths beyond about 5 m. Training labels were capped at 10 m and all evaluation used 0.1–5 m.
- Safety-critical uses such as robot navigation or obstacle avoidance.

## Limitations

- **The scale still varies from view to view.** Fine-tuning moves the average scale close to correct, but individual
  frames of the same room can still disagree. This is the main reason 3D fusion reaches 15.7 cm rather than LiDAR's
  3 cm. For comparison, DA-v2 Large reaches 5.3 cm when it is given the true scale for every frame, which is not
  possible in practice.
- **One dataset, one device.** Every capture is an iPad Pro in homes (ARKitScenes, mostly in the US).
  Offices, empty rooms, low light and other phones or cameras have not been evaluated.
- **The evaluation set is small.** Results cover 6 rooms in 3D and 26 rooms in 2D. The checkpoint was chosen on
  only 3 validation rooms. Across our three fine-tunes, which used Faro, LiDAR on 10 rooms and LiDAR on 24 rooms as
  labels, the differences are within noise. What we can claim is that fine-tuned beats pretrained; we cannot rank the
  teachers.
- **LiDAR label quality.** Labels are ARKit `lowres_depth` (256×192) with confidence ≥ 1, resized with nearest
  neighbour. The model therefore inherits LiDAR's weaknesses, such as dark or shiny surfaces and thin objects.

## Training details

| | |
|---|---|
| Base model | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` (335 M params) |
| Trainable | DPT neck + depth head only (31 M params); DINOv2 backbone **frozen** |
| Labels | ARKit LiDAR `lowres_depth`, confidence ≥ 1, masked beyond 10 m |
| Training data | 24 ARKitScenes rooms (Training fold), 19,896 frames (`vga_wide` 640×480, about 10 fps) |
| Validation | 3 rooms from the ARKitScenes **Validation** fold, scored against Faro; used only for checkpoint choice |
| Loss | Scale-invariant log loss (SiLog, β = 0.15, ×10) |
| Optimiser | AdamW, lr 5e-5, weight decay 0.01, 200 warm-up steps, grad clip 1.0 |
| Batch | 2 × 4 grad accumulation = 8 effective, bf16 autocast |
| Augmentation | Horizontal flip (p = 0.5), colour jitter 0.2. No scale-changing augmentations, because the labels are in metres |
| Schedule | 12,000 steps; **checkpoint = step 2,000** (lowest validation AbsRel) |
| Validation AbsRel | 0.461 (pretrained, step 0) → 0.133 (this checkpoint), old protocol 1 |
| Hardware | 1× RTX 3070 Ti (8 GB), peak VRAM about 2.7 GB |
| Config sha1 | `9e81e677c4aa` |

**Data splits are separated by `visit_id`**, i.e. by home rather than by video, so no house appears in more than one
split. The 6 test rooms were fixed before any training experiment and were never used for training, validation or
checkpoint choice. The 20-room 2D test set shares no visit with any training or validation room.

### Sibling checkpoints

| Repo | Label | Training rooms |
|---|---|---|
| `NackPanupong/roomscan-dav2-metric-large-ft-faro` | Faro laser depth | 10 |
| `NackPanupong/roomscan-dav2-metric-large-ft-lidar` | iPad LiDAR | 10 (same rooms as above) |
| **`NackPanupong/roomscan-dav2-metric-large-ft-lidar-all`** (this one) | iPad LiDAR | 24 |

We recommend this one, because it is the model the paper's main result is reported on.

## License

These weights are derived from Depth Anything V2 **Large**, which is released under **CC-BY-NC-4.0**. They were
fine-tuned on **ARKitScenes**, which is licensed for non-commercial research only. The weights are therefore for
**non-commercial use only**.

## Citation

If you use this model, please cite Depth Anything V2 and ARKitScenes:

```bibtex
@inproceedings{yang2024depthanythingv2,
  title     = {Depth Anything V2},
  author    = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}

@inproceedings{baruch2021arkitscenes,
  title     = {{ARKitScenes}: A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB-D} Data},
  author    = {Baruch, Gilad and Chen, Zhuoyuan and Dehghan, Afshin and Dimry, Tal and Feigin, Yuri and Fu, Peter and Gebauer, Thomas and Joffe, Brandon and Kurz, Daniel and Schwartz, Arik and Shulman, Elad},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2021}
}
```

A citation for the roomscan paper will be added once it is published.
