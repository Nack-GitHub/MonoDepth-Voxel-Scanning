"""Write a synthetic ARKitScenes-format scene for end-to-end testing without real data.

    python scripts/make_synthetic_scene.py --root data/synthetic --frames 60

Then point any config at it:
    roomscan run --config configs/depth/gt.yaml --set dataset.root=data/synthetic dataset.scene=90000001
"""

from __future__ import annotations

import argparse

from roomscan.dataio.synthetic import write_synthetic_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--video-id", default="90000001")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--hires", type=int, nargs=2, default=(1920, 1440), metavar=("W", "H"))
    ap.add_argument("--no-color", action="store_true", help="skip 1920x1440 RGB (saves disk)")
    args = ap.parse_args()
    out = write_synthetic_scene(args.root, args.video_id, n_frames=args.frames,
                                hires=tuple(args.hires), write_color=not args.no_color)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
