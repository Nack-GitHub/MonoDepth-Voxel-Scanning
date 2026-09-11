import numpy as np
import pytest

from roomscan.types import Frame, Intrinsics


@pytest.fixture
def intr() -> Intrinsics:
    return Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


@pytest.fixture
def flat_wall_frame(intr) -> Frame:
    """A fronto-parallel wall at z=2m. Back-projection must give z==2 everywhere."""
    depth = np.full((intr.height, intr.width), 2.0, dtype=np.float32)
    rgb = np.zeros((intr.height, intr.width, 3), dtype=np.uint8)
    return Frame(idx=0, rgb=rgb, pose_c2w=np.eye(4), gt_depth=depth)
