"""Guards the architecture rule: metric sources use identity, relative sources don't."""

import pytest

from roomscan.config import load_config
from roomscan.depth_sources.ground_truth import GroundTruthDepth
from roomscan.geometry.scale_align import IdentityAligner, OraclePerFrameAligner
from roomscan.pipeline import ReconstructionPipeline


class _EmptyDataset:
    scene_id = "fake"
    intrinsics = None

    def __len__(self):
        return 0

    def frame(self, idx):
        raise IndexError

    def frames(self, stride=1, max_frames=None):
        return iter(())

    def gt_mesh(self):
        return None


def test_gt_with_identity_is_accepted():
    cfg = load_config("configs/depth/gt.yaml")
    ReconstructionPipeline(cfg, dataset=_EmptyDataset(), depth_source=GroundTruthDepth(),
                           aligner=IdentityAligner())


def test_gt_with_oracle_is_rejected():
    cfg = load_config("configs/depth/gt.yaml")
    with pytest.raises(ValueError, match="already metric"):
        ReconstructionPipeline(cfg, dataset=_EmptyDataset(), depth_source=GroundTruthDepth(),
                               aligner=OraclePerFrameAligner())
