"""Test AnchorRanker."""
import torch
from coremake.models.anchor_ranker import AnchorRanker


def test_anchor_ranker_forward():
    model = AnchorRanker(dim=64)
    legacy = torch.randn(4, 64)
    candidate = torch.randn(4, 64)
    graph_feat = torch.randn(4, 2)
    memory_feat = torch.randn(4, 2)
    out = model(legacy, candidate, graph_feat, memory_feat)
    assert out.shape == (4,)
