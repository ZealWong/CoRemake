"""Test RelationClassifier."""
import torch
from coremake.models.relation_classifier import RelationClassifier


def test_relation_classifier_forward():
    model = RelationClassifier(input_dim=64, num_labels=5)
    left = torch.randn(4, 64)
    right = torch.randn(4, 64)
    year_diff = torch.randn(4, 1)
    citation_flag = torch.randn(4, 1)
    out = model(left, right, year_diff, citation_flag)
    assert out.shape == (4, 5)
