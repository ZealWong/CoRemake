"""Test PaperEncoder (requires transformers model download, skip if unavailable)."""
import pytest
import torch


def test_paper_encoder_output_structure():
    """Test encoder output shape without loading a real backbone."""
    from coremake.models.paper_encoder import PaperEncoderOutput

    out = PaperEncoderOutput(
        embedding=torch.randn(2, 256),
        cls_hidden=torch.randn(2, 768),
        memory_fused=torch.randn(2, 768),
    )
    assert out.embedding.shape == (2, 256)
    assert out.cls_hidden.shape == (2, 768)
