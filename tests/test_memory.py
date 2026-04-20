"""Test EngramMemory module."""
import torch
from coremake.models.engram_memory import EngramMemory


def test_engram_memory_forward():
    mem = EngramMemory(hidden_dim=64, memory_dim=64, num_buckets=100, max_ngram=3, num_heads_per_ngram=4)
    hidden = torch.randn(2, 64)
    bucket_ids = torch.randint(0, 100, (2, 8))  # (max_ngram-1)*num_heads = 8
    out = mem(hidden, bucket_ids)
    assert out.shape == (2, 64)


def test_engram_memory_lookup():
    mem = EngramMemory(hidden_dim=64, memory_dim=64, num_buckets=100, max_ngram=3, num_heads_per_ngram=4)
    bucket_ids = torch.randint(0, 100, (4, 8))
    looked_up = mem.lookup(bucket_ids)
    assert looked_up.shape == (4, 64)
