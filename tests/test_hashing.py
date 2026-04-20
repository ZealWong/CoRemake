"""Test n-gram hashing utilities."""
from coremake.models.engram_memory import EngramHasher
from coremake.utils.hashing import build_ngram_bucket_ids


def test_hasher_deterministic():
    h = EngramHasher(1000)
    assert h.hash_ngram_ids([1, 2, 3]) == h.hash_ngram_ids([1, 2, 3])


def test_hasher_range():
    h = EngramHasher(1000)
    for ids in [[1, 2], [100, 200, 300], [0, 0, 0, 0]]:
        assert 0 <= h.hash_ngram_ids(ids) < 1000


def test_build_ngram_bucket_ids():
    ids = build_ngram_bucket_ids([10, 20, 30, 40], num_buckets=1000, max_ngram=3, num_heads_per_ngram=4)
    # (3-1) * 4 = 8 slots
    assert len(ids) == 8
    assert all(0 <= b < 1000 for b in ids)
