"""N-gram hashing utilities for Engram memory bucket assignment."""
from __future__ import annotations

from typing import List

from coremake.models.engram_memory import EngramHasher


def build_ngram_bucket_ids(
    token_ids: List[int],
    num_buckets: int,
    max_ngram: int = 3,
    num_heads_per_ngram: int = 4,
) -> List[int]:
    """Build bucket IDs for all n-gram slots from a list of token IDs.
    
    Returns a flat list of bucket IDs, one per slot (total_slots = (max_ngram-1) * num_heads_per_ngram).
    """
    hasher = EngramHasher(num_buckets)
    bucket_ids: List[int] = []

    for n in range(2, max_ngram + 1):
        ngrams = [token_ids[i : i + n] for i in range(len(token_ids) - n + 1)]
        for head_idx in range(num_heads_per_ngram):
            if not ngrams:
                bucket_ids.append(0)
            else:
                # Use different ngrams for different heads; cycle if needed
                ng = ngrams[head_idx % len(ngrams)]
                bucket_ids.append(hasher.hash_ngram_ids(ng))

    return bucket_ids
