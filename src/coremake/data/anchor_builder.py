"""Anchor dataset builder: creates training data for anchor ranker."""
from __future__ import annotations

import random
from typing import Dict, List

from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def build_anchor_dataset(
    papers: List[Dict],
    citations: List[Dict],
    legacy_year_threshold: int = 2010,
    max_candidates: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Build anchor training examples.
    
    For each legacy paper (year <= threshold), find papers that cite it
    (positives) and random non-citing papers (negatives) as anchor candidates.
    """
    rng = random.Random(seed)

    # Build reverse citation map: who cites this paper?
    cited_by: Dict[str, List[str]] = {}
    for edge in citations:
        dst = edge["dst_paper_id"]
        cited_by.setdefault(dst, []).append(edge["src_paper_id"])

    paper_ids = [p["paper_id"] for p in papers]
    all_ids_set = set(paper_ids)

    records = []
    for paper in papers:
        year = paper.get("year") or 9999
        if year > legacy_year_threshold:
            continue

        pid = paper["paper_id"]
        pos_ids = cited_by.get(pid, [])
        if not pos_ids:
            continue

        neg_pool = list(all_ids_set - set(pos_ids) - {pid})
        num_neg = min(max_candidates - len(pos_ids), len(neg_pool))
        neg_ids = rng.sample(neg_pool, max(0, num_neg)) if neg_pool else []

        records.append({
            "legacy_paper_id": pid,
            "candidate_ids": pos_ids + neg_ids,
            "positive_ids": pos_ids,
            "negative_ids": neg_ids,
        })

    logger.info(f"Built {len(records)} anchor training examples")
    return records
