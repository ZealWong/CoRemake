"""Pair mining: build positive/negative pairs from citation graph for contrastive training."""
from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def build_positive_pairs(
    papers: List[Dict],
    citations: List[Dict],
) -> List[Dict]:
    """Citation-linked papers are positive pairs."""
    pairs = []
    for edge in citations:
        pairs.append({
            "left_paper_id": edge["src_paper_id"],
            "right_paper_id": edge["dst_paper_id"],
            "label": 1,
            "task": "citation",
        })
    logger.info(f"Built {len(pairs)} positive pairs from citations")
    return pairs


def build_negative_pairs(
    papers: List[Dict],
    positive_set: Set[Tuple[str, str]],
    num_negatives_per_paper: int = 7,
    seed: int = 42,
) -> List[Dict]:
    """Random non-citation papers as negative pairs."""
    rng = random.Random(seed)
    paper_ids = [p["paper_id"] for p in papers]
    pairs = []

    for pid in paper_ids:
        candidates = [c for c in paper_ids if c != pid and (pid, c) not in positive_set]
        if not candidates:
            continue
        negs = rng.sample(candidates, min(num_negatives_per_paper, len(candidates)))
        for neg in negs:
            pairs.append({
                "left_paper_id": pid,
                "right_paper_id": neg,
                "label": 0,
                "task": "contrastive",
            })

    logger.info(f"Built {len(pairs)} negative pairs")
    return pairs


def build_hard_negatives(
    papers: List[Dict],
    positive_set: Set[Tuple[str, str]],
    year_window: int = 5,
    num_hard: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """Same-year-window non-cited papers as hard negatives."""
    rng = random.Random(seed)
    year_map: Dict[str, int] = {p["paper_id"]: p.get("year", 0) or 0 for p in papers}
    paper_ids = [p["paper_id"] for p in papers]
    pairs = []

    for pid in paper_ids:
        y = year_map.get(pid, 0)
        candidates = [
            c for c in paper_ids
            if c != pid
            and (pid, c) not in positive_set
            and abs(year_map.get(c, 0) - y) <= year_window
        ]
        if not candidates:
            continue
        negs = rng.sample(candidates, min(num_hard, len(candidates)))
        for neg in negs:
            pairs.append({
                "left_paper_id": pid,
                "right_paper_id": neg,
                "label": 0,
                "task": "hard_negative",
            })

    logger.info(f"Built {len(pairs)} hard negative pairs")
    return pairs
