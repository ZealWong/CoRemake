"""Build training pairs (positive + negative) from papers and citation graph."""
from __future__ import annotations

import argparse
from typing import Set, Tuple

from coremake.data.pair_mining import build_hard_negatives, build_negative_pairs, build_positive_pairs
from coremake.utils.io import load_jsonl, load_yaml, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/build_pairs.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    papers = load_jsonl(cfg["data"]["papers_path"])
    citations = load_jsonl(cfg["data"]["citations_path"])

    pos_pairs = build_positive_pairs(papers, citations)
    positive_set: Set[Tuple[str, str]] = {(p["left_paper_id"], p["right_paper_id"]) for p in pos_pairs}

    neg_pairs = build_negative_pairs(
        papers, positive_set,
        num_negatives_per_paper=cfg["pairs"]["num_negatives"],
        seed=cfg["splits"]["seed"],
    )
    hard_neg = build_hard_negatives(
        papers, positive_set,
        year_window=cfg["pairs"].get("max_year_diff", 5),
        seed=cfg["splits"]["seed"],
    )

    save_jsonl(pos_pairs, cfg["data"]["output_dir"] + "/positive_pairs.jsonl")
    save_jsonl(neg_pairs + hard_neg, cfg["data"]["output_dir"] + "/negative_pairs.jsonl")

    print(f"Positive pairs: {len(pos_pairs)}")
    print(f"Negative pairs: {len(neg_pairs)} + {len(hard_neg)} hard")


if __name__ == "__main__":
    main()
