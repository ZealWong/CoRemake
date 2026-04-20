"""Build weak-supervision relation labels."""
from __future__ import annotations

import argparse

from coremake.constants import RELATION2ID
from coremake.data.relation_labeler import label_relations
from coremake.utils.io import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=str, default="data/processed/papers.jsonl")
    parser.add_argument("--citations", type=str, default="data/processed/citations.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/relation_weak_labels.jsonl")
    args = parser.parse_args()

    papers = load_jsonl(args.papers)
    citations = load_jsonl(args.citations)
    paper_map = {p["paper_id"]: p for p in papers}

    labeled = label_relations(papers, citations, paper_map)

    # Add relation_id for training
    for rec in labeled:
        rec["relation_id"] = RELATION2ID.get(rec["relation"], 4)

    save_jsonl(labeled, args.output)
    print(f"Labeled {len(labeled)} relation pairs → {args.output}")


if __name__ == "__main__":
    main()
