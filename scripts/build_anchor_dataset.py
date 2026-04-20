"""Build anchor training dataset."""
from __future__ import annotations

import argparse

from coremake.data.anchor_builder import build_anchor_dataset
from coremake.utils.io import load_jsonl, load_yaml, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/build_anchor.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    papers = load_jsonl(cfg["data"]["papers_path"])
    citations = load_jsonl(cfg["data"]["citations_path"])

    records = build_anchor_dataset(
        papers, citations,
        legacy_year_threshold=cfg["anchor"]["legacy_year_threshold"],
        max_candidates=cfg["anchor"]["max_candidates"],
    )

    save_jsonl(records, cfg["data"]["output"])
    print(f"Built {len(records)} anchor examples → {cfg['data']['output']}")


if __name__ == "__main__":
    main()
