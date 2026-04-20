"""Enrich paper metadata using DOI extraction from filenames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coremake.data.metadata_extractor import enrich_paper_record, extract_doi_from_filename
from coremake.utils.io import load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=str, default="data/processed/papers.jsonl")
    parser.add_argument("--metadata_dir", type=str, default="data/raw/metadata")
    parser.add_argument("--output", type=str, default="data/processed/papers.jsonl")
    args = parser.parse_args()

    papers = load_jsonl(args.papers)

    # Try to load external metadata DB if present
    metadata_db = {}
    meta_path = Path(args.metadata_dir) / "doi_metadata.jsonl"
    if meta_path.exists():
        for rec in load_jsonl(str(meta_path)):
            doi = rec.get("doi", "")
            if doi:
                metadata_db[doi] = rec

    enriched = []
    for paper in papers:
        paper = enrich_paper_record(paper, metadata_db if metadata_db else None)
        enriched.append(paper)

    save_jsonl(enriched, args.output)
    print(f"Enriched {len(enriched)} papers → {args.output}")


if __name__ == "__main__":
    main()
