"""Parse all PDFs in data/raw/pdfs/ and output papers.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from coremake.data.pdf_parser import parse_single_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, default="data/raw/pdfs")
    parser.add_argument("--output", type=str, default="data/processed/papers.jsonl")
    parser.add_argument("--domain", type=str, default=None)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {pdf_dir} (recursive)")

    records = []
    for pdf_path in tqdm(pdf_files, desc="Parsing"):
        try:
            rec = parse_single_pdf(pdf_path, domain=args.domain)
            records.append(rec)
        except Exception as e:
            print(f"  SKIP {pdf_path.name}: {e}")

    with open(output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")

    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
