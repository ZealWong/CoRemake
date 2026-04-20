"""Metadata extraction and enrichment using DOI / filename patterns."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


def extract_doi_from_filename(filename: str) -> Optional[str]:
    """Try to recover DOI from a filename like 'doi.org_10.1021_nl3041417.pdf'."""
    name = Path(filename).stem
    name = name.replace("doi.org_", "")
    parts = name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("10."):
        return f"{parts[0]}/{parts[1]}"
    return None


def extract_year_from_doi(doi: str) -> Optional[int]:
    """Heuristic year extraction from DOI patterns."""
    # Some DOIs embed year info, but this is unreliable
    return None


def enrich_paper_record(record: Dict, metadata_db: Optional[Dict] = None) -> Dict:
    """Enrich a paper record with external metadata if available."""
    if metadata_db is None:
        return record

    doi = extract_doi_from_filename(record.get("pdf_path", ""))
    if doi and doi in metadata_db:
        meta = metadata_db[doi]
        if not record.get("title"):
            record["title"] = meta.get("title", "")
        if not record.get("abstract"):
            record["abstract"] = meta.get("abstract", "")
        if not record.get("year"):
            record["year"] = meta.get("year")
        if not record.get("authors"):
            record["authors"] = meta.get("authors", [])
        if not record.get("venue"):
            record["venue"] = meta.get("venue", "")
        if not record.get("keywords"):
            record["keywords"] = meta.get("keywords", [])

    return record
