"""Memory table: stores paper metadata + embeddings for retrieval and graph queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MemoryEntry:
    paper_id: str
    year: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    role: str = "unknown"
    relations: Dict[str, List[str]] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)


class MemoryTable:
    """In-memory table: M = {(paper_id, year, embedding, role, relations, citations)}."""

    def __init__(self) -> None:
        self.entries: Dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        self.entries[entry.paper_id] = entry

    def get(self, paper_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(paper_id)

    def get_by_year_range(self, start: int, end: int) -> List[MemoryEntry]:
        return [e for e in self.entries.values() if e.year and start <= e.year <= end]

    def get_cited_by(self, paper_id: str) -> List[str]:
        return [
            pid for pid, entry in self.entries.items()
            if paper_id in entry.citations
        ]

    def size(self) -> int:
        return len(self.entries)
