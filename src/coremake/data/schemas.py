from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SectionMap(BaseModel):
    introduction: Optional[str] = None
    related_work: Optional[str] = None
    method: Optional[str] = None
    experiment: Optional[str] = None
    conclusion: Optional[str] = None


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    keywords: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    domain: Optional[str] = None
    pdf_path: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    sections: SectionMap = Field(default_factory=SectionMap)
    extra: Dict[str, str] = Field(default_factory=dict)


class CitationEdge(BaseModel):
    src_paper_id: str
    dst_paper_id: str
    context: Optional[str] = None
    section: Optional[str] = None


class PairRecord(BaseModel):
    left_paper_id: str
    right_paper_id: str
    label: int
    task: str


class RelationRecord(BaseModel):
    left_paper_id: str
    right_paper_id: str
    relation: str
    confidence: float = 1.0
    source: str = "weak_supervision"


class AnchorRecord(BaseModel):
    legacy_paper_id: str
    candidate_ids: List[str]
    positive_ids: List[str]
    negative_ids: List[str]
