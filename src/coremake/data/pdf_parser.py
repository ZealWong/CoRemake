"""PDF parsing utilities using PyMuPDF."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import fitz  # pymupdf

from coremake.data.schemas import PaperRecord, SectionMap


_SECTION_PATTERNS = [
    (r"(?i)\b(introduction)\b", "introduction"),
    (r"(?i)\b(related\s+work|background|literature)\b", "related_work"),
    (r"(?i)\b(method|approach|model|framework)\b", "method"),
    (r"(?i)\b(experiment|result|evaluation)\b", "experiment"),
    (r"(?i)\b(conclusion|summary|future)\b", "conclusion"),
]


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"')\]}>]+")


def _doi_from_filename(path: Path) -> str | None:
    """从文件名中提取 DOI，如 'doi.org_10.1021_nl3041417.pdf' → '10.1021/nl3041417'."""
    stem = path.stem
    # 格式: doi.org_10.XXXX_YYYY  或  10.XXXX_YYYY
    m = re.search(r"(10\.\d{4,9})[_/](.+)", stem.replace("doi.org_", ""))
    if m:
        suffix = m.group(2).rstrip("_").replace("_", "/", 1)  # first _ → /
        return f"{m.group(1)}/{suffix}"
    return None


def _extract_dois_from_text(text: str) -> list[str]:
    """从全文中提取所有 DOI。"""
    dois = set()
    for m in _DOI_RE.finditer(text):
        doi = m.group(0).rstrip(".")
        dois.add(doi.lower())
    return list(dois)


def _make_paper_id(path: Path) -> str:
    return hashlib.sha1(path.stem.encode()).hexdigest()[:12]


def _extract_title(doc: fitz.Document) -> str:
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title and len(meta_title) > 5:
        return meta_title
    first_page = doc[0]
    blocks = first_page.get_text("blocks")
    if blocks:
        # 取第一个文本块的最大字号行作为标题
        return blocks[0][4].strip().split("\n")[0]
    return ""


def _split_sections(full_text: str) -> SectionMap:
    mapping: dict[str, str] = {}
    positions = []
    for pat, name in _SECTION_PATTERNS:
        m = re.search(pat, full_text)
        if m:
            positions.append((m.start(), name))
    positions.sort()

    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        mapping[name] = full_text[start:end].strip()[:5000]

    return SectionMap(**mapping)


def _extract_references(full_text: str) -> list[str]:
    """从全文提取引用的 DOI 列表。"""
    return _extract_dois_from_text(full_text)


def parse_single_pdf(pdf_path: Path, domain: Optional[str] = None) -> PaperRecord:
    doc = fitz.open(str(pdf_path))
    full_text = "\n".join(page.get_text() for page in doc)
    title = _extract_title(doc)

    # 简单启发式抽取 abstract
    abstract = ""
    abs_match = re.search(r"(?i)abstract[:\s]*(.+?)(?:\n\n|introduction)", full_text, re.DOTALL)
    if abs_match:
        abstract = abs_match.group(1).strip()[:2000]

    sections = _split_sections(full_text)
    year_match = re.search(r"((?:19|20)\d{2})", full_text[:3000])
    year = int(year_match.group(1)) if year_match else None

    doc.close()

    own_doi = _doi_from_filename(pdf_path)
    ref_dois = _extract_references(full_text)
    # 排除自身 DOI
    if own_doi:
        ref_dois = [d for d in ref_dois if d != own_doi.lower()]

    extra: dict[str, str] = {}
    if own_doi:
        extra["doi"] = own_doi

    return PaperRecord(
        paper_id=_make_paper_id(pdf_path),
        title=title,
        abstract=abstract,
        year=year,
        domain=domain,
        pdf_path=str(pdf_path),
        sections=sections,
        references=ref_dois,
        extra=extra,
    )
