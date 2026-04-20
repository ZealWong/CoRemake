"""Weak-supervision relation labeler using language patterns."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from coremake.constants import RELATION_LABELS
from coremake.utils.logging import get_logger

logger = get_logger(__name__)

# Language patterns for weak relation labeling
_EXTEND_PATTERNS = [
    r"build(?:s|ing)?\s+(?:on|upon)",
    r"extend(?:s|ing)?",
    r"improv(?:e|es|ing)\s+(?:on|upon)",
    r"follow(?:s|ing)?\s+up",
]
_REFRAME_PATTERNS = [
    r"reinterpret",
    r"reframe",
    r"new\s+perspective",
    r"alternative\s+(?:view|approach|framework)",
]
_REPLACE_PATTERNS = [
    r"supersede",
    r"replace",
    r"obsolete",
    r"outperform(?:s|ing)?.*(?:completely|significantly)",
]
_STRENGTHEN_PATTERNS = [
    r"confirm(?:s|ing)?",
    r"corroborate",
    r"validate(?:s|ing)?",
    r"support(?:s|ing)?\s+(?:the\s+)?(?:finding|conclusion|result)",
    r"reproduce",
]


def _match_relation(text: str) -> Optional[str]:
    if not text:
        return None
    text_lower = text.lower()
    for pat in _EXTEND_PATTERNS:
        if re.search(pat, text_lower):
            return "extends"
    for pat in _REFRAME_PATTERNS:
        if re.search(pat, text_lower):
            return "reframes"
    for pat in _REPLACE_PATTERNS:
        if re.search(pat, text_lower):
            return "replaces"
    for pat in _STRENGTHEN_PATTERNS:
        if re.search(pat, text_lower):
            return "strengthens"
    return None


def label_relations(
    papers: List[Dict],
    citations: List[Dict],
    paper_map: Dict[str, Dict],
) -> List[Dict]:
    """Apply weak supervision patterns to citation contexts to label relations."""
    labeled = []

    for edge in citations:
        context = edge.get("context", "")
        if not context:
            # Try to find context from citing paper's sections
            src = paper_map.get(edge["src_paper_id"], {})
            sections = src.get("sections", {})
            context = sections.get("related_work", "") or sections.get("introduction", "")

        relation = _match_relation(context)
        if relation is not None:
            labeled.append({
                "left_paper_id": edge["src_paper_id"],
                "right_paper_id": edge["dst_paper_id"],
                "relation": relation,
                "confidence": 0.6,
                "source": "weak_supervision",
            })

    logger.info(f"Labeled {len(labeled)} relation pairs via weak supervision")
    return labeled
