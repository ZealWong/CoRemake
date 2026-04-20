"""Citation parsing and matching utilities."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple


def extract_citation_markers(text: str) -> List[str]:
    """Extract numeric citation markers like [1], [2,3], [1-5]."""
    markers = re.findall(r"\[(\d+(?:[,\s\-\u2013]+\d+)*)\]", text)
    result = []
    for m in markers:
        for part in re.split(r"[,\s]+", m):
            if "-" in part or "\u2013" in part:
                bounds = re.split(r"[-\u2013]", part)
                if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                    for i in range(int(bounds[0]), int(bounds[1]) + 1):
                        result.append(str(i))
            elif part.strip().isdigit():
                result.append(part.strip())
    return result


def match_references_to_papers(
    ref_strings: List[str],
    paper_titles: Dict[str, str],
    threshold: float = 0.6,
) -> Dict[str, Optional[str]]:
    """Fuzzy-match reference strings to known paper_ids by title overlap."""
    matches: Dict[str, Optional[str]] = {}
    for ref in ref_strings:
        ref_lower = ref.lower()
        best_id, best_score = None, 0.0
        for pid, title in paper_titles.items():
            title_lower = title.lower()
            words = set(title_lower.split())
            ref_words = set(ref_lower.split())
            if not words:
                continue
            overlap = len(words & ref_words) / len(words)
            if overlap > best_score:
                best_score = overlap
                best_id = pid
        matches[ref] = best_id if best_score >= threshold else None
    return matches
