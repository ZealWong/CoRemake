"""Text preprocessing utilities."""
from __future__ import annotations

import re
import unicodedata
from typing import List


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_ngrams(tokens: List[str], n: int) -> List[List[str]]:
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]


def clean_abstract(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\s*(abstract|summary)[:\s]*", "", text, flags=re.IGNORECASE)
    return text.strip()


def truncate_text(text: str, max_chars: int = 5000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
