"""Dataset builder: converts papers.jsonl + citations.jsonl into PyTorch datasets."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from coremake.utils.io import load_jsonl


class PaperPairDataset(Dataset):
    """Dataset for contrastive / citation / year-order training."""

    def __init__(
        self,
        pairs_path: str,
        papers_path: str,
        tokenizer,
        max_length: int = 512,
        num_negatives: int = 7,
    ) -> None:
        self.pairs = load_jsonl(pairs_path)
        papers = load_jsonl(papers_path)
        self.paper_map: Dict[str, Dict] = {p["paper_id"]: p for p in papers}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_paper(self, paper_id: str) -> Dict[str, torch.Tensor]:
        paper = self.paper_map.get(paper_id, {})
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        left = self._encode_paper(pair["left_paper_id"])
        right = self._encode_paper(pair["right_paper_id"])
        return {
            "left_input_ids": left["input_ids"],
            "left_attention_mask": left["attention_mask"],
            "right_input_ids": right["input_ids"],
            "right_attention_mask": right["attention_mask"],
            "label": torch.tensor(pair["label"], dtype=torch.long),
        }


class RelationDataset(Dataset):
    """Dataset for relation classification."""

    def __init__(self, labels_path: str, embeddings: Dict[str, torch.Tensor]) -> None:
        self.records = load_jsonl(labels_path)
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        left = self.embeddings.get(rec["left_paper_id"], torch.zeros(256))
        right = self.embeddings.get(rec["right_paper_id"], torch.zeros(256))
        return {
            "left": left,
            "right": right,
            "label": torch.tensor(rec.get("relation_id", 4), dtype=torch.long),
        }


class AnchorDataset(Dataset):
    """Dataset for anchor ranker training."""

    def __init__(self, anchor_path: str, embeddings: Dict[str, torch.Tensor]) -> None:
        self.records = load_jsonl(anchor_path)
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        legacy = self.embeddings.get(rec["legacy_paper_id"], torch.zeros(256))
        pos_ids = rec.get("positive_ids", [])
        neg_ids = rec.get("negative_ids", [])
        pos = self.embeddings.get(pos_ids[0], torch.zeros(256)) if pos_ids else torch.zeros(256)
        neg = self.embeddings.get(neg_ids[0], torch.zeros(256)) if neg_ids else torch.zeros(256)
        return {"legacy": legacy, "positive": pos, "negative": neg}
