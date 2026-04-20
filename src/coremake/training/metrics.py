"""Training metrics computation."""
from __future__ import annotations

from typing import Dict, List

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def compute_recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Recall@k for ranking tasks."""
    topk_indices = scores.topk(k, dim=-1).indices
    hits = 0
    total = 0
    for i in range(scores.size(0)):
        positives = (labels[i] == 1).nonzero(as_tuple=True)[0]
        if len(positives) == 0:
            continue
        hits += len(set(topk_indices[i].tolist()) & set(positives.tolist()))
        total += len(positives)
    return hits / max(total, 1)


def compute_mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean Reciprocal Rank."""
    sorted_indices = scores.argsort(dim=-1, descending=True)
    mrr = 0.0
    count = 0
    for i in range(scores.size(0)):
        positives = set((labels[i] == 1).nonzero(as_tuple=True)[0].tolist())
        if not positives:
            continue
        for rank, idx in enumerate(sorted_indices[i].tolist(), 1):
            if idx in positives:
                mrr += 1.0 / rank
                break
        count += 1
    return mrr / max(count, 1)
