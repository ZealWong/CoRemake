from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(query: torch.Tensor, pos: torch.Tensor, negatives: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    pos_sim = F.cosine_similarity(query, pos, dim=-1).unsqueeze(1)
    neg_sim = F.cosine_similarity(query.unsqueeze(1), negatives, dim=-1)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / tau
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def citation_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def year_order_loss(left_year_score: torch.Tensor, right_year_score: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    return torch.relu(margin - (right_year_score - left_year_score)).mean()


def pairwise_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    return torch.relu(margin - pos_scores + neg_scores).mean()
