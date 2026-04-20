"""Prediction heads for multi-task training."""
from __future__ import annotations

import torch
import torch.nn as nn


class CitationHead(nn.Module):
    """Binary head: predict whether paper_i cites paper_j."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([left, right], dim=-1)
        return self.linear(feat).squeeze(-1)


class YearOrderHead(nn.Module):
    """Scalar head: predict relative year score for ordering."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(embedding).squeeze(-1)
