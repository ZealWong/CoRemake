from __future__ import annotations

import torch
import torch.nn as nn


class AnchorRanker(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4 + 4, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )

    def forward(self, legacy: torch.Tensor, candidate: torch.Tensor,
                graph_feat: torch.Tensor, memory_feat: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([legacy, candidate, legacy - candidate, legacy * candidate,
                          graph_feat, memory_feat], dim=-1)
        return self.scorer(feat).squeeze(-1)
