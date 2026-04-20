from __future__ import annotations

import torch
import torch.nn as nn


class RelationClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int = 5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4 + 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                year_diff: torch.Tensor, citation_flag: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([left, right, left - right, left * right, year_diff, citation_flag], dim=-1)
        return self.mlp(feat)
