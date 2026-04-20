from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .engram_memory import EngramMemory


@dataclass
class PaperEncoderOutput:
    embedding: torch.Tensor
    cls_hidden: torch.Tensor
    memory_fused: Optional[torch.Tensor]


class PaperEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int,
        proj_dim: int,
        use_engram: bool = True,
        memory_dim: int = 512,
        num_buckets: int = 500000,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.use_engram = use_engram
        self.engram = None
        if use_engram:
            self.engram = EngramMemory(
                hidden_dim=hidden_dim,
                memory_dim=memory_dim,
                num_buckets=num_buckets,
                max_ngram=3,
                num_heads_per_ngram=4,
            )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> PaperEncoderOutput:
        outputs = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls_hidden = outputs.last_hidden_state[:, 0]
        fused = None
        hidden = cls_hidden

        if self.use_engram and self.engram is not None and "ngram_bucket_ids" in batch:
            fused = self.engram(cls_hidden, batch["ngram_bucket_ids"])
            hidden = hidden + fused

        embedding = self.proj(self.out_norm(hidden))
        embedding = nn.functional.normalize(embedding, dim=-1)
        return PaperEncoderOutput(embedding=embedding, cls_hidden=cls_hidden, memory_fused=fused)
