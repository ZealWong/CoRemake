from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class EngramHasher:
    def __init__(self, num_buckets: int) -> None:
        self.num_buckets = num_buckets

    def hash_ngram_ids(self, token_ids: List[int]) -> int:
        h = 1469598103934665603
        for tid in token_ids:
            h ^= int(tid) + 0x9E3779B97F4A7C15
            h *= 1099511628211
            h &= 0xFFFFFFFFFFFFFFFF
        return h % self.num_buckets


class EngramMemory(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_buckets: int,
        max_ngram: int = 3,
        num_heads_per_ngram: int = 4,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_buckets = num_buckets
        self.max_ngram = max_ngram
        self.num_heads_per_ngram = num_heads_per_ngram

        total_slots = (max_ngram - 1) * num_heads_per_ngram
        slot_dim = memory_dim // total_slots
        if slot_dim * total_slots != memory_dim:
            raise ValueError("memory_dim must be divisible by total_slots")

        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(num_buckets, slot_dim).normal_(0, 0.02))
            for _ in range(total_slots)
        ])

        self.key_proj = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.q_norm = RMSNorm(hidden_dim)
        self.k_norm = RMSNorm(hidden_dim)
        self.v_norm = RMSNorm(hidden_dim)

        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=hidden_dim,
        )

    def lookup(self, batch_ngram_bucket_ids: torch.Tensor) -> torch.Tensor:
        pieces = []
        for i, table in enumerate(self.tables):
            bucket_ids = batch_ngram_bucket_ids[:, i]
            pieces.append(table[bucket_ids])
        return torch.cat(pieces, dim=-1)

    def forward(self, hidden: torch.Tensor, batch_ngram_bucket_ids: torch.Tensor) -> torch.Tensor:
        memory = self.lookup(batch_ngram_bucket_ids)
        key = self.k_norm(self.key_proj(memory))
        value = self.v_norm(self.value_proj(memory))
        query = self.q_norm(hidden)

        gate_logits = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        alpha = torch.sigmoid(gate_logits)
        gated = alpha * value

        conv_in = gated.unsqueeze(-1)
        conv_out = self.depthwise_conv(conv_in)
        conv_out = conv_out[..., :1]  # trim padding to match input length
        conv_out = conv_out.squeeze(-1)
        fused = F.silu(conv_out) + gated
        return fused
