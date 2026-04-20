"""Optimizer factory."""
from __future__ import annotations

from typing import Iterator, Tuple

import torch.nn as nn
from torch.optim import AdamW, Optimizer


def build_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    embedding_lr_scale: float = 5.0,
) -> Optimizer:
    """Build AdamW optimizer with separate LR for embedding parameters."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    
    embedding_params = []
    other_params_decay = []
    other_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "engram" in name or "tables" in name:
            embedding_params.append(param)
        elif any(nd in name for nd in no_decay):
            other_params_no_decay.append(param)
        else:
            other_params_decay.append(param)

    param_groups = [
        {"params": other_params_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": other_params_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": embedding_params, "lr": lr * embedding_lr_scale, "weight_decay": 0.0},
    ]

    return AdamW(param_groups)
