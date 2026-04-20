"""Train paper encoder with contrastive learning.

Supports both single-GPU and multi-GPU (torchrun) training.
Usage:
  python scripts/train_paper_encoder.py --config configs/train/pretrain_encoder.yaml
  torchrun --nproc_per_node=6 scripts/train_paper_encoder.py --config configs/train/pretrain_encoder.yaml
"""
from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from coremake.models.paper_encoder import PaperEncoder
from coremake.models.losses import info_nce
from coremake.training.optim import build_optimizer
from coremake.training.scheduler import get_cosine_schedule_with_warmup
from coremake.utils.io import load_yaml
from coremake.utils.logging import get_logger
from coremake.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/pretrain_encoder.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(42)

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model = PaperEncoder(
        backbone_name=model_cfg["backbone_name"],
        hidden_dim=model_cfg["hidden_dim"],
        proj_dim=model_cfg["proj_dim"],
        use_engram=model_cfg.get("use_engram", True),
        memory_dim=model_cfg.get("memory_dim", 512),
        num_buckets=model_cfg.get("num_buckets", 300000),
    )

    optimizer = build_optimizer(model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PaperEncoder initialized")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Backbone: {model_cfg['backbone_name']}")
    logger.info(f"  Engram: {model_cfg.get('use_engram', True)}")
    logger.info("Awaiting training data pipeline (run Phase A first).")

    # TODO: Once Phase A data is ready:
    # 1. Build PaperPairDataset from positive_pairs.jsonl + negative_pairs.jsonl
    # 2. Create DataLoader
    # 3. Use Trainer class with multi-task loss
    # 4. Save best checkpoint to checkpoints/paper_encoder/best.pt


if __name__ == "__main__":
    main()
