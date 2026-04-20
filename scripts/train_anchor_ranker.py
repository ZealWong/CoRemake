"""Train anchor ranker."""
from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from coremake.models.anchor_ranker import AnchorRanker
from coremake.models.losses import pairwise_ranking_loss
from coremake.utils.io import load_yaml
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/train_anchor.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model = AnchorRanker(dim=cfg["model"]["dim"])
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    logger.info(f"AnchorRanker initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info("Awaiting pre-computed embeddings from paper encoder.")
    # TODO: Load encoder checkpoint, compute embeddings, build AnchorDataset, train


if __name__ == "__main__":
    main()
