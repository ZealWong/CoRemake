"""Train relation classifier."""
from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from coremake.models.relation_classifier import RelationClassifier
from coremake.utils.io import load_yaml
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/finetune_relation.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model = RelationClassifier(
        input_dim=cfg["model"]["input_dim"],
        num_labels=cfg["model"]["num_labels"],
    )
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    logger.info(f"RelationClassifier initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info("Awaiting pre-computed embeddings from paper encoder.")
    # TODO: Load encoder checkpoint, compute embeddings, build RelationDataset, train


if __name__ == "__main__":
    main()
