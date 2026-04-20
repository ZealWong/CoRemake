"""Build memory index: run encoder on all papers, save FAISS index + memory table."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from coremake.retrieval.faiss_index import PaperFaissIndex
from coremake.utils.io import load_jsonl, load_yaml
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model/engram_memory.yaml")
    parser.add_argument("--papers", type=str, default="data/processed/papers.jsonl")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/paper_encoder/best.pt")
    parser.add_argument("--output", type=str, default="data/processed/faiss_index")
    args = parser.parse_args()

    papers = load_jsonl(args.papers)
    logger.info(f"Loaded {len(papers)} papers")

    # TODO: Load trained encoder, compute embeddings for all papers
    # For now, create placeholder index
    dim = 256
    index = PaperFaissIndex(dim=dim)

    logger.info("Memory index builder ready. Awaiting trained encoder checkpoint.")
    logger.info(f"Will save index to: {args.output}")


if __name__ == "__main__":
    main()
