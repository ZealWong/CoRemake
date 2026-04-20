"""Evaluate logic chain quality."""
from __future__ import annotations

import argparse

from coremake.evaluation.chain_metrics import evaluate_chains
from coremake.utils.io import load_jsonl
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    gts = load_jsonl(args.ground_truth)

    pred_chains = [p.get("chain", []) for p in preds]
    gt_chains = [g.get("chain", []) for g in gts]

    metrics = evaluate_chains(pred_chains, gt_chains)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
