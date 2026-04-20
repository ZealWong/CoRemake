"""Evaluate anchor paper selection quality."""
from __future__ import annotations

import argparse

from coremake.evaluation.anchor_metrics import evaluate_anchors
from coremake.utils.io import load_jsonl
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    preds = load_jsonl(args.predictions)
    gts = load_jsonl(args.ground_truth)

    pred_anchors = [p.get("anchors", []) for p in preds]
    gt_anchors = [g.get("anchors", []) for g in gts]

    metrics = evaluate_anchors(pred_anchors, gt_anchors, k_values=args.k_values)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
