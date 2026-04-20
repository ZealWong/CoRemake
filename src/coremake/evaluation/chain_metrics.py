"""Logic chain evaluation metrics."""
from __future__ import annotations

from typing import Dict, List


def chain_precision(predicted: List[str], ground_truth: List[str]) -> float:
    if not predicted:
        return 0.0
    hits = len(set(predicted) & set(ground_truth))
    return hits / len(predicted)


def chain_recall(predicted: List[str], ground_truth: List[str]) -> float:
    if not ground_truth:
        return 0.0
    hits = len(set(predicted) & set(ground_truth))
    return hits / len(ground_truth)


def chain_f1(predicted: List[str], ground_truth: List[str]) -> float:
    p = chain_precision(predicted, ground_truth)
    r = chain_recall(predicted, ground_truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_chains(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
) -> Dict[str, float]:
    precisions, recalls, f1s = [], [], []
    for pred, gt in zip(predictions, ground_truths):
        precisions.append(chain_precision(pred, gt))
        recalls.append(chain_recall(pred, gt))
        f1s.append(chain_f1(pred, gt))

    return {
        "chain_precision": sum(precisions) / max(len(precisions), 1),
        "chain_recall": sum(recalls) / max(len(recalls), 1),
        "chain_f1": sum(f1s) / max(len(f1s), 1),
    }
