"""Anchor selection evaluation metrics."""
from __future__ import annotations

import math
from typing import Dict, List


def precision_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = predicted[:k]
    return len(set(top_k) & set(ground_truth)) / k


def ndcg_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, pid in enumerate(predicted[:k]):
        if pid in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal = sorted([1 if pid in gt_set else 0 for pid in predicted[:k]], reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_anchors(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    results = {}
    for k in k_values:
        p_at_k = [precision_at_k(pred, gt, k) for pred, gt in zip(predictions, ground_truths)]
        n_at_k = [ndcg_at_k(pred, gt, k) for pred, gt in zip(predictions, ground_truths)]
        results[f"precision@{k}"] = sum(p_at_k) / max(len(p_at_k), 1)
        results[f"ndcg@{k}"] = sum(n_at_k) / max(len(n_at_k), 1)
    return results
