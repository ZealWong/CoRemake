"""End-to-end remake evaluation metrics."""
from __future__ import annotations

from typing import Dict, List


def evaluate_remake_quality(
    chain_metrics: Dict[str, float],
    anchor_metrics: Dict[str, float],
    relation_accuracy: float,
) -> Dict[str, float]:
    """Aggregate all metrics into a unified evaluation report."""
    report = {}
    report.update({f"chain/{k}": v for k, v in chain_metrics.items()})
    report.update({f"anchor/{k}": v for k, v in anchor_metrics.items()})
    report["relation/accuracy"] = relation_accuracy

    # Composite score (equal weighting)
    chain_f1 = chain_metrics.get("chain_f1", 0.0)
    anchor_p1 = anchor_metrics.get("precision@1", 0.0)
    composite = (chain_f1 + anchor_p1 + relation_accuracy) / 3.0
    report["composite_score"] = composite

    return report
