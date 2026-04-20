# Evaluation Guide

## Metrics

### Logic Chain
- **Chain Precision**: fraction of predicted chain steps in ground truth
- **Chain Recall**: fraction of ground truth steps found in prediction
- **Chain F1**: harmonic mean

### Anchor Selection
- **Precision@k**: fraction of top-k predictions that are correct
- **NDCG@k**: normalized discounted cumulative gain

### Relation Classification
- **Accuracy**: multi-class classification accuracy
- **Per-class F1**: extends / reframes / replaces / strengthens / none

### Composite Score
- Equal weighting of chain F1 + anchor precision@1 + relation accuracy
