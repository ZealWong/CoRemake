# Training Guide

## Prerequisites

```bash
conda create -n engram python=3.10 -y
conda activate engram
pip install -e ".[dev]"
```

## Full Pipeline (Recommended)

```bash
python scripts/run_pipeline.py --phases all
```

## Step-by-Step

### Phase A: Data Preparation

```bash
python scripts/parse_pdfs.py
python scripts/enrich_metadata.py
python scripts/build_citation_graph.py
python scripts/build_training_pairs.py --config configs/data/build_pairs.yaml
python scripts/build_relation_labels.py
python scripts/build_anchor_dataset.py --config configs/data/build_anchor.yaml
```

### Phase B: Encoder Pretraining

```bash
python scripts/train_paper_encoder.py --config configs/train/pretrain_encoder.yaml
```

### Phase C: Relation Classifier

```bash
python scripts/train_relation_classifier.py --config configs/train/finetune_relation.yaml
```

### Phase D: Memory Index

```bash
python scripts/build_memory_index.py --config configs/model/engram_memory.yaml
```

### Phase E: Anchor Ranker

```bash
python scripts/train_anchor_ranker.py --config configs/train/train_anchor.yaml
```

### Phase F: Evaluation

```bash
python scripts/evaluate_logic_chain.py --predictions ... --ground_truth ...
python scripts/evaluate_anchor.py --predictions ... --ground_truth ...
```

## Multi-GPU Training

For multi-GPU training on 6×RTX 5090:

```bash
torchrun --nproc_per_node=6 scripts/train_paper_encoder.py --config configs/train/pretrain_encoder.yaml
```
