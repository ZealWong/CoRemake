# CoRemake Architecture

## Overview

CoRemake is a training system for scientific literature update tracking and logic chain extraction, using Engram-style conditional memory augmentation.

## Core Components

### 1. Data Layer (Phase A)
- PDF parsing → structured `PaperRecord`
- Citation graph construction (NetworkX)
- Training pair mining (positive/negative/hard-negative)
- Weak-supervision relation labeling

### 2. Representation Layer (Phase B)
- `PaperEncoder`: SciBERT/SPECTER2 backbone + Engram memory fusion
- `EngramMemory`: N-gram hash lookup → gated memory retrieval → depthwise conv fusion
- Multi-task loss: contrastive (InfoNCE) + citation prediction (BCE) + year ordering

### 3. Relation Layer (Phase C)
- `RelationClassifier`: MLP on concatenated paper pair features
- Labels: extends / reframes / replaces / strengthens / none
- Weak supervision from language patterns in citation contexts

### 4. Retrieval & Graph Layer (Phase D)
- FAISS vector index for nearest-neighbor retrieval
- Memory table: {paper_id, year, embedding, role, relations, citations}
- Logic chain assembly via citation graph traversal

### 5. Decision Layer (Phase E)
- `AnchorRanker`: scores candidate anchor papers
- Two-stage: FAISS retrieve → ranker rerank
- Pairwise ranking loss

## Memory Flow

```
Paper text → Tokenizer → Token IDs → N-gram extraction → Hash bucket lookup
                                                              ↓
Paper text → Backbone → CLS hidden ←--- Gated fusion ←--- Memory values
                            ↓
                      Projection → L2-normalized embedding
```
