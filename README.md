# CoRemake

**Scientific Literature Tracking and Logic Chain Extraction with Engram Memory Augmentation**

CoRemake is a training system for scientific literature update tracking and logic chain extraction. It leverages Engram's conditional memory, hashed n-gram lookup, and context-aware gating mechanisms for paper-level representation learning, relation extraction, logic chain construction, and anchor paper selection.

## Core Capabilities

1. **Paper Parsing** — PDF → structured paper records with DOI extraction
2. **Paper Encoder** — Paper-level encoder augmented with Engram memory (SciBERT backbone)
3. **Relation Classification** — Classify citation relationships (extends / reframes / replaces / strengthens)
4. **Logic Chain Mining** — Discover legacy → frontier evolution chains
5. **Anchor Paper Selection** — Rank and select key anchor papers in a research domain

## Quick Start

```bash
# Install
pip install -e .

# 1. Place PDFs in data/raw/pdfs/
# 2. Run the full pipeline
python scripts/run_pipeline.py --phases all

# Or run individual steps:
python scripts/parse_pdfs.py                # Parse PDFs
python scripts/build_citation_graph.py      # Build citation graph via DOI matching
python scripts/build_training_pairs.py      # Generate training pairs
python scripts/train_paper_encoder.py       # Train the paper encoder
```

## Project Structure

```
CoRemake/
 configs/                # YAML configuration files
   ├── base.yaml
   ├── data/               # Data pipeline configs
   ├── model/              # Model architecture configs
   └── train/              # Training configs
 scripts/                # Runnable pipeline scripts
   ├── run_pipeline.py     # Full pipeline orchestrator
   ├── parse_pdfs.py       # A1: PDF parsing
   ├── enrich_metadata.py  # A2: Metadata enrichment
   ├── build_citation_graph.py  # A3: Citation graph construction
   └── ...
 src/coremake/           # Core library
   ├── data/               # Data processing (PDF parser, pair mining, etc.)
   ├── models/             # Neural models (encoder, classifier, ranker)
   ├── training/           # Training loop, callbacks, optimizers
   ├── retrieval/          # FAISS index, memory table, logic chain
 evaluation/         # Metrics and evaluation   ├─
   └── utils/              # Utilities (IO, logging, hashing, etc.)
 tests/                  # Unit tests
 docs/                   # Documentation
 notebooks/              # Jupyter notebooks
```

## Pipeline Phases

| Phase | Description | Script |
|-------|-------------|--------|
| A1 | Parse PDFs → papers.jsonl | parse_pdfs.py |
| A2 | Enrich metadata | enrich_metadata.py |
| A3 | Build citation graph (DOI cross-matching) | build_citation_graph.py |
| A4 | Build training pairs (positive + negative) | build_training_pairs.py |
| A5 | Weak supervision relation labels | build_relation_labels.py |
| A6 | Build anchor training dataset | build_anchor_dataset.py |
| B | Train paper encoder (contrastive + citation) | train_paper_encoder.py |
| C | Train relation classifier | train_relation_classifier.py |
| D | Build memory index (FAISS + Engram) | build_memory_index.py |
| E | Train anchor ranker | train_anchor_ranker.py |
| F | End-to-end evaluation | evaluate_logic_chain.py |

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- Transformers >= 4.36
- See `requirements.txt` for full dependencies

## License

MIT
