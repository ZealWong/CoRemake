"""Full pipeline: parse → enrich → pairs → train → index → eval.

Matches the blueprint's recommended training commands:
  Phase A: PDF → papers.jsonl + citations.jsonl
  Phase B: Encoder pretraining (contrastive + citation + year)
  Phase C: Relation classifier
  Phase D: Memory table / index
  Phase E: Anchor ranker
  Phase F: End-to-end evaluation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Tuple


def _steps(phases: str) -> List[Tuple[str, List[str]]]:
    """Return pipeline steps filtered by phase flags."""
    all_steps = [
        # Phase A: Data preparation
        ("A1", "Phase A: Parse PDFs",
         [sys.executable, "scripts/parse_pdfs.py"]),
        ("A2", "Phase A: Enrich metadata",
         [sys.executable, "scripts/enrich_metadata.py"]),
        ("A3", "Phase A: Build citation graph",
         [sys.executable, "scripts/build_citation_graph.py"]),
        ("A4", "Phase A: Build training pairs",
         [sys.executable, "scripts/build_training_pairs.py", "--config", "configs/data/build_pairs.yaml"]),
        ("A5", "Phase A: Build relation labels",
         [sys.executable, "scripts/build_relation_labels.py"]),
        ("A6", "Phase A: Build anchor dataset",
         [sys.executable, "scripts/build_anchor_dataset.py", "--config", "configs/data/build_anchor.yaml"]),

        # Phase B: Encoder pretraining
        ("B1", "Phase B: Train paper encoder",
         [sys.executable, "scripts/train_paper_encoder.py", "--config", "configs/train/pretrain_encoder.yaml"]),

        # Phase C: Relation classifier
        ("C1", "Phase C: Train relation classifier",
         [sys.executable, "scripts/train_relation_classifier.py", "--config", "configs/train/finetune_relation.yaml"]),

        # Phase D: Memory + index
        ("D1", "Phase D: Build memory index",
         [sys.executable, "scripts/build_memory_index.py", "--config", "configs/model/engram_memory.yaml"]),

        # Phase E: Anchor ranker
        ("E1", "Phase E: Train anchor ranker",
         [sys.executable, "scripts/train_anchor_ranker.py", "--config", "configs/train/train_anchor.yaml"]),

        # Phase F: Evaluation
        ("F1", "Phase F: End-to-end evaluation (placeholder)",
         [sys.executable, "-c", "print('Evaluation requires ground-truth data. Skipping.')"]),
    ]

    if phases == "all":
        return [(name, cmd) for _, name, cmd in all_steps]

    selected_phases = set(phases.upper())
    return [(name, cmd) for tag, name, cmd in all_steps if tag[0] in selected_phases]


def main() -> None:
    parser = argparse.ArgumentParser(description="CoRemake full training pipeline")
    parser.add_argument(
        "--phases", type=str, default="all",
        help="Which phases to run: 'all' or combination like 'AB' for Phase A+B only",
    )
    args = parser.parse_args()

    steps = _steps(args.phases)
    total = len(steps)
    print(f"\n🚀 CoRemake Pipeline: {total} steps")

    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] {name}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  ❌ FAILED at: {name}")
            sys.exit(1)
        print(f"  ✅ Done")

    print(f"\n{'='*60}")
    print("✅ Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
