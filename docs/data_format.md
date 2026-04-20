# Data Format Specification

## papers.jsonl

Each line is a JSON object:

```json
{
  "paper_id": "sha1_hash_12chars",
  "title": "...",
  "abstract": "...",
  "keywords": [],
  "authors": [],
  "year": 2021,
  "venue": "Nature",
  "domain": "materials",
  "pdf_path": "data/raw/pdfs/xxx.pdf",
  "references": ["ref_id_1", "ref_id_2"],
  "sections": {
    "introduction": "...",
    "related_work": "...",
    "method": "...",
    "experiment": "...",
    "conclusion": "..."
  }
}
```

## citations.jsonl

```json
{"src_paper_id": "citing_paper", "dst_paper_id": "cited_paper", "context": "...", "section": "related_work"}
```

## positive_pairs.jsonl / negative_pairs.jsonl

```json
{"left_paper_id": "...", "right_paper_id": "...", "label": 1, "task": "citation"}
```

## relation_weak_labels.jsonl

```json
{"left_paper_id": "...", "right_paper_id": "...", "relation": "extends", "confidence": 0.6, "source": "weak_supervision", "relation_id": 0}
```

## anchor_train.jsonl

```json
{"legacy_paper_id": "...", "candidate_ids": [...], "positive_ids": [...], "negative_ids": [...]}
```
