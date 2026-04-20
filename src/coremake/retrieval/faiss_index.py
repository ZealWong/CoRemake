"""FAISS index for paper embedding retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from coremake.utils.logging import get_logger

logger = get_logger(__name__)


class PaperFaissIndex:
    def __init__(self, dim: int = 256, use_gpu: bool = False) -> None:
        if faiss is None:
            raise ImportError("faiss is required. Install via: pip install faiss-cpu")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
        self.paper_ids: List[str] = []

    def add(self, paper_id: str, embedding: np.ndarray) -> None:
        embedding = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding)
        self.paper_ids.append(paper_id)

    def add_batch(self, paper_ids: List[str], embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.paper_ids.extend(paper_ids)

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        query = query.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.paper_ids) and idx >= 0:
                results.append((self.paper_ids[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "paper_ids.txt", "w") as f:
            for pid in self.paper_ids:
                f.write(pid + "\n")
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {path}")

    def load(self, path: str) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "paper_ids.txt") as f:
            self.paper_ids = [line.strip() for line in f]
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) from {path}")
