"""Anchor paper search using ranker + index."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from coremake.retrieval.faiss_index import PaperFaissIndex
from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def search_anchor_papers(
    legacy_embedding: np.ndarray,
    index: PaperFaissIndex,
    ranker: nn.Module,
    top_k_retrieve: int = 50,
    top_k_final: int = 5,
    graph_feat_fn=None,
    memory_feat_fn=None,
    device: str = "cuda",
) -> List[Tuple[str, float]]:
    """Two-stage anchor search: FAISS retrieve → ranker rerank."""
    
    # Stage 1: FAISS retrieval
    candidates = index.search(legacy_embedding, top_k=top_k_retrieve)
    if not candidates:
        return []

    # Stage 2: Rerank with anchor ranker
    ranker.eval()
    legacy_t = torch.tensor(legacy_embedding, dtype=torch.float32, device=device).unsqueeze(0)

    scored = []
    for pid, faiss_score in candidates:
        cand_emb = index.index.reconstruct(index.paper_ids.index(pid))
        cand_t = torch.tensor(cand_emb, dtype=torch.float32, device=device).unsqueeze(0)
        
        graph_feat = graph_feat_fn(pid) if graph_feat_fn else torch.zeros(2, device=device).unsqueeze(0)
        memory_feat = memory_feat_fn(pid) if memory_feat_fn else torch.zeros(2, device=device).unsqueeze(0)

        with torch.no_grad():
            score = ranker(legacy_t, cand_t, graph_feat, memory_feat).item()
        scored.append((pid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k_final]
