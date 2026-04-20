"""Logic chain assembly: trace evolution from legacy paper to current frontier."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from coremake.utils.logging import get_logger

logger = get_logger(__name__)


def build_logic_chain(
    start_paper_id: str,
    citation_graph: nx.DiGraph,
    relation_map: Optional[Dict[Tuple[str, str], str]] = None,
    max_depth: int = 10,
    max_chain_length: int = 20,
) -> List[Dict]:
    """Trace citation chain forward from a legacy paper to the frontier.
    
    Returns a list of chain steps: [{paper_id, depth, relation_to_prev}]
    """
    if start_paper_id not in citation_graph:
        return []

    chain = []
    visited: Set[str] = set()
    queue = [(start_paper_id, 0, None)]

    while queue and len(chain) < max_chain_length:
        pid, depth, rel = queue.pop(0)
        if pid in visited or depth > max_depth:
            continue
        visited.add(pid)

        chain.append({
            "paper_id": pid,
            "depth": depth,
            "relation_to_prev": rel,
        })

        # Papers that cite this one (reverse edges = forward in time)
        successors = list(citation_graph.predecessors(pid))
        for succ in successors:
            rel_label = None
            if relation_map and (succ, pid) in relation_map:
                rel_label = relation_map[(succ, pid)]
            queue.append((succ, depth + 1, rel_label))

    logger.info(f"Built logic chain from {start_paper_id}: {len(chain)} steps")
    return chain


def find_frontier_papers(
    citation_graph: nx.DiGraph,
    min_year: Optional[int] = None,
) -> List[str]:
    """Find papers with no outgoing citations (frontier/newest)."""
    frontier = [
        n for n in citation_graph.nodes()
        if citation_graph.out_degree(n) == 0
    ]
    if min_year is not None:
        frontier = [
            n for n in frontier
            if citation_graph.nodes[n].get("year", 0) and citation_graph.nodes[n]["year"] >= min_year
        ]
    return frontier
