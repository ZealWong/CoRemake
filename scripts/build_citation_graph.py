"""Build citation graph from papers.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=str, default="data/processed/papers.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/citations.jsonl")
    args = parser.parse_args()

    papers = []
    with open(args.papers, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    G = nx.DiGraph()
    for p in papers:
        G.add_node(p["paper_id"], title=p["title"], year=p.get("year"))

    # 建 DOI → paper_id 映射
    doi_to_id: dict[str, str] = {}
    for p in papers:
        doi = p.get("extra", {}).get("doi", "")
        if doi:
            doi_to_id[doi.lower()] = p["paper_id"]

    print(f"Papers with DOI: {len(doi_to_id)}/{len(papers)}")

    # references 存的是 DOI 字符串，通过映射表转为 paper_id
    edges = []
    for p in papers:
        for ref_doi in p.get("references", []):
            target_id = doi_to_id.get(ref_doi.lower())
            if target_id and target_id != p["paper_id"]:
                edges.append({"src_paper_id": p["paper_id"], "dst_paper_id": target_id})
                G.add_edge(p["paper_id"], target_id)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Wrote {len(edges)} citation edges to {args.output}")


if __name__ == "__main__":
    main()
