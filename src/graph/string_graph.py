"""
graph/string_graph.py — single source of truth for STRING PPI parsing.

parse_string_edges is the one shared parser (combined_score threshold + alias
mapping to HGNC, restricted to the gene universe). StringGraphCache parses once
over the universe and serves per-fold induced subgraphs. 
"""
from __future__ import annotations

from typing import Sequence, Set, Tuple

import numpy as np
import pandas as pd


def parse_string_edges(ppi_path: str, alias_path: str,
                       universe_genes: Sequence[str],
                       confidence_threshold: int = 400) -> Set[Tuple[str, str]]:
    """Return deduped undirected HGNC symbol edges among the gene universe."""
    universe = set(universe_genes)

    aliases = pd.read_csv(alias_path, sep="\t")
    aliases.rename(columns={"#string_protein_id": "string_protein_id"}, inplace=True)
    aliases = aliases[aliases["alias"].isin(universe)]
    id_map = dict(zip(aliases["string_protein_id"], aliases["alias"]))

    ppi = pd.read_csv(ppi_path, sep=" ")
    ppi = ppi[ppi["combined_score"] >= confidence_threshold]
    ppi["g1"] = ppi["protein1"].map(id_map)
    ppi["g2"] = ppi["protein2"].map(id_map)
    ppi = ppi.dropna(subset=["g1", "g2"])
    ppi = ppi[ppi["g1"] != ppi["g2"]]

    edges: Set[Tuple[str, str]] = set()
    for a, b in zip(ppi["g1"].to_numpy(), ppi["g2"].to_numpy()):
        edges.add((a, b) if a <= b else (b, a))
    return edges


class StringGraphCache:
    """Parse STRING once over the universe; serve per-fold induced subgraphs."""

    def __init__(self, ppi_path: str, alias_path: str,
                 universe_genes: Sequence[str], confidence_threshold: int = 400):
        self.confidence_threshold = confidence_threshold
        self.edges = parse_string_edges(ppi_path, alias_path, universe_genes,
                                        confidence_threshold)
        self.n_universe_edges = len(self.edges)
        print(f"  [graph] {len(self.edges)} undirected edges among the "
              f"{len(set(universe_genes))}-gene universe "
              f"(combined_score >= {confidence_threshold}).")

    def induced_adjacency(self, selected_genes: Sequence[str]) -> np.ndarray:
        """Binary N x N adjacency on the induced subgraph, node order =
        `selected_genes` (isolated nodes kept)."""
        idx = {g: i for i, g in enumerate(selected_genes)}
        sel = set(selected_genes)
        n = len(selected_genes)
        A = np.zeros((n, n), dtype=np.float32)
        for a, b in self.edges:
            if a in sel and b in sel:
                i, j = idx[a], idx[b]
                A[i, j] = 1.0
                A[j, i] = 1.0
        return A
