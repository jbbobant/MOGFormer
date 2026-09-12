"""STRING protein–protein interaction edges as a gene-level graph.

STRING ships interactions between protein identifiers, so building a gene graph
means mapping those identifiers onto HGNC symbols through the alias table and
keeping only interactions above a confidence threshold.

Parsing the full human interactome is expensive and the gene universe is fixed
across folds, so :class:`StringGraphCache` parses once and serves the induced
subgraph for each fold's selected genes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: STRING combined-score floor. 400 is STRING's own "medium confidence" cutoff.
DEFAULT_CONFIDENCE_THRESHOLD = 400


def parse_string_edges(
    ppi_path: str,
    alias_path: str,
    universe_genes: Sequence[str],
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> set[tuple[str, str]]:
    """Read STRING interactions and return undirected gene-symbol edges.

    Protein identifiers are mapped to HGNC symbols via the alias table,
    restricted to ``universe_genes``. Self-loops are dropped and each edge is
    stored once with its endpoints in lexicographic order, so the returned set
    is genuinely undirected.

    Args:
        ppi_path: Path to STRING ``protein.links`` (space-separated, with
            ``protein1``, ``protein2`` and ``combined_score`` columns).
        alias_path: Path to STRING ``protein.aliases`` (tab-separated, with
            ``#string_protein_id`` and ``alias`` columns).
        universe_genes: Gene symbols the graph may contain.
        confidence_threshold: Minimum STRING combined score to keep an edge.

    Returns:
        Deduplicated ``(gene_a, gene_b)`` pairs with ``gene_a <= gene_b``, both
        drawn from ``universe_genes``.
    """
    universe = set(universe_genes)

    aliases = pd.read_csv(alias_path, sep="\t")
    aliases = aliases.rename(columns={"#string_protein_id": "string_protein_id"})
    aliases = aliases[aliases["alias"].isin(universe)]
    protein_to_symbol = dict(
        zip(aliases["string_protein_id"], aliases["alias"], strict=True)
    )

    interactions = pd.read_csv(ppi_path, sep=" ")
    interactions = interactions[interactions["combined_score"] >= confidence_threshold]
    interactions["gene_a"] = interactions["protein1"].map(protein_to_symbol)
    interactions["gene_b"] = interactions["protein2"].map(protein_to_symbol)
    interactions = interactions.dropna(subset=["gene_a", "gene_b"])
    interactions = interactions[interactions["gene_a"] != interactions["gene_b"]]

    edges: set[tuple[str, str]] = set()
    for gene_a, gene_b in zip(
        interactions["gene_a"].to_numpy(),
        interactions["gene_b"].to_numpy(),
        strict=True,
    ):
        edges.add((gene_a, gene_b) if gene_a <= gene_b else (gene_b, gene_a))
    return edges


class StringGraphCache:
    """Parse STRING once over a gene universe, then serve induced subgraphs.

    Gene selection is fold-local, so the graph must be rebuilt for every fold.
    Re-parsing the interactome each time would dominate runtime; this class
    pays that cost once and reduces per-fold work to a set intersection.

    Attributes:
        confidence_threshold: Score floor applied when the edges were parsed.
        edges: Undirected edges over the universe, as returned by
            :func:`parse_string_edges`.
        n_universe_edges: Number of cached edges.
    """

    def __init__(
        self,
        ppi_path: str,
        alias_path: str,
        universe_genes: Sequence[str],
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Parse the interactome over ``universe_genes``.

        Args:
            ppi_path: Path to STRING ``protein.links``.
            alias_path: Path to STRING ``protein.aliases``.
            universe_genes: Gene symbols the graph may contain.
            confidence_threshold: Minimum STRING combined score to keep an edge.
        """
        self.confidence_threshold = confidence_threshold
        self.edges = parse_string_edges(
            ppi_path, alias_path, universe_genes, confidence_threshold
        )
        self.n_universe_edges = len(self.edges)
        logger.info(
            "STRING: %d undirected edges among the %d-gene universe "
            "(combined_score >= %d)",
            self.n_universe_edges,
            len(set(universe_genes)),
            confidence_threshold,
        )

    def induced_adjacency(self, selected_genes: Sequence[str]) -> np.ndarray:
        """Build the binary adjacency of the subgraph induced on selected genes.

        Isolated nodes are retained so that row ``i`` always corresponds to
        ``selected_genes[i]`` — downstream shape contracts depend on it.

        Args:
            selected_genes: Fold-local gene symbols, defining the node order.

        Returns:
            Symmetric binary adjacency of shape ``(n_selected, n_selected)``
            with a zero diagonal.
        """
        position = {gene: i for i, gene in enumerate(selected_genes)}
        selected = set(selected_genes)
        n_selected = len(selected_genes)

        adjacency = np.zeros((n_selected, n_selected), dtype=np.float32)
        for gene_a, gene_b in self.edges:
            if gene_a in selected and gene_b in selected:
                i, j = position[gene_a], position[gene_b]
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
        return adjacency
