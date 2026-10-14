"""STRING protein–protein interaction edges as a gene-level graph.

STRING ships interactions between protein identifiers, so building a gene graph
means mapping those identifiers onto HGNC symbols through the alias table and
keeping only interactions above a confidence threshold.

Parsing the full human interactome is expensive and the gene universe is fixed
across folds, so :class:`StringGraphCache` parses once and serves the induced
subgraph for each fold's selected genes.

STRING's combined score pools several evidence channels, one of which is
co-expression. A prior used by an encoder that reconstructs expression must not
be built from expression covariation, or the prior restates the data it is
meant to constrain; :func:`parse_string_channel_edges` therefore rebuilds the
score from chosen channels only (experimental and curated-database evidence by
default), with STRING's own combination rule.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: STRING combined-score floor. 400 is STRING's own "medium confidence" cutoff.
DEFAULT_CONFIDENCE_THRESHOLD = 400

#: Field separator of the portable edge list.
TSV_SEPARATOR = "	"


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
    protein_to_symbol = _protein_symbols(alias_path, set(universe_genes))

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


#: Evidence channels of STRING's detailed links file.
STRING_CHANNELS: tuple[str, ...] = (
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
    "textmining",
)

#: Channels independent of the expression data an encoder reconstructs:
#: physical-interaction experiments and curated pathway databases.
EVIDENCE_CHANNELS: tuple[str, ...] = ("experimental", "database")

#: STRING's prior probability of a random protein pair interacting, removed
#: from each channel before combining and added back after (von Mering 2005).
STRING_PRIOR = 0.041


def _protein_symbols(alias_path: str, universe: set[str]) -> dict[str, str]:
    """Map STRING protein identifiers to the universe's gene symbols."""
    aliases = pd.read_csv(alias_path, sep="\t")
    aliases = aliases.rename(columns={"#string_protein_id": "string_protein_id"})
    aliases = aliases[aliases["alias"].isin(universe)]
    return dict(zip(aliases["string_protein_id"], aliases["alias"], strict=True))


def combine_channel_scores(scores: np.ndarray) -> np.ndarray:
    """Combine per-channel STRING scores the way STRING combines them.

    Each channel score (STRING's 0–1000 scale) becomes a probability ``s``
    stripped of the prior ``p``, ``s' = max(s − p, 0) / (1 − p)``; channels
    are combined as independent evidence, ``c = 1 − Π(1 − s')``; and the
    prior is added back, ``c + p (1 − c)``.

    Args:
        scores: Channel scores, shape ``(n_pairs, n_channels)``, 0–1000.

    Returns:
        Combined scores on the 0–1000 scale, shape ``(n_pairs,)``.
    """
    probability = np.asarray(scores, dtype=np.float64) / 1000.0
    without_prior = np.clip(
        (probability - STRING_PRIOR) / (1.0 - STRING_PRIOR), 0.0, 1.0
    )
    combined = 1.0 - np.prod(1.0 - without_prior, axis=1)
    return 1000.0 * (combined + STRING_PRIOR * (1.0 - combined))


def parse_string_channel_edges(
    detailed_path: str,
    alias_path: str,
    universe_genes: Sequence[str],
    channels: Sequence[str] = EVIDENCE_CHANNELS,
    confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[tuple[str, str], float]:
    """Read STRING's detailed links and score edges from chosen channels only.

    Args:
        detailed_path: STRING ``protein.links.detailed`` file (space-separated,
            one column per channel of :data:`STRING_CHANNELS`).
        alias_path: STRING ``protein.aliases`` file.
        universe_genes: Gene symbols the graph may contain.
        channels: Channels the score is rebuilt from.
        confidence_threshold: Minimum rebuilt score (0–1000) to keep an edge.

    Returns:
        Undirected edges ``(gene_a, gene_b)`` with ``gene_a < gene_b``, mapped
        to their rebuilt score scaled to ``(0, 1]``. Where several protein
        pairs map to one gene pair, the highest score is kept.

    Raises:
        ValueError: If a requested channel is not a STRING channel.
    """
    unknown = sorted(set(channels) - set(STRING_CHANNELS))
    if unknown:
        raise ValueError(f"unknown STRING channel(s) {unknown}; {STRING_CHANNELS}")
    protein_to_symbol = _protein_symbols(alias_path, set(universe_genes))
    links = pd.read_csv(
        detailed_path, sep=" ", usecols=["protein1", "protein2", *channels]
    )
    score = combine_channel_scores(links[list(channels)].to_numpy())
    keep = score >= confidence_threshold
    frame = pd.DataFrame(
        {
            "a": links.loc[keep, "protein1"].map(protein_to_symbol).to_numpy(),
            "b": links.loc[keep, "protein2"].map(protein_to_symbol).to_numpy(),
            "score": score[keep] / 1000.0,
        }
    ).dropna(subset=["a", "b"])
    frame = frame[frame["a"] != frame["b"]]
    ordered = pd.DataFrame(
        {
            "a": np.minimum(frame["a"].to_numpy(), frame["b"].to_numpy()),
            "b": np.maximum(frame["a"].to_numpy(), frame["b"].to_numpy()),
            "score": frame["score"].to_numpy(),
        }
    )
    best = ordered.groupby(["a", "b"])["score"].max()
    return {
        (str(a), str(b)): float(value)
        for a, b, value in zip(
            best.index.get_level_values(0),
            best.index.get_level_values(1),
            best.to_numpy(),
            strict=True,
        )
    }


class StringGraphCache:
    """Parse STRING once over a gene universe, then serve induced subgraphs.

    Gene selection is fold-local, so the graph must be rebuilt for every fold.
    Re-parsing the interactome each time would dominate runtime; this class
    pays that cost once and reduces per-fold work to a set intersection.

    Attributes:
        confidence_threshold: Score floor applied when the edges were parsed.
        edges: Undirected edges over the universe, as returned by
            :func:`parse_string_edges`.
        weights: Confidence of each edge in ``(0, 1]``; 1 for every edge when
            the source carried no score.
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
        self.weights = dict.fromkeys(self.edges, 1.0)
        self.n_universe_edges = len(self.edges)
        logger.info(
            "STRING: %d undirected edges among the %d-gene universe "
            "(combined_score >= %d)",
            self.n_universe_edges,
            len(set(universe_genes)),
            confidence_threshold,
        )

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str]],
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        weights: Mapping[tuple[str, str], float] | None = None,
    ) -> StringGraphCache:
        """Build a cache from edges that were parsed earlier.

        Parsing needs the 800 MB STRING pair, which is far larger than the
        induced universe it produces. This constructor lets that cost be paid
        once, saved with :func:`save_string_edges`, and reloaded anywhere.

        Args:
            edges: Undirected ``(gene_a, gene_b)`` pairs.
            confidence_threshold: Score floor that produced ``edges``, recorded
                for provenance only.
            weights: Confidence per edge, keyed like ``edges``; every edge
                weighs 1 when omitted.

        Returns:
            A cache serving induced subgraphs over those edges.
        """
        cache = cls.__new__(cls)
        cache.confidence_threshold = confidence_threshold
        cache.edges = {(a, b) if a <= b else (b, a) for a, b in edges if a != b}
        cache.weights = {
            edge: 1.0 if weights is None else float(weights.get(edge, 1.0))
            for edge in cache.edges
        }
        cache.n_universe_edges = len(cache.edges)
        return cache

    def induced_adjacency(
        self, selected_genes: Sequence[str], weighted: bool = False
    ) -> np.ndarray:
        """Build the adjacency of the subgraph induced on selected genes.

        Isolated nodes are retained so that row ``i`` always corresponds to
        ``selected_genes[i]`` — downstream shape contracts depend on it.

        Args:
            selected_genes: Fold-local gene symbols, defining the node order.
            weighted: Fill each edge with its confidence instead of 1.

        Returns:
            Symmetric adjacency of shape ``(n_selected, n_selected)`` with a
            zero diagonal, binary or confidence-weighted.
        """
        position = {gene: i for i, gene in enumerate(selected_genes)}
        selected = set(selected_genes)
        n_selected = len(selected_genes)

        adjacency = np.zeros((n_selected, n_selected), dtype=np.float32)
        for gene_a, gene_b in self.edges:
            if gene_a in selected and gene_b in selected:
                i, j = position[gene_a], position[gene_b]
                weight = self.weights[(gene_a, gene_b)] if weighted else 1.0
                adjacency[i, j] = weight
                adjacency[j, i] = weight
        return adjacency


def save_string_edges(
    edges: Iterable[tuple[str, str]],
    path: str | Path,
    weights: Mapping[tuple[str, str], float] | None = None,
) -> Path:
    """Write undirected gene-symbol edges as a TSV.

    Args:
        edges: Undirected ``(gene_a, gene_b)`` pairs.
        path: Destination file. Parent directories are created.
        weights: Confidence per edge, written as a ``score`` column when given.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((a, b) if a <= b else (b, a) for a, b in edges if a != b)
    frame = pd.DataFrame(ordered, columns=["gene_a", "gene_b"])
    if weights is not None:
        frame["score"] = [round(weights[edge], 4) for edge in ordered]
    frame.to_csv(destination, sep=TSV_SEPARATOR, index=False)
    logger.info("wrote %d STRING edges to %s", len(ordered), destination)
    return destination


def load_string_edges(path: str | Path) -> set[tuple[str, str]]:
    """Read edges written by :func:`save_string_edges`.

    Args:
        path: File to read.

    Returns:
        Deduplicated ``(gene_a, gene_b)`` pairs with ``gene_a <= gene_b``.
    """
    frame = pd.read_csv(path, sep=TSV_SEPARATOR)
    return {
        (a, b) if a <= b else (b, a)
        for a, b in zip(frame["gene_a"], frame["gene_b"], strict=True)
        if a != b
    }


def load_string_edge_weights(path: str | Path) -> dict[tuple[str, str], float]:
    """Read the confidence of each edge written by :func:`save_string_edges`.

    Args:
        path: File to read.

    Returns:
        Confidence per ``(gene_a, gene_b)`` pair with ``gene_a <= gene_b``; 1
        for every edge when the file carries no ``score`` column.
    """
    frame = pd.read_csv(path, sep=TSV_SEPARATOR)
    scores = frame["score"] if "score" in frame.columns else pd.Series(1.0, frame.index)
    return {
        ((a, b) if a <= b else (b, a)): float(score)
        for a, b, score in zip(frame["gene_a"], frame["gene_b"], scores, strict=True)
        if a != b
    }
