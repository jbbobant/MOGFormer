"""CollecTRI signed regulatory edges as a directed gene graph.

The STRING graph is undirected and unsigned: it says two proteins interact, not
who regulates whom or in which direction. CollecTRI supplies signed, directed
transcription-factor to target edges, which is what the trans probe needs in
order to ask whether a perturbation propagates with the sign biology predicts.

Mirrors :mod:`mogformer.graph.string_graph`: parse once over the universe, serve
induced subgraphs per fold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Orientations accepted by :meth:`GRNSignedCache.induced_signed_adjacency`.
GRN_DIRECTIONS: tuple[str, ...] = ("target_attends_tf", "tf_attends_target")


class GRNSignedCache:
    """Parse CollecTRI once over a gene universe, then serve induced subgraphs.

    Edges are directed and signed: ``+1`` for activation, ``-1`` for repression.
    Ambiguous edges are expected to have been resolved during preprocessing.

    Attributes:
        edges: ``(transcription_factor, target, sign)`` triples over the
            universe.
        n_universe_edges: Number of cached edges.
    """

    def __init__(self, grn_path: str, universe_genes: Sequence[str]) -> None:
        """Parse the regulatory network over ``universe_genes``.

        Args:
            grn_path: Path to the CollecTRI edge table (tab-separated, with
                ``source``, ``target`` and ``sign`` columns, sign in
                ``{-1, +1}``).
            universe_genes: Gene symbols the graph may contain.
        """
        universe = set(universe_genes)

        table = pd.read_csv(grn_path, sep="\t")
        table = table[table["source"].isin(universe) & table["target"].isin(universe)]
        table = table[table["source"] != table["target"]]

        self.edges: list[tuple[str, str, int]] = list(
            zip(
                table["source"].to_numpy(),
                table["target"].to_numpy(),
                table["sign"].astype(int).to_numpy(),
                strict=True,
            )
        )
        self.n_universe_edges = len(self.edges)

        n_activating = sum(1 for *_, sign in self.edges if sign > 0)
        logger.info(
            "CollecTRI: %d directed signed edges among the %d-gene universe "
            "(%d activating / %d repressing)",
            self.n_universe_edges,
            len(universe),
            n_activating,
            self.n_universe_edges - n_activating,
        )

    def induced_signed_adjacency(
        self,
        selected_genes: Sequence[str],
        direction: str = "target_attends_tf",
        use_sign: bool = True,
    ) -> np.ndarray:
        """Build the signed adjacency of the subgraph induced on selected genes.

        Under the default ``"target_attends_tf"`` orientation the matrix reads
        row-wise as "target ``i`` attends regulator ``j``"::

            matrix[i, j] = +1   gene j activates gene i
            matrix[i, j] = -1   gene j represses gene i
            matrix[i, j] =  0   no known regulatory edge

        Isolated nodes are retained so row ``i`` always corresponds to
        ``selected_genes[i]``.

        Args:
            selected_genes: Fold-local gene symbols, defining the node order.
            direction: Orientation convention, one of :data:`GRN_DIRECTIONS`.
            use_sign: When False, every edge is recorded as ``+1``, giving an
                unsigned directed graph. Used to separate the contribution of
                edge existence from edge sign.

        Returns:
            Signed directed adjacency of shape ``(n_selected, n_selected)``.

        Raises:
            ValueError: If ``direction`` is not in :data:`GRN_DIRECTIONS`.
        """
        if direction not in GRN_DIRECTIONS:
            raise ValueError(
                f"unknown grn direction {direction!r}; "
                f"available: {list(GRN_DIRECTIONS)}"
            )

        position = {gene: i for i, gene in enumerate(selected_genes)}
        selected = set(selected_genes)
        n_selected = len(selected_genes)

        matrix = np.zeros((n_selected, n_selected), dtype=np.float32)
        for transcription_factor, target, sign in self.edges:
            if transcription_factor not in selected or target not in selected:
                continue
            value = float(sign) if use_sign else 1.0
            if direction == "target_attends_tf":
                matrix[position[target], position[transcription_factor]] = value
            else:
                matrix[position[transcription_factor], position[target]] = value
        return matrix
