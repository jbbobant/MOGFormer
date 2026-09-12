"""Shortest-path distances over the gene interaction graph.

The global transformer biases its attention by the hop distance between genes,
so every fold needs an integer distance matrix over that fold's selected genes.
Distances are truncated: the bias table has one learnable scalar per distance
bucket, and unbounded distances would give it an unbounded vocabulary.
"""

from __future__ import annotations

import torch


def compute_shortest_path_matrix(
    adjacency: torch.Tensor,
    max_distance: int = 5,
) -> torch.Tensor:
    """Compute the truncated all-pairs shortest-path matrix of a gene graph.

    Runs Floyd–Warshall over the binarised adjacency. Distances greater than
    ``max_distance`` — including pairs in different connected components — are
    collapsed onto the single bucket ``max_distance + 1``, which the attention
    bias treats as "far or unreachable".

    Args:
        adjacency: Square adjacency matrix, shape ``(n_genes, n_genes)``. Any
            strictly positive entry is treated as an edge; weights are ignored.
        max_distance: Largest hop count represented exactly. Must be positive.

    Returns:
        Integer distance matrix of shape ``(n_genes, n_genes)`` on the same
        device as ``adjacency``, with values in ``[0, max_distance + 1]`` and a
        zero diagonal.

    Raises:
        ValueError: If ``adjacency`` is not a square 2-D tensor, or if
            ``max_distance`` is not positive.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be a square 2-D tensor, got shape {tuple(adjacency.shape)}"
        )
    if max_distance < 1:
        raise ValueError(f"max_distance must be >= 1, got {max_distance}")

    binary = (adjacency > 0).float()
    n_genes = binary.shape[0]

    distances = torch.full((n_genes, n_genes), float("inf"), device=adjacency.device)
    distances.fill_diagonal_(0)
    distances[binary == 1] = 1

    # Floyd-Warshall: relax every pair through intermediate node k.
    for k in range(n_genes):
        distances = torch.minimum(
            distances, distances[:, k : k + 1] + distances[k : k + 1, :]
        )

    distances[distances > max_distance] = max_distance + 1
    return distances.long()
