"""Structural positional encodings for gene tokens.

Attention over a gene sequence is permutation-invariant on its own, so each gene
token carries an encoding of its topological role in the fold's interaction
graph. Two schemes are available and interchangeable: random-walk return
probabilities, and the low-frequency eigenvectors of the normalised Laplacian.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Encoding schemes accepted by :class:`GraphPositionalEncoding`.
PE_METHODS: tuple[str, ...] = ("rwpe", "laplacian")


class GraphPositionalEncoding(nn.Module):
    """Compute a fixed structural encoding for every node of a gene graph.

    The module holds no learnable parameters: it is a deterministic function of
    the adjacency, recomputed per fold because gene selection is fold-local.

    Attributes:
        pe_dim: Width of the produced encoding.
        method: Active scheme, one of :data:`PE_METHODS`.
    """

    def __init__(self, pe_dim: int = 32, method: str = "rwpe") -> None:
        """Initialise the encoder.

        Args:
            pe_dim: Number of encoding dimensions per node. For ``"rwpe"`` this
                is the number of random-walk steps; for ``"laplacian"`` it is
                the number of non-trivial eigenvectors retained.
            method: Encoding scheme, one of :data:`PE_METHODS`.

        Raises:
            ValueError: If ``method`` is unknown or ``pe_dim`` is not positive.
        """
        super().__init__()
        if pe_dim < 1:
            raise ValueError(f"pe_dim must be >= 1, got {pe_dim}")
        if method not in PE_METHODS:
            raise ValueError(
                f"unknown positional encoding method {method!r}; "
                f"available: {list(PE_METHODS)}"
            )
        self.pe_dim = pe_dim
        self.method = method

    @torch.no_grad()
    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Encode every node of ``adjacency``.

        Args:
            adjacency: Square adjacency, shape ``(n_genes, n_genes)``. Any
                strictly positive entry is treated as an edge.

        Returns:
            Float tensor of shape ``(n_genes, pe_dim)`` on the same device as
            ``adjacency``.

        Raises:
            ValueError: If ``adjacency`` is not a square 2-D tensor.
        """
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"adjacency must be a square 2-D tensor, got shape "
                f"{tuple(adjacency.shape)}"
            )
        binary = (adjacency > 0).float()
        if self.method == "rwpe":
            return self._random_walk(binary)
        return self._laplacian(binary)

    def _random_walk(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Return per-node random-walk return probabilities.

        With ``P = D^-1 A`` the row-stochastic transition matrix, dimension
        ``k`` of node ``i`` is ``[P^(k+1)]_ii``: the probability that a walk
        starting at ``i`` is back at ``i`` after ``k + 1`` steps. Isolated nodes
        get a clamped degree so the inverse stays finite.

        Args:
            adjacency: Binarised adjacency, shape ``(n_genes, n_genes)``.

        Returns:
            Float tensor of shape ``(n_genes, pe_dim)``.
        """
        degree = torch.clamp(adjacency.sum(dim=1), min=1e-12)
        transition = torch.matmul(torch.diag(1.0 / degree), adjacency)

        returns: list[torch.Tensor] = []
        walk = transition
        for _ in range(self.pe_dim):
            returns.append(torch.diagonal(walk))
            walk = torch.matmul(walk, transition)
        return torch.stack(returns, dim=1)

    def _laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Return the low-frequency eigenvectors of the normalised Laplacian.

        The trivial constant eigenvector is dropped. Graphs too small to supply
        ``pe_dim`` non-trivial eigenvectors are zero-padded on the right.

        Args:
            adjacency: Binarised adjacency, shape ``(n_genes, n_genes)``.

        Returns:
            Float tensor of shape ``(n_genes, pe_dim)``.
        """
        import scipy.sparse as sparse
        from scipy.sparse.linalg import eigsh

        dense = adjacency.detach().cpu().numpy().astype(float)
        n_genes = dense.shape[0]

        degree = dense.sum(axis=1)
        degree[degree == 0] = 1e-12
        inverse_sqrt_degree = sparse.diags(1.0 / np.sqrt(degree))
        laplacian = (
            sparse.eye(n_genes)
            - inverse_sqrt_degree @ sparse.csr_matrix(dense) @ inverse_sqrt_degree
        )

        n_eigenvectors = min(self.pe_dim + 1, n_genes - 1)
        if n_eigenvectors < 1:
            return torch.zeros(
                (n_genes, self.pe_dim),
                dtype=torch.float32,
                device=adjacency.device,
            )

        _, eigenvectors = eigsh(laplacian, k=n_eigenvectors, which="SM", tol=1e-2)
        encoding = eigenvectors[:, 1 : self.pe_dim + 1]
        if encoding.shape[1] < self.pe_dim:
            encoding = np.pad(encoding, ((0, 0), (0, self.pe_dim - encoding.shape[1])))
        return torch.tensor(encoding, dtype=torch.float32, device=adjacency.device)
