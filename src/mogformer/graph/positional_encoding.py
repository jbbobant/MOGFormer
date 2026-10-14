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
        weighted: Whether random walks follow edge weights.
    """

    def __init__(
        self, pe_dim: int = 32, method: str = "rwpe", weighted: bool = False
    ) -> None:
        """Initialise the encoder.

        Args:
            pe_dim: Number of encoding dimensions per node. For ``"rwpe"`` this
                is the number of random-walk steps; for ``"laplacian"`` it is
                the number of non-trivial eigenvectors retained.
            method: Encoding scheme, one of :data:`PE_METHODS`.
            weighted: Let random walks step along an edge with probability
                proportional to its weight (edge confidence), rather than
                uniformly over neighbours. Only read by ``"rwpe"``.

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
        self.weighted = weighted

    @torch.no_grad()
    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Encode every node of ``adjacency``.

        Args:
            adjacency: Square adjacency, shape ``(n_genes, n_genes)``. Any
                strictly positive entry is an edge; its value is its weight
                when ``weighted`` is set.

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
            return self._random_walk(
                adjacency.float().clamp(min=0) if self.weighted else binary
            )
        return self._laplacian(binary)

    def _random_walk(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Return per-node random-walk return probabilities.

        With ``P = D^-1 A`` the row-stochastic transition matrix (``D`` the
        weighted degree when ``A`` carries weights), dimension
        ``k`` of node ``i`` is ``[P^(k+1)]_ii``: the probability that a walk
        starting at ``i`` is back at ``i`` after ``k + 1`` steps. Isolated nodes
        get a clamped degree so the inverse stays finite.

        Args:
            adjacency: Non-negative adjacency, shape ``(n_genes, n_genes)``.

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


def nearest_encoding_distance(encodings: np.ndarray) -> np.ndarray:
    """Measure how far each node's encoding is from its closest other node's.

    A model without a per-gene identity embedding can tell two genes apart only
    through their encodings, so genes whose encodings nearly coincide are
    effectively interchangeable to it. Distances are expressed as a fraction of
    the encodings' spread (mean distance to their centroid), which makes them
    comparable across folds and schemes.

    Args:
        encodings: Node encodings, shape ``(n_nodes, pe_dim)``, at least two.

    Returns:
        Relative nearest-neighbour distance per node, shape ``(n_nodes,)``.
        Zero marks a node whose encoding is identical to another node's.

    Raises:
        ValueError: If fewer than two nodes are given or the encodings do not
            vary at all.
    """
    values = np.asarray(encodings, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"need at least two encodings, got shape {values.shape}")
    spread = float(np.linalg.norm(values - values.mean(axis=0), axis=1).mean())
    if spread == 0.0:
        raise ValueError("every encoding is identical; distances are undefined")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1) / spread
