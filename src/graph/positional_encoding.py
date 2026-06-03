"""
graph/positional_encoding.py — graph positional encodings computation. 

    pe = GraphPositionalEncoding(pe_dim=32, method="rwpe")(base_adj)   # (N, pe_dim)

Supported methods: 
"rwpe" (random-walk return probabilities), 
"laplacian" (normalized-Laplacian eigenvectors, trivial vector dropped)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GraphPositionalEncoding(nn.Module):
    def __init__(self, pe_dim: int = 32, method: str = "rwpe"):
        super().__init__()
        self.pe_dim = pe_dim
        self.method = method
        self._methods = {"rwpe": self._rwpe, "laplacian": self._laplacian}
        if method not in self._methods:
            raise ValueError(f"unknown PE method '{method}'; "
                             f"available: {list(self._methods)}")

    @torch.no_grad()
    def forward(self, base_adj: torch.Tensor) -> torch.Tensor:
        A = (base_adj > 0).float()
        return self._methods[self.method](A)

    # ---- random-walk PE  ----
    def _rwpe(self, A: torch.Tensor) -> torch.Tensor:
        deg = torch.clamp(A.sum(dim=1), min=1e-12)
        P = torch.matmul(torch.diag(1.0 / deg), A)        # D^{-1} A
        outs, Pk = [], P
        for _ in range(self.pe_dim):
            outs.append(torch.diagonal(Pk))               # return probabilities
            Pk = torch.matmul(Pk, P)
        return torch.stack(outs, dim=1)                   # (N, pe_dim)

    # ---- normalized-Laplacian eigenvectors (from legacy graph_utils) ----
    def _laplacian(self, A: torch.Tensor) -> torch.Tensor:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        A_np = A.detach().cpu().numpy().astype(float)
        n = A_np.shape[0]
        deg = A_np.sum(axis=1)
        deg[deg == 0] = 1e-12
        Dinv = sp.diags(1.0 / np.sqrt(deg))
        L = sp.eye(n) - Dinv @ sp.csr_matrix(A_np) @ Dinv
        k = min(self.pe_dim + 1, n - 1)
        if k < 1:
            return torch.zeros((n, self.pe_dim), dtype=torch.float32, device=A.device)
        _, vecs = eigsh(L, k=k, which="SM", tol=1e-2)
        pe = vecs[:, 1:self.pe_dim + 1]                   # drop trivial eigenvector
        if pe.shape[1] < self.pe_dim:                     # pad small graphs
            pe = np.pad(pe, ((0, 0), (0, self.pe_dim - pe.shape[1])))
        return torch.tensor(pe, dtype=torch.float32, device=A.device)
