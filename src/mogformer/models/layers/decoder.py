"""Dual-head decoder reconstructing masked measurements during pretraining.

Reconstruction is addressed: to predict a value the decoder must be told which
gene and which modality it is predicting, and that address has to be the same
one the encoder used or the two speak different languages. The decoder
therefore holds no gene or modality embeddings of its own — it receives the
encoder's.

Two heads read the same address from different directions, and the contrast
between them is informative rather than redundant:

* the **global** head sees only the patient-level summary vector, modulating the
  address by feature-wise scale and shift, so it can only know what the whole
  tumor state implies about a gene;
* the **local** head additionally sees that gene's own final-layer row, so it
  knows what the graph routed to that gene specifically.

The interventional probes read the local head, because a perturbation's effect
should arrive through the gene's own row. The global head is kept as a
robustness check: an effect visible only there did not travel through the graph.
"""

from __future__ import annotations

import torch
from torch import nn


class DualHeadDecoder(nn.Module):
    """Reconstruct per-gene, per-modality values from a shared address grid.

    Attributes:
        p_dec: Learnable decoder-role token, distinguishing the reconstruction
            address from the encoder's own embeddings.
    """

    def __init__(self, d: int = 128) -> None:
        """Build the two heads and the modulation projections.

        Args:
            d: Token width; must match the encoder's.
        """
        super().__init__()
        self.p_dec = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.p_dec, std=0.02)

        self.mlp_gamma = self._build_block(d)
        self.mlp_beta = self._build_block(d)
        self.ffn_global = self._build_block(d)
        self.lin_global = nn.Linear(d, 1)
        self.ffn_local = self._build_block(d)
        self.lin_local = nn.Linear(d, 1)

    @staticmethod
    def _build_block(d: int) -> nn.Sequential:
        """Return the two-layer block used throughout the decoder."""
        return nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(
        self,
        summary: torch.Tensor,
        hidden_last: torch.Tensor,
        gene_embedding: torch.Tensor,
        modality_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict every gene-by-modality value from both heads.

        Args:
            summary: Patient summary token, shape ``(batch, d)``.
            hidden_last: Final encoder states including the summary token at
                index 0, shape ``(batch, n_genes + 1, d)``.
            gene_embedding: The encoder's gene identity embeddings, shape
                ``(n_genes, d)``.
            modality_embedding: The encoder's modality embeddings, shape
                ``(3, d)`` in modality order.

        Returns:
            Tuple of global-head and local-head predictions, each of shape
            ``(batch, n_genes, 3)``.
        """
        gene_rows = hidden_last[:, 1:, :]

        # Address grid: one query per (gene, modality) pair.
        address = (
            gene_embedding[:, None, :]
            + modality_embedding[None, :, :]
            + self.p_dec[None, None, :]
        )

        # Global head: the patient enters only as a feature-wise scale and shift.
        gamma = self.mlp_gamma(summary)
        beta = self.mlp_beta(summary)
        modulated = (
            self.ffn_global(address)[None] * gamma[:, None, None, :]
            + beta[:, None, None, :]
        )
        global_prediction = self.lin_global(modulated).squeeze(-1)

        # Local head: the address is offset by the gene's own final state.
        local_input = address[None] + gene_rows[:, :, None, :]
        local_prediction = self.lin_local(self.ffn_local(local_input)).squeeze(-1)

        return global_prediction, local_prediction
