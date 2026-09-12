"""Inter-gene transformer over the whole selected-gene sequence.

Each gene arrives as a single fused token. This layer lets genes exchange
information globally — so long-range dependencies absent from the reference
network can still be learned — while the structural bias keeps known pathways
privileged. A prepended summary token accumulates the whole-tumor state and is
what the classification and reconstruction heads read.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from mogformer.models.layers.structural_attention import StructuralAttentionBlock


class GlobalGraphTransformer(nn.Module):
    """Run structurally biased attention across all genes of a patient.

    Positional encodings are concatenated onto the gene tokens and projected
    back to token width, rather than added, so the encoding cannot be drowned
    out by a large-magnitude token.

    Attributes:
        d: Token width.
        max_distance: Largest hop count represented exactly by the bias.
        summary_distance: Sentinel distance assigned to the summary token's row
            and column, giving it a bias bucket of its own.
        tumor_cls: The learnable summary token.
        layers: The stack of structural attention blocks.
    """

    def __init__(
        self,
        d: int = 64,
        pe_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_distance: int = 5,
        attention_bias_mode: str = "dual",
        dropout: float = 0.1,
        lambda_gate: bool = False,
        use_grn: bool = False,
    ) -> None:
        """Build the projector, the summary token and the attention stack.

        Args:
            d: Token width.
            pe_dim: Width of the graph positional encoding.
            num_heads: Attention heads per block.
            num_layers: Number of stacked blocks.
            dim_feedforward: Hidden width of each block's feed-forward sublayer.
            max_distance: Largest hop count represented exactly by the bias.
            attention_bias_mode: One of
                :data:`~mogformer.models.layers.structural_attention.ATTENTION_BIAS_MODES`.
            dropout: Dropout for attention weights and feed-forward sublayers.
            lambda_gate: Per-head learnable scaling of the distance bias.
            use_grn: Enable the signed regulatory bias in every block. The
                regulatory matrix is still supplied per forward pass.
        """
        super().__init__()
        self.d = d
        self.max_distance = max_distance
        self.summary_distance = max_distance + 2

        self.concat_projector = nn.Linear(d + pe_dim, d)

        self.tumor_cls = nn.Parameter(torch.randn(1, 1, d))
        nn.init.normal_(self.tumor_cls, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                StructuralAttentionBlock(
                    d_model=d,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    max_distance=max_distance,
                    mode=attention_bias_mode,
                    dropout=dropout,
                    lambda_gate=lambda_gate,
                )
                for _ in range(num_layers)
            ]
        )

        if use_grn:
            for layer in self.layers:
                cast(StructuralAttentionBlock, layer).attn.use_grn = True

    def forward(
        self,
        h: torch.Tensor,
        graph_pe: torch.Tensor,
        spd_matrix: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        structural_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contextualise gene tokens against one another.

        Args:
            h: Fused per-gene tokens, shape ``(batch, n_genes, d)``.
            graph_pe: Positional encodings, shape ``(n_genes, pe_dim)``.
            spd_matrix: Integer distances over genes, shape
                ``(n_genes, n_genes)``. Padded here for the summary token.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.
            structural_bias: When False, attention ignores both graph biases.

        Returns:
            Tuple of the summary token state, shape ``(batch, d)``, and the full
            sequence including it, shape ``(batch, n_genes + 1, d)``.
        """
        batch, _, width = h.shape

        expanded_pe = graph_pe.unsqueeze(0).expand(batch, -1, -1)
        tokens = self.concat_projector(torch.cat([h, expanded_pe], dim=-1))

        summary = self.tumor_cls.expand(batch, 1, width)
        sequence = torch.cat([summary, tokens], dim=1)

        # The summary token gets a distance bucket of its own rather than
        # sharing the zero-distance bucket that means "this gene is itself".
        padded_spd = F.pad(spd_matrix, (1, 0, 1, 0), value=self.summary_distance)
        padded_grn = (
            None if grn_matrix is None else F.pad(grn_matrix, (1, 0, 1, 0), value=0)
        )

        state = sequence
        for layer in self.layers:
            state = layer(
                state, padded_spd, padded_grn, structural_bias=structural_bias
            )
        return state[:, 0, :], state
