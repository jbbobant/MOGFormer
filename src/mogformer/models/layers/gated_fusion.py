"""Gated intra-gene fusion, a cheaper alternative to intra-gene attention.

Drop-in replacement for
:class:`~mogformer.models.layers.mini_transformer.MiniTransformer`,
with the same call signature and the same dropout semantics, so the two can be
swapped in a single-variable ablation.

The gates compete: a softmax over the three modalities means raising one
modality's weight necessarily lowers another's. That is deliberate — independent
sigmoid gates cannot express silencing, because nothing forces methylation's
influence to come at expression's expense. Competition makes "methylation
suppressed this gene's expression" a state the layer can actually represent.

Interpretability differs accordingly: instead of a four-by-four attention matrix
this layer returns the gate weight per modality, occupying the same return slot
so the enclosing model's contract is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mogformer.models.layers.modality_dropout import ModalityDropout


class GatedFusion(nn.Module):
    """Fuse per-gene modality tokens by a competing softmax gate.

    Attributes:
        d: Token width.
        modality_dropout: The shared exclusive-dropout module.
    """

    def __init__(
        self,
        d: int = 64,
        dropout: float = 0.1,
        rna_dropout_prob: float = 0.3,
        cnv_dropout_prob: float = 0.15,
        meth_dropout_prob: float = 0.15,
        unimodal_dropout_fill: str = "zero",
    ) -> None:
        """Build the gate scorer and the output projection.

        Args:
            d: Token width.
            dropout: Dropout applied to the fused output.
            rna_dropout_prob: Probability of hiding expression per gene.
            cnv_dropout_prob: Probability of hiding copy number per gene.
            meth_dropout_prob: Probability of hiding methylation per gene.
            unimodal_dropout_fill: One of
                :data:`~mogformer.models.layers.modality_dropout.DROPOUT_FILLS`.
        """
        super().__init__()
        self.d = d

        self.modality_dropout = ModalityDropout(
            d=d,
            rna_dropout_prob=rna_dropout_prob,
            cnv_dropout_prob=cnv_dropout_prob,
            meth_dropout_prob=meth_dropout_prob,
            fill=unimodal_dropout_fill,
        )

        # Scored from all three tokens jointly, so methylation can drive
        # expression's weight down.
        self.gate_score = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3)
        )
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        z_rna: torch.Tensor,
        z_cnv: torch.Tensor,
        z_methy: torch.Tensor,
        eval_mask: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse the three modality streams into one token per gene.

        Args:
            z_rna: Expression tokens, shape ``(batch, n_genes, d)``.
            z_cnv: Copy-number tokens, same shape.
            z_methy: Methylation tokens, same shape.
            eval_mask: Modalities to hide deterministically at evaluation time.

        Returns:
            Tuple of the fused gene tokens, shape ``(batch, n_genes, d)``, and
            the gate weights, shape ``(batch, n_genes, 3)`` in modality order,
            summing to one per gene.
        """
        z_rna, z_cnv, z_methy = self.modality_dropout(
            z_rna, z_cnv, z_methy, eval_mask=eval_mask
        )

        context = torch.cat([z_rna, z_cnv, z_methy], dim=-1)
        weights = torch.softmax(self.gate_score(context), dim=-1)

        fused = (
            weights[..., 0:1] * z_rna
            + weights[..., 1:2] * z_cnv
            + weights[..., 2:3] * z_methy
        )
        return self.out_drop(self.proj(fused)), weights
