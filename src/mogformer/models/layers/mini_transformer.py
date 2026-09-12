"""Intra-gene attention fusing three modalities into one gene token.

A gene's expression is bounded by its copy number and regulated by its promoter
methylation. Concatenating the three treats them as independent, which they are
not, so this layer runs a small four-token attention within each gene — a
summary token plus one token per modality — and reads the updated summary token
as the gene's unified representation.

The four-by-four attention matrix is also the model's finest-grained
interpretability handle: it shows which modality drove a given gene's state,
without any post-hoc attribution method.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mogformer.models.layers.modality_dropout import ModalityDropout


class MiniTransformer(nn.Module):
    """Fuse per-gene modality tokens by attention over a four-token sequence.

    Attributes:
        d: Token width.
        modality_dropout: The shared exclusive-dropout module.
        z_cls: Learnable per-gene summary token.
    """

    def __init__(
        self,
        d: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        rna_dropout_prob: float = 0.3,
        cnv_dropout_prob: float = 0.15,
        meth_dropout_prob: float = 0.15,
        unimodal_dropout_fill: str = "zero",
    ) -> None:
        """Build the summary token, attention and feed-forward sublayers.

        Args:
            d: Token width.
            num_heads: Attention heads for the intra-gene attention.
            dropout: Dropout on the attention weights.
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

        # Broadcast across batch and genes: every gene gets its own summary token.
        self.z_cls = nn.Parameter(torch.randn(1, 1, 1, d))
        nn.init.normal_(self.z_cls, mean=0.0, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

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
            the intra-gene attention weights, shape
            ``(batch * n_genes, 4, 4)`` over ``[summary, rna, cnv, methy]``.
        """
        batch, n_genes, width = z_rna.shape

        z_rna, z_cnv, z_methy = self.modality_dropout(
            z_rna, z_cnv, z_methy, eval_mask=eval_mask
        )

        modalities = torch.stack([z_rna, z_cnv, z_methy], dim=2)
        summary = self.z_cls.expand(batch, n_genes, 1, width)
        # Sequence order: 0 summary, 1 rna, 2 cnv, 3 methy.
        tokens = torch.cat([summary, modalities], dim=2)

        # Multi-head attention takes one batch dimension, so fold genes into it.
        flat = tokens.view(batch * n_genes, 4, width)

        normed = self.norm1(flat)
        attended, attention_weights = self.attention(
            query=normed, key=normed, value=normed, need_weights=True
        )
        flat = flat + attended
        flat = flat + self.ffn(self.norm2(flat))

        fused = flat.view(batch, n_genes, 4, width)[:, :, 0, :]
        return fused, attention_weights
