"""Exclusive modality dropout, shared by every intra-gene fusion layer.

Expression is by far the densest and most predictive of the three modalities,
so a fusion layer left to itself learns to read expression and ignore the rest.
Randomly hiding one modality per gene forces the layer to reconstruct the
missing channel from the other two, which is what makes cross-modal routing
appear at all.

At most one modality is dropped per gene, never two: with two hidden there is
too little left to impute from, and the layer learns to output a constant.

The same block previously existed verbatim in both the attention-based and the
gated fusion layer. It lives here once so the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

#: Fill strategies for a dropped modality.
DROPOUT_FILLS: tuple[str, ...] = ("zero", "mask_token")


class ModalityDropout(nn.Module):
    """Hide at most one modality per gene during training.

    Attributes:
        rna_dropout_prob: Probability of hiding expression for a given gene.
        cnv_dropout_prob: Probability of hiding copy number.
        meth_dropout_prob: Probability of hiding methylation.
        fill: Active fill strategy, one of :data:`DROPOUT_FILLS`.
    """

    def __init__(
        self,
        d: int,
        rna_dropout_prob: float = 0.3,
        cnv_dropout_prob: float = 0.15,
        meth_dropout_prob: float = 0.15,
        fill: str = "zero",
        mask_token_std: float = 0.10,
    ) -> None:
        """Store the probabilities and, if needed, build the mask tokens.

        Args:
            d: Token width.
            rna_dropout_prob: Probability of hiding expression. Higher than the
                others by default, because expression otherwise dominates.
            cnv_dropout_prob: Probability of hiding copy number.
            meth_dropout_prob: Probability of hiding methylation.
            fill: ``"zero"`` replaces a hidden token with zeros;
                ``"mask_token"`` replaces it with a learnable per-modality token,
                which also enables deterministic evaluation-time masking.
            mask_token_std: Initialisation scale of the mask tokens.

        Raises:
            ValueError: If ``fill`` is unknown or the probabilities sum above 1.
        """
        super().__init__()
        if fill not in DROPOUT_FILLS:
            raise ValueError(
                f"unknown dropout fill {fill!r}; available: {list(DROPOUT_FILLS)}"
            )
        total = rna_dropout_prob + cnv_dropout_prob + meth_dropout_prob
        if total > 1.0:
            raise ValueError(
                f"modality dropout probabilities sum to {total:.3f}, which "
                "exceeds 1.0; they partition a single uniform draw"
            )

        self.rna_dropout_prob = rna_dropout_prob
        self.cnv_dropout_prob = cnv_dropout_prob
        self.meth_dropout_prob = meth_dropout_prob
        self.fill = fill

        if fill == "mask_token":
            self.mask_rna = nn.Parameter(torch.zeros(1, 1, d))
            self.mask_cnv = nn.Parameter(torch.zeros(1, 1, d))
            self.mask_methy = nn.Parameter(torch.zeros(1, 1, d))
            for token in (self.mask_rna, self.mask_cnv, self.mask_methy):
                nn.init.normal_(token, std=mask_token_std)

    def forward(
        self,
        z_rna: torch.Tensor,
        z_cnv: torch.Tensor,
        z_methy: torch.Tensor,
        eval_mask: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply stochastic dropout in training, or deterministic masking in eval.

        A single uniform draw per gene is partitioned across the three
        probabilities, which is what makes the choice exclusive.

        Args:
            z_rna: Expression tokens, shape ``(batch, n_genes, d)``.
            z_cnv: Copy-number tokens, same shape.
            z_methy: Methylation tokens, same shape.
            eval_mask: Modality names to hide deterministically while in
                evaluation mode, used by the modality-ablation probes. Ignored
                during training.

        Returns:
            The three token streams, with hidden modalities replaced.

        Raises:
            ValueError: If ``eval_mask`` is used without mask tokens, or names
                an unknown modality.
        """
        if self.training:
            draw = torch.rand(z_rna.shape[0], z_rna.shape[1], 1, device=z_rna.device)
            rna_edge = self.rna_dropout_prob
            cnv_edge = rna_edge + self.cnv_dropout_prob
            methy_edge = cnv_edge + self.meth_dropout_prob

            drop_rna = draw < rna_edge
            drop_cnv = (draw >= rna_edge) & (draw < cnv_edge)
            drop_methy = (draw >= cnv_edge) & (draw < methy_edge)

            if self.fill == "mask_token":
                z_rna = torch.where(
                    drop_rna.expand_as(z_rna), self.mask_rna.expand_as(z_rna), z_rna
                )
                z_cnv = torch.where(
                    drop_cnv.expand_as(z_cnv), self.mask_cnv.expand_as(z_cnv), z_cnv
                )
                z_methy = torch.where(
                    drop_methy.expand_as(z_methy),
                    self.mask_methy.expand_as(z_methy),
                    z_methy,
                )
            else:
                z_rna = z_rna.masked_fill(drop_rna, 0.0)
                z_cnv = z_cnv.masked_fill(drop_cnv, 0.0)
                z_methy = z_methy.masked_fill(drop_methy, 0.0)

            return z_rna, z_cnv, z_methy

        if not eval_mask:
            return z_rna, z_cnv, z_methy

        if self.fill != "mask_token":
            raise ValueError(
                "eval_mask requires fill='mask_token'; zero-filling at "
                "evaluation time is not a trained state of the model"
            )
        unknown = set(eval_mask) - {"rna", "cnv", "methy"}
        if unknown:
            raise ValueError(f"eval_mask names unknown modalities: {sorted(unknown)}")

        if "rna" in eval_mask:
            z_rna = self.mask_rna.expand_as(z_rna)
        if "cnv" in eval_mask:
            z_cnv = self.mask_cnv.expand_as(z_cnv)
        if "methy" in eval_mask:
            z_methy = self.mask_methy.expand_as(z_methy)
        return z_rna, z_cnv, z_methy
