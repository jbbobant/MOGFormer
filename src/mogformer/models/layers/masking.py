"""Masking strategy for masked multi-modal pretraining.

Two kinds of mask are drawn per patient, and the distinction is the point:

* **whole-gene** masking hides all three modalities of a gene, so the only way
  to reconstruct it is through the graph, from that gene's neighbours;
* **single-modality** masking hides one channel of a gene whose other two remain
  visible, so the reconstruction must come from within the gene.

Keeping the two sets disjoint means an interventional probe can attribute a
recovered value to graph routing or to intra-gene cross-talk, rather than to an
unresolvable mixture.

Expression is weighted more heavily among the single-modality masks because it
is the channel the model would otherwise lean on.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class MaskedMultiModalMasker(nn.Module):
    """Draw disjoint whole-gene and single-modality masks per patient.

    Attributes:
        mask_gene_frac: Fraction of genes hidden across all three modalities.
        mask_modality_frac: Probability that a remaining gene has exactly one
            modality hidden.
    """

    def __init__(
        self,
        mask_gene_frac: float = 0.15,
        mask_modality_frac: float = 0.15,
        mask_modality_weights: tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        """Store the masking rates and the per-modality weighting.

        Args:
            mask_gene_frac: Fraction of genes to hide entirely, per patient.
            mask_modality_frac: Probability of hiding one modality of a gene not
                already wholly hidden.
            mask_modality_weights: Relative weights over ``(rna, cnv, methy)``
                when choosing which single modality to hide. Expression is
                weighted up by default.

        Raises:
            ValueError: If either fraction falls outside ``[0, 1]``, or if the
                weights are not all positive.
        """
        super().__init__()
        for name, value in (
            ("mask_gene_frac", mask_gene_frac),
            ("mask_modality_frac", mask_modality_frac),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if any(weight <= 0 for weight in mask_modality_weights):
            raise ValueError(
                f"mask_modality_weights must all be positive, got "
                f"{mask_modality_weights}"
            )

        self.mask_gene_frac = mask_gene_frac
        self.mask_modality_frac = mask_modality_frac
        self.register_buffer(
            "modality_weights",
            torch.tensor(mask_modality_weights, dtype=torch.float32),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a mask for one batch of patients.

        Args:
            x: Per-gene values, shape ``(batch, n_genes, 3)`` in modality order.

        Returns:
            Tuple of the boolean mask, shape ``(batch, n_genes, 3)`` and True
            where a value is hidden, and the reconstruction targets, which are a
            copy of ``x`` taken before any masking is applied downstream.
        """
        batch, n_genes, _ = x.shape
        device = x.device

        targets = x.clone()
        mask = torch.zeros(batch, n_genes, 3, dtype=torch.bool, device=device)

        n_whole_genes = math.floor(self.mask_gene_frac * n_genes)
        gene_masked = torch.zeros(batch, n_genes, dtype=torch.bool, device=device)
        if n_whole_genes > 0:
            # topk over uniform noise draws distinct genes per patient.
            chosen = (
                torch.rand(batch, n_genes, device=device)
                .topk(n_whole_genes, dim=1)
                .indices
            )
            gene_masked.scatter_(1, chosen, True)
            mask |= gene_masked.unsqueeze(-1)

        # Single-modality masks are drawn only among genes still fully visible,
        # which is what keeps the two sets disjoint.
        eligible = (
            torch.rand(batch, n_genes, device=device) < self.mask_modality_frac
        ) & (~gene_masked)
        weights = cast(torch.Tensor, self.modality_weights)
        probabilities = weights / weights.sum()
        picked = torch.multinomial(
            probabilities, batch * n_genes, replacement=True
        ).view(batch, n_genes)
        one_hot = F.one_hot(picked, num_classes=3).bool()
        mask |= eligible.unsqueeze(-1) & one_hot

        return mask, targets
