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

Two harder masks are optional:

* **neighbourhood** masks hide a gene together with its one-hop interaction
  neighbours, so a hidden value cannot simply be copied from a visible
  neighbour and has to come through the patient summary or longer paths;
* **omics-level** masks hide one entire omics of a patient, which is the only
  way to train (and therefore to measure) reconstruction of one omics from the
  other two.
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
        neighbourhood_share: Share of the whole-gene masks drawn as
            neighbourhoods, once :meth:`set_neighbourhoods` has been called.
        omics_mask_prob: Probability that a patient has one whole omics hidden.
    """

    def __init__(
        self,
        mask_gene_frac: float = 0.15,
        mask_modality_frac: float = 0.15,
        mask_modality_weights: tuple[float, float, float] = (2.0, 1.0, 1.0),
        neighbourhood_share: float = 0.0,
        omics_mask_prob: float = 0.0,
    ) -> None:
        """Store the masking rates and the per-modality weighting.

        Args:
            mask_gene_frac: Fraction of genes to hide entirely, per patient.
            mask_modality_frac: Probability of hiding one modality of a gene not
                already wholly hidden.
            mask_modality_weights: Relative weights over ``(rna, cnv, methy)``
                when choosing which single modality to hide. Expression is
                weighted up by default.
            neighbourhood_share: Share of the whole-gene masks drawn as a gene
                plus its one-hop neighbours rather than gene by gene.
            omics_mask_prob: Probability, per patient, of hiding one omics
                (chosen uniformly) for every gene.

        Raises:
            ValueError: If a fraction or probability falls outside ``[0, 1]``,
                or if the weights are not all positive.
        """
        super().__init__()
        for name, value in (
            ("mask_gene_frac", mask_gene_frac),
            ("mask_modality_frac", mask_modality_frac),
            ("neighbourhood_share", neighbourhood_share),
            ("omics_mask_prob", omics_mask_prob),
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
        self.neighbourhood_share = neighbourhood_share
        self.omics_mask_prob = omics_mask_prob
        self.register_buffer(
            "modality_weights",
            torch.tensor(mask_modality_weights, dtype=torch.float32),
        )
        # A property of the run's gene axis, not a learned parameter: kept out
        # of the state dict so checkpoints load whatever the graph.
        self.register_buffer("neighbours", None, persistent=False)

    def set_neighbourhoods(self, adjacency: torch.Tensor) -> None:
        """Give the masker the interaction graph its neighbourhood masks follow.

        Args:
            adjacency: Square adjacency over the gene axis, shape
                ``(n_genes, n_genes)``; any positive entry is an edge.
        """
        self.neighbours = (adjacency > 0).float().fill_diagonal_(0.0)

    def _gene_mask(
        self,
        batch: int,
        n_genes: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Draw the whole-gene mask, with neighbourhoods when configured.

        Returns:
            Boolean mask of shape ``(batch, n_genes)``.
        """
        n_whole_genes = math.floor(self.mask_gene_frac * n_genes)
        gene_masked = torch.zeros(batch, n_genes, dtype=torch.bool, device=device)
        if n_whole_genes == 0:
            return gene_masked
        neighbours = cast(torch.Tensor | None, self.neighbours)
        if neighbours is None or self.neighbourhood_share == 0.0:
            # topk over uniform noise draws distinct genes per patient.
            chosen = (
                torch.rand(batch, n_genes, device=device, generator=generator)
                .topk(n_whole_genes, dim=1)
                .indices
            )
            return gene_masked.scatter_(1, chosen, True)

        # Seeds whose closed neighbourhoods cover, on average, the requested
        # share of the whole-gene budget.
        neighbours = neighbours.to(device)
        mean_size = 1.0 + float(neighbours.sum(dim=1).mean())
        n_seeds = max(1, round(self.neighbourhood_share * n_whole_genes / mean_size))
        seeds = (
            torch.rand(batch, n_genes, device=device, generator=generator)
            .topk(n_seeds, dim=1)
            .indices
        )
        seed_mask = torch.zeros(batch, n_genes, device=device).scatter_(1, seeds, 1.0)
        gene_masked = (seed_mask + seed_mask @ neighbours) > 0

        # The rest of the budget, gene by gene, among genes still visible.
        remaining = (n_whole_genes - gene_masked.sum(dim=1)).clamp(min=0)
        scores = torch.rand(batch, n_genes, device=device, generator=generator)
        scores = scores.masked_fill(gene_masked, -1.0)
        rank = scores.argsort(dim=1, descending=True).argsort(dim=1)
        return gene_masked | (rank < remaining.unsqueeze(1))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        hidden_omics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a mask for one batch of patients.

        Args:
            x: Per-gene values, shape ``(batch, n_genes, 3)`` in modality order.
            generator: Random generator on ``x``'s device, for a reproducible
                draw (a fixed monitoring mask); the global stream when None.
            hidden_omics: Omics index per patient, shape ``(batch,)``, to hide
                entirely (-1 hides none), in place of the random omics-level
                draw. Used to build two views of a patient that lack
                different omics.

        Returns:
            Tuple of the boolean mask, shape ``(batch, n_genes, 3)`` and True
            where a value is hidden, and the reconstruction targets, which are a
            copy of ``x`` taken before any masking is applied downstream.
        """
        batch, n_genes, _ = x.shape
        device = x.device

        targets = x.clone()
        mask = torch.zeros(batch, n_genes, 3, dtype=torch.bool, device=device)

        gene_masked = self._gene_mask(batch, n_genes, device, generator)
        mask |= gene_masked.unsqueeze(-1)

        # Single-modality masks are drawn only among genes still fully visible,
        # which is what keeps the two sets disjoint.
        eligible = (
            torch.rand(batch, n_genes, device=device, generator=generator)
            < self.mask_modality_frac
        ) & (~gene_masked)
        weights = cast(torch.Tensor, self.modality_weights)
        # Drawn on the input's device, so a seeded CPU generator can draw a
        # reproducible mask for a model that lives on the GPU.
        probabilities = (weights / weights.sum()).to(device)
        picked = torch.multinomial(
            probabilities, batch * n_genes, replacement=True, generator=generator
        ).view(batch, n_genes)
        one_hot = F.one_hot(picked, num_classes=3).bool()
        mask |= eligible.unsqueeze(-1) & one_hot

        if hidden_omics is not None:
            hidden_omics = hidden_omics.to(device)
            whole = F.one_hot(hidden_omics.clamp(min=0), num_classes=3).bool()
            mask |= (whole & (hidden_omics >= 0).unsqueeze(-1)).unsqueeze(1)
        elif self.omics_mask_prob > 0.0:
            hidden = (
                torch.rand(batch, device=device, generator=generator)
                < self.omics_mask_prob
            )
            omics = torch.randint(0, 3, (batch,), device=device, generator=generator)
            whole = F.one_hot(omics, num_classes=3).bool() & hidden.unsqueeze(-1)
            mask |= whole.unsqueeze(1)

        return mask, targets
