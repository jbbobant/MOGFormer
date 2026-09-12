"""Self-supervised encoder with a masked multi-modal reconstruction objective.

Pretraining exists because the cohort is small relative to the feature space.
Hiding measurements and asking the model to reconstruct them supplies a training
signal that needs no labels, and — critically for this project — it is the
objective that makes the interventional probes meaningful: a model trained to
predict a gene's expression from its neighbours and its own other channels has
learned a conditional it can then be interrogated about.

The encoder is trained outcome-blind. Survival never enters here, which is what
lets a later confirmatory analysis on the learned representation be treated as a
genuine test rather than a restatement of its own training signal.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mogformer.models.layers.decoder import DualHeadDecoder
from mogformer.models.layers.gated_fusion import GatedFusion
from mogformer.models.layers.global_transformer import GlobalGraphTransformer
from mogformer.models.layers.masking import MaskedMultiModalMasker
from mogformer.models.layers.mini_transformer import MiniTransformer
from mogformer.models.layers.modality_lifting import ModalityLifting

#: Intra-gene fusion layers selectable for pretraining.
FUSION_TYPES: tuple[str, ...] = ("gated", "attention")


class MOGFormerSSL(nn.Module):
    """Masked multi-modal encoder over the gene graph, with a dual-head decoder.

    Sequence layout is fixed throughout: the tumor summary token sits at index 0
    and gene ``i`` at index ``i + 1``. The decoder and every probe depend on it.

    Attributes:
        d: Token width.
        rna_only: Zero the non-expression modalities, for the single-omics rung
            of the ablation ladder.
        cnv_off: Zero copy number only.
    """

    def __init__(
        self,
        d: int = 128,
        pe_dim: int = 32,
        mini_heads: int = 4,
        global_heads: int = 8,
        global_layers: int = 3,
        dropout: float = 0.2,
        max_distance: int = 4,
        attention_bias_mode: str = "inside",
        use_grn: bool = True,
        gene_id_embedding: bool = True,
        n_universe: int = 0,
        gene_ids: Sequence[int] | None = None,
        lambda_gate: bool = False,
        fusion_type: str = "gated",
        mask_gene_frac: float = 0.15,
        mask_modality_frac: float = 0.15,
        mask_modality_weights: tuple[float, float, float] = (2.0, 1.0, 1.0),
        rna_only: bool = False,
        cnv_off: bool = False,
    ) -> None:
        """Assemble encoder, masker and decoder.

        Args:
            d: Token width.
            pe_dim: Width of the graph positional encoding.
            mini_heads: Attention heads for attention-based fusion.
            global_heads: Attention heads in the inter-gene stage.
            global_layers: Number of inter-gene transformer blocks.
            dropout: Dropout throughout.
            max_distance: Largest hop count represented exactly by the bias.
            attention_bias_mode: One of
                :data:`~mogformer.models.layers.structural_attention.ATTENTION_BIAS_MODES`.
            use_grn: Enable the signed regulatory attention bias.
            gene_id_embedding: Must be True — see Raises.
            n_universe: Gene universe size for the identity table.
            gene_ids: Universe positions of the selected genes.
            lambda_gate: Per-head learnable scaling of the distance bias.
            fusion_type: One of :data:`FUSION_TYPES`.
            mask_gene_frac: Fraction of genes hidden entirely per patient.
            mask_modality_frac: Probability of hiding one modality of a gene.
            mask_modality_weights: Relative weights over ``(rna, cnv, methy)``
                when choosing which single modality to hide.
            rna_only: Zero copy number and methylation before encoding.
            cnv_off: Zero copy number before encoding.

        Raises:
            ValueError: If ``fusion_type`` is unknown, or if
                ``gene_id_embedding`` is False — the decoder addresses its
                predictions by gene identity, so without those embeddings it
                cannot say which gene a prediction is about.
        """
        super().__init__()
        if fusion_type not in FUSION_TYPES:
            raise ValueError(
                f"unknown fusion type {fusion_type!r}; available: {list(FUSION_TYPES)}"
            )
        if not gene_id_embedding:
            raise ValueError(
                "gene_id_embedding is required: the decoder addresses its "
                "reconstructions by per-gene identity"
            )

        self.d = d
        self.rna_only = rna_only
        self.cnv_off = cnv_off

        self.lifter = ModalityLifting(
            d=d,
            gene_id_embedding=gene_id_embedding,
            n_universe=n_universe,
            gene_ids=gene_ids,
        )

        # The masker owns all masking during pretraining, so the fusion layer's
        # own modality dropout is switched off — two independent maskers would
        # make a reconstruction target unattributable.
        if fusion_type == "gated":
            self.mini_transformer: GatedFusion | MiniTransformer = GatedFusion(
                d=d,
                dropout=dropout,
                rna_dropout_prob=0.0,
                cnv_dropout_prob=0.0,
                meth_dropout_prob=0.0,
            )
        else:
            self.mini_transformer = MiniTransformer(
                d=d,
                num_heads=mini_heads,
                dropout=dropout,
                rna_dropout_prob=0.0,
                cnv_dropout_prob=0.0,
                meth_dropout_prob=0.0,
            )

        self.global_transformer = GlobalGraphTransformer(
            d=d,
            pe_dim=pe_dim,
            num_heads=global_heads,
            num_layers=global_layers,
            max_distance=max_distance,
            attention_bias_mode=attention_bias_mode,
            dropout=dropout,
            lambda_gate=lambda_gate,
            use_grn=use_grn,
        )
        self.masker = MaskedMultiModalMasker(
            mask_gene_frac, mask_modality_frac, mask_modality_weights
        )
        self.decoder = DualHeadDecoder(d=d)

    def forward(
        self,
        rna: torch.Tensor,
        cnv: torch.Tensor,
        methy: torch.Tensor,
        graph_pe: torch.Tensor,
        spd_matrix: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        mask: bool = True,
        mask_bool: torch.Tensor | None = None,
        structural_bias: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch, and reconstruct masked values when masking is on.

        Args:
            rna: Expression values, shape ``(batch, n_genes)``.
            cnv: Copy-number values, same shape.
            methy: Methylation values, same shape.
            graph_pe: Positional encodings, shape ``(n_genes, pe_dim)``.
            spd_matrix: Integer gene distances, shape ``(n_genes, n_genes)``.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.
            mask: Run the reconstruction objective. False returns embeddings
                only, which is the path used for downstream analysis.
            mask_bool: Supply a deterministic mask, shape
                ``(batch, n_genes, 3)``, instead of drawing one. The
                interventional probes rely on this to hide exactly one value.
            structural_bias: When False, attention ignores the graph.

        Returns:
            With ``mask`` False: ``c`` (summary token), ``H_final`` (full
            sequence) and ``gate`` (fusion interpretability).
            With ``mask`` True: additionally ``xhat_g`` and ``xhat_l``
            (global-head and local-head reconstructions, shape
            ``(batch, n_genes, 3)``), ``mask_bool`` and ``targets``.

        Raises:
            RuntimeError: If the encoder does not return exactly one summary
                token ahead of the genes.
        """
        if self.rna_only:
            cnv = torch.zeros_like(cnv)
            methy = torch.zeros_like(methy)
        if self.cnv_off:
            cnv = torch.zeros_like(cnv)

        observed = torch.stack([rna, cnv, methy], dim=-1)
        targets = observed

        if mask:
            if mask_bool is None:
                mask_bool, targets = self.masker(observed)
            # A zeroed modality carries no signal, so masking it would create a
            # reconstruction target that is trivially predictable.
            if self.rna_only:
                mask_bool = mask_bool.clone()
                mask_bool[..., 1:] = False
            if self.cnv_off:
                mask_bool = mask_bool.clone()
                mask_bool[..., 1] = False

        z_rna, z_cnv, z_methy = self.lifter(
            rna, cnv, methy, mask_bool=mask_bool if mask else None
        )
        fused, gate = self.mini_transformer(z_rna, z_cnv, z_methy, eval_mask=None)
        summary, hidden_last = self.global_transformer(
            fused, graph_pe, spd_matrix, grn_matrix, structural_bias=structural_bias
        )

        if hidden_last.shape[1] != fused.shape[1] + 1:
            raise RuntimeError(
                f"expected one summary token ahead of {fused.shape[1]} genes, "
                f"got a sequence of {hidden_last.shape[1]}"
            )

        if not mask:
            return {"c": summary, "H_final": hidden_last, "gate": gate}

        # The decoder must address predictions with the encoder's own
        # embeddings, not copies of its own.
        gene_embedding = self.lifter.get_gene_embedding()
        modality_embedding = self.lifter.get_modality_embeddings()
        global_prediction, local_prediction = self.decoder(
            summary, hidden_last, gene_embedding, modality_embedding
        )
        return {
            "xhat_g": global_prediction,
            "xhat_l": local_prediction,
            "mask_bool": mask_bool,
            "targets": targets,
            "c": summary,
            "gate": gate,
        }
