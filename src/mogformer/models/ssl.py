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
from mogformer.models.layers.gated_fusion import FUSION_TYPES, GatedFusion
from mogformer.models.layers.global_transformer import GlobalGraphTransformer
from mogformer.models.layers.masking import MaskedMultiModalMasker
from mogformer.models.layers.mini_transformer import MiniTransformer
from mogformer.models.layers.modality_lifting import ModalityLifting
from mogformer.models.layers.program_readout import PROGRAM_INPUTS, ProgramReadout


class MOGFormerSSL(nn.Module):
    """Masked multi-modal encoder over the gene graph, with a dual-head decoder.

    Sequence layout is fixed throughout: the tumor summary token sits at index 0
    and gene ``i`` at index ``i + 1``. The decoder and every probe depend on it.

    Attributes:
        d: Token width.
        embedding_width: Width of the patient embedding ``c``.
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
        global_offset: bool = True,
        isolated_token: bool = False,
        identity_modulation: bool = False,
        n_summary_tokens: int = 1,
        global_head: str = "film",
        fusion_residual: bool = False,
        neighbourhood_mask_share: float = 0.0,
        omics_mask_prob: float = 0.0,
        n_programs: int = 0,
        program_width: int = 32,
        program_layers: int = 2,
        program_input: str = "final",
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
            fusion_type: One of
                :data:`~mogformer.models.layers.gated_fusion.FUSION_TYPES`.
            mask_gene_frac: Fraction of genes hidden entirely per patient.
            mask_modality_frac: Probability of hiding one modality of a gene.
            mask_modality_weights: Relative weights over ``(rna, cnv, methy)``
                when choosing which single modality to hide.
            rna_only: Zero copy number and methylation before encoding.
            cnv_off: Zero copy number before encoding.
            global_offset: Let the decoder's global head add a patient-level
                offset shared by every gene and modality.
            isolated_token: Give genes without an interaction edge a learned
                positional code of their own.
            identity_modulation: Let gene identity scale and shift each value
                token, not only sit beside it.
            n_summary_tokens: Summary tokens ``K`` ahead of the genes; the
                patient embedding is their concatenation, ``K * d`` wide.
            global_head: Form of the decoder's global head, one of
                :data:`~mogformer.models.layers.decoder.GLOBAL_HEADS`.
            fusion_residual: Carry the unfused omics tokens past the gated
                fusion (gated fusion only).
            neighbourhood_mask_share: Share of whole-gene masks drawn as a gene
                and its interaction neighbours.
            omics_mask_prob: Probability, per patient, of hiding one whole
                omics during pretraining.
            n_programs: Gene programs read out by
                :class:`~mogformer.models.layers.program_readout.ProgramReadout`;
                0 keeps the summary-token embedding. With programs, the patient
                embedding is the program embeddings concatenated and the
                decoder's global head is scoped by program. The assignment is
                installed after construction, once it is fitted.
            program_width: Width of each program embedding.
            program_layers: Read-out layers of the program queries.
            program_input: Gene states the programs read, one of
                :data:`~mogformer.models.layers.program_readout.PROGRAM_INPUTS`.

        Raises:
            ValueError: If ``fusion_type`` is unknown, or if
                ``gene_id_embedding`` is False — the decoder addresses its
                predictions by gene identity, so without those embeddings it
                cannot say which gene a prediction is about. Also if
                programs are combined with a global head other than ``film``.
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
        if program_input not in PROGRAM_INPUTS:
            raise ValueError(
                f"unknown program_input {program_input!r}; "
                f"available: {list(PROGRAM_INPUTS)}"
            )
        if fusion_residual and fusion_type != "gated":
            raise ValueError("fusion_residual is defined for gated fusion only")

        self.d = d
        self.rna_only = rna_only
        self.cnv_off = cnv_off

        self.lifter = ModalityLifting(
            d=d,
            gene_id_embedding=gene_id_embedding,
            n_universe=n_universe,
            gene_ids=gene_ids,
            identity_modulation=identity_modulation,
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
                residual=fusion_residual,
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
            isolated_token=isolated_token,
            n_summary_tokens=n_summary_tokens,
        )
        self.n_summary_tokens = n_summary_tokens
        self.masker = MaskedMultiModalMasker(
            mask_gene_frac,
            mask_modality_frac,
            mask_modality_weights,
            neighbourhood_share=neighbourhood_mask_share,
            omics_mask_prob=omics_mask_prob,
        )
        self.n_programs = n_programs
        self.program_input = program_input
        self.program_readout: ProgramReadout | None = None
        if n_programs > 0:
            self.program_readout = ProgramReadout(
                d=d,
                n_genes=len(gene_ids) if gene_ids is not None else n_universe,
                n_programs=n_programs,
                width=program_width,
                num_layers=program_layers,
                dropout=dropout,
            )
        self.embedding_width = (
            n_programs * program_width if n_programs > 0 else n_summary_tokens * d
        )
        self.decoder = DualHeadDecoder(
            d=d,
            global_offset=global_offset,
            global_head=global_head,
            n_summary_tokens=n_summary_tokens,
            program_width=program_width if n_programs > 0 else None,
        )

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
            With ``mask`` False: ``c`` (the patient embedding: summary tokens
            concatenated, ``(batch, K * d)``, or with gene programs the program
            embeddings concatenated), ``H_final`` (full sequence, summary tokens
            first), ``H_genes`` (the genes' final states) and ``gate`` (fusion
            interpretability).
            With ``mask`` True: ``c``, ``gate``, ``xhat_g`` and ``xhat_l``
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

        k = self.n_summary_tokens
        if hidden_last.shape[1] != fused.shape[1] + k:
            raise RuntimeError(
                f"expected {k} summary token(s) ahead of {fused.shape[1]} genes, "
                f"got a sequence of {hidden_last.shape[1]}"
            )

        programs = None
        if self.program_readout is not None:
            programs = self.program_readout(
                fused if self.program_input == "fused" else hidden_last[:, k:, :]
            )
            summary = programs.reshape(programs.shape[0], -1)

        if not mask:
            return {
                "c": summary,
                "H_final": hidden_last,
                "H_genes": hidden_last[:, k:, :],
                "gate": gate,
            }

        # The decoder must address predictions with the encoder's own
        # embeddings, not copies of its own.
        gene_embedding = self.lifter.get_gene_embedding()
        modality_embedding = self.lifter.get_modality_embeddings()
        global_prediction, local_prediction = self.decoder(
            summary,
            hidden_last,
            gene_embedding,
            modality_embedding,
            programs=programs,
            assignment=(
                None
                if self.program_readout is None
                else self.program_readout.assignment
            ),
        )
        return {
            "xhat_g": global_prediction,
            "xhat_l": local_prediction,
            "mask_bool": mask_bool,
            "targets": targets,
            "c": summary,
            "gate": gate,
        }
