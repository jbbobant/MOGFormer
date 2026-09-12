"""Supervised multi-omics graph transformer for subtype classification.

Assembles the full supervised stack: lift three scalars per gene into tokens,
fuse them within each gene, contextualise genes against one another under the
structural bias, and read the tumor summary token through a classification head.

Interpretability is native rather than post-hoc. The intra-gene attention says
which modality drove a gene's state and the summary token's attention says which
genes drove the prediction, so no external attribution method is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mogformer.models.layers.global_transformer import GlobalGraphTransformer
from mogformer.models.layers.mini_transformer import MiniTransformer
from mogformer.models.layers.modality_lifting import ModalityLifting


class MultiOmicsGraphClassifier(nn.Module):
    """Predict a tumor subtype from three omics modalities and a gene graph.

    Attributes:
        d: Token width.
        num_classes: Number of subtypes predicted.
        lifter: Scalar-to-token stage.
        mini_transformer: Intra-gene fusion stage.
        global_transformer: Inter-gene contextualisation stage.
        classifier_head: Bottleneck head over the tumor summary token.
    """

    def __init__(
        self,
        num_classes: int = 5,
        d: int = 64,
        pe_dim: int = 16,
        mini_heads: int = 4,
        global_heads: int = 8,
        global_layers: int = 4,
        dropout: float = 0.1,
        rna_dropout_prob: float = 0.15,
        cnv_dropout_prob: float = 0.15,
        meth_dropout_prob: float = 0.15,
        max_distance: int = 5,
        attention_bias_mode: str = "dual",
        gene_id_embedding: bool = False,
        n_universe: int = 0,
        gene_ids: Sequence[int] | None = None,
        pretrained_gene_emb: torch.Tensor | None = None,
        pretrained_emb_adapter_rank: int = 0,
        numerical_tokenizer: str = "mlp",
        plr_n_frequencies: int = 16,
        plr_sigma: float = 1.0,
        lambda_gate: bool = False,
        unimodal_dropout_fill: str = "zero",
        use_grn: bool = False,
    ) -> None:
        """Assemble the four stages.

        Args:
            num_classes: Number of subtypes.
            d: Token width.
            pe_dim: Width of the graph positional encoding.
            mini_heads: Attention heads in the intra-gene stage.
            global_heads: Attention heads in the inter-gene stage.
            global_layers: Number of inter-gene transformer blocks.
            dropout: Dropout throughout.
            rna_dropout_prob: Probability of hiding expression per gene.
            cnv_dropout_prob: Probability of hiding copy number per gene.
            meth_dropout_prob: Probability of hiding methylation per gene.
            max_distance: Largest hop count represented exactly by the bias.
            attention_bias_mode: One of
                :data:`~mogformer.models.layers.structural_attention.ATTENTION_BIAS_MODES`.
            gene_id_embedding: Add per-gene identity embeddings.
            n_universe: Gene universe size, for the learnable identity table.
            gene_ids: Universe positions of the selected genes.
            pretrained_gene_emb: Pretrained gene embeddings, or None.
            pretrained_emb_adapter_rank: Adapter rank for the pretrained path.
            numerical_tokenizer: One of
                :data:`~mogformer.models.layers.modality_lifting.NUMERICAL_TOKENIZERS`.
            plr_n_frequencies: Basis size for the periodic tokenizer.
            plr_sigma: Frequency scale for the periodic tokenizer.
            lambda_gate: Per-head learnable scaling of the distance bias.
            unimodal_dropout_fill: One of
                :data:`~mogformer.models.layers.modality_dropout.DROPOUT_FILLS`.
            use_grn: Enable the signed regulatory attention bias.
        """
        super().__init__()
        self.d = d
        self.num_classes = num_classes

        self.lifter = ModalityLifting(
            d=d,
            gene_id_embedding=gene_id_embedding,
            n_universe=n_universe,
            gene_ids=gene_ids,
            numerical_tokenizer=numerical_tokenizer,
            plr_n_frequencies=plr_n_frequencies,
            plr_sigma=plr_sigma,
            pretrained_gene_emb=pretrained_gene_emb,
            pretrained_emb_adapter_rank=pretrained_emb_adapter_rank,
        )

        self.mini_transformer = MiniTransformer(
            d=d,
            num_heads=mini_heads,
            dropout=dropout,
            rna_dropout_prob=rna_dropout_prob,
            cnv_dropout_prob=cnv_dropout_prob,
            meth_dropout_prob=meth_dropout_prob,
            unimodal_dropout_fill=unimodal_dropout_fill,
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

        self.classifier_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.BatchNorm1d(d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, num_classes),
        )

    def forward(
        self,
        rna: torch.Tensor,
        cnv: torch.Tensor,
        methy: torch.Tensor,
        graph_pe: torch.Tensor,
        spd_matrix: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        eval_mask: Sequence[str] | None = None,
        structural_bias: bool = True,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict subtype logits for a batch of patients.

        Args:
            rna: Expression values, shape ``(batch, n_genes)``.
            cnv: Copy-number values, same shape.
            methy: Methylation values, same shape.
            graph_pe: Positional encodings, shape ``(n_genes, pe_dim)``.
            spd_matrix: Integer gene distances, shape ``(n_genes, n_genes)``.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.
            eval_mask: Modalities to hide deterministically at evaluation time,
                for the modality-contribution probes.
            structural_bias: When False, the inter-gene stage ignores the graph.
            return_attention: Also return the interpretability tensors.

        Returns:
            Mapping with ``logits`` of shape ``(batch, num_classes)`` and, when
            ``return_attention`` is set, ``tumor_state``, ``intra_attn_weights``
            and ``hidden_last``.
        """
        z_rna, z_cnv, z_methy = self.lifter(rna, cnv, methy)
        fused, intra_attn_weights = self.mini_transformer(
            z_rna, z_cnv, z_methy, eval_mask=eval_mask
        )
        tumor_state, hidden_last = self.global_transformer(
            fused,
            graph_pe,
            spd_matrix,
            grn_matrix,
            structural_bias=structural_bias,
        )

        output = {"logits": self.classifier_head(tumor_state)}
        if return_attention:
            output["tumor_state"] = tumor_state
            output["intra_attn_weights"] = intra_attn_weights
            output["hidden_last"] = hidden_last
        return output
