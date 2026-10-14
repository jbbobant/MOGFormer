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

Two forms of the global head exist. The ``film`` head scales a fixed address
by a function of the summary: a low-rank, multiplicative read-out, which lets
a single dominant summary direction explain most of the reconstruction. The
``cross_attention`` head lets every address attend over the summary tokens and
passes the result through a non-linear block, so different genes can read
different summary tokens and different directions within them.

With gene programs (``program_width`` set), the ``film`` head is scoped: gene
``i`` is modulated by the embedding of its own program only, so a program's
embedding is the one route by which the patient state reaches its genes.
"""

from __future__ import annotations

import torch
from torch import nn

#: Forms of the global head accepted by :class:`DualHeadDecoder`.
GLOBAL_HEADS: tuple[str, ...] = ("film", "cross_attention")


class DualHeadDecoder(nn.Module):
    """Reconstruct per-gene, per-modality values from a shared address grid.

    Attributes:
        p_dec: Learnable decoder-role token, distinguishing the reconstruction
            address from the encoder's own embeddings.
        global_offset: Whether the global head adds the patient-level shift.
        global_head: Form of the global head, one of :data:`GLOBAL_HEADS`.
        n_summary_tokens: Summary tokens ahead of the genes in the sequence.
        program_width: Width of each program embedding when the global head is
            scoped by gene programs, else None.
    """

    def __init__(
        self,
        d: int = 128,
        global_offset: bool = True,
        global_head: str = "film",
        n_summary_tokens: int = 1,
        cross_attention_heads: int = 4,
        program_width: int | None = None,
    ) -> None:
        """Build the two heads and the modulation projections.

        Args:
            d: Token width; must match the encoder's.
            global_offset: Let the global head shift every address by the same
                patient-level vector ``beta(summary)``. After the shared
                read-out that becomes a per-patient constant added to every
                gene of every modality, which carries no biology and lets the
                summary spend capacity on a global level rather than on
                gene-specific structure. False keeps only the scale. Only read
                by the ``film`` head.
            global_head: One of :data:`GLOBAL_HEADS`.
            n_summary_tokens: Summary tokens the encoder prepends; the ``film``
                head reads them concatenated, the ``cross_attention`` head
                attends over them.
            cross_attention_heads: Heads of the ``cross_attention`` head.
            program_width: Scope the ``film`` head by gene programs, each read
                from a program embedding of this width.

        Raises:
            ValueError: If ``global_head`` is unknown, or programs are combined
                with a head other than ``film``.
        """
        if global_head not in GLOBAL_HEADS:
            raise ValueError(
                f"unknown global head {global_head!r}; available: {list(GLOBAL_HEADS)}"
            )
        if program_width is not None and global_head != "film":
            raise ValueError("gene programs scope the film head only")
        super().__init__()
        self.p_dec = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.p_dec, std=0.02)

        self.global_head = global_head
        self.n_summary_tokens = n_summary_tokens
        self.global_offset = global_offset and global_head == "film"
        self.program_width = program_width
        if global_head == "film":
            summary_width = program_width or n_summary_tokens * d
            self.mlp_gamma = self._build_block(d, summary_width)
            if self.global_offset:
                self.mlp_beta = self._build_block(d, summary_width)
        else:
            self.query_norm = nn.LayerNorm(d)
            self.summary_norm = nn.LayerNorm(d)
            self.cross_attention = nn.MultiheadAttention(
                d, cross_attention_heads, batch_first=True
            )
        self.ffn_global = self._build_block(d)
        self.lin_global = nn.Linear(d, 1)
        self.ffn_local = self._build_block(d)
        self.lin_local = nn.Linear(d, 1)

    @staticmethod
    def _build_block(d: int, d_in: int | None = None) -> nn.Sequential:
        """Return the two-layer block used throughout the decoder."""
        return nn.Sequential(
            nn.Linear(d_in or d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(
        self,
        summary: torch.Tensor,
        hidden_last: torch.Tensor,
        gene_embedding: torch.Tensor,
        modality_embedding: torch.Tensor,
        programs: torch.Tensor | None = None,
        assignment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict every gene-by-modality value from both heads.

        Args:
            summary: Patient summary tokens concatenated, shape
                ``(batch, K * d)``.
            hidden_last: Final encoder states, the ``K`` summary tokens first,
                shape ``(batch, K + n_genes, d)``.
            gene_embedding: The encoder's gene identity embeddings, shape
                ``(n_genes, d)``.
            modality_embedding: The encoder's modality embeddings, shape
                ``(3, d)`` in modality order.
            programs: Program embeddings, shape ``(batch, K, program_width)``,
                when the head is scoped by programs; ``summary`` is then unused.
            assignment: Program index per gene, shape ``(n_genes,)``.

        Returns:
            Tuple of global-head and local-head predictions, each of shape
            ``(batch, n_genes, 3)``.
        """
        k = self.n_summary_tokens
        gene_rows = hidden_last[:, k:, :]

        # Address grid: one query per (gene, modality) pair.
        address = (
            gene_embedding[:, None, :]
            + modality_embedding[None, :, :]
            + self.p_dec[None, None, :]
        )

        if self.global_head == "film" and programs is not None:
            if assignment is None:
                raise ValueError("a program-scoped head needs the assignment")
            # Each gene is scaled (and shifted) by its own program's embedding.
            gamma = self.mlp_gamma(programs)[:, assignment, :]
            modulated = self.ffn_global(address)[None] * gamma[:, :, None, :]
            if self.global_offset:
                beta = self.mlp_beta(programs)[:, assignment, :]
                modulated = modulated + beta[:, :, None, :]
            global_prediction = self.lin_global(modulated).squeeze(-1)
        elif self.global_head == "film":
            # The patient enters only as a feature-wise scale (and shift).
            gamma = self.mlp_gamma(summary)
            modulated = self.ffn_global(address)[None] * gamma[:, None, None, :]
            if self.global_offset:
                modulated = modulated + self.mlp_beta(summary)[:, None, None, :]
            global_prediction = self.lin_global(modulated).squeeze(-1)
        else:
            # Every (gene, omics) address attends over the summary tokens.
            batch = hidden_last.shape[0]
            n_genes = address.shape[0]
            queries = address.reshape(1, n_genes * 3, -1).expand(batch, -1, -1)
            tokens = self.summary_norm(hidden_last[:, :k, :])
            attended, _ = self.cross_attention(
                self.query_norm(queries), tokens, tokens, need_weights=False
            )
            read = self.ffn_global(queries + attended)
            global_prediction = self.lin_global(read).view(batch, n_genes, 3)

        # Local head: the address is offset by the gene's own final state.
        local_input = address[None] + gene_rows[:, :, None, :]
        local_prediction = self.lin_local(self.ffn_local(local_input)).squeeze(-1)

        return global_prediction, local_prediction
