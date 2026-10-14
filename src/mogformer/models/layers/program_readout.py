"""Program-scoped read-out of the patient from the genes' final states.

Each gene program (see :mod:`mogformer.data.gene_programs`) gets a learned
query that attends only to the program's member genes. Genes never attend back
to these queries, so the read-out cannot mix one program's content into
another's genes. Each program's state is projected to a narrow embedding, and
the patient embedding is their concatenation, one block per program. Paired
with the decoder's scoped global head, a program's block is the only route by
which the patient state reaches that program's reconstruction.

What the queries read matters. The genes' final states have already been mixed
across the whole axis by the inter-gene transformer, so every member gene
carries the same few patient-wide directions, and every block ends up holding
the same information. Reading the fused per-gene tokens instead, before any
mixing between genes, gives each block only its own genes' measurements.
"""

from __future__ import annotations

import torch
from torch import nn

#: Gene states a program read-out can read: ``"final"`` (after the inter-gene
#: transformer) or ``"fused"`` (each gene's own fused token, before it).
PROGRAM_INPUTS: tuple[str, ...] = ("final", "fused")


class ProgramReadout(nn.Module):
    """Read one embedding per gene program with membership-masked attention.

    Attributes:
        n_programs: Number of programs ``K``.
        width: Width of each program's embedding.
        assignment: Program index per gene, shape ``(n_genes,)``; a persistent
            buffer, so it is saved with the weights.
    """

    assignment: torch.Tensor

    def __init__(
        self,
        d: int,
        n_genes: int,
        n_programs: int,
        width: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Build the program queries, the attention stack and the projection.

        Args:
            d: Token width of the gene states.
            n_genes: Genes on the axis.
            n_programs: Number of programs ``K``.
            width: Width of each program's embedding.
            num_heads: Attention heads per layer.
            num_layers: Read-out layers; each lets a program's query refine
                what it reads from its genes.
            dropout: Dropout in attention and feed-forward sublayers.
        """
        super().__init__()
        self.n_programs = n_programs
        self.width = width
        self.queries = nn.Parameter(torch.zeros(1, n_programs, d))
        nn.init.normal_(self.queries, std=0.02)
        self.register_buffer(
            "assignment", torch.arange(n_genes, dtype=torch.long) % n_programs
        )
        self.query_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(num_layers))
        self.gene_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(num_layers))
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(num_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d, d),
            )
            for _ in range(num_layers)
        )
        self.projection = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width))

    @torch.no_grad()
    def set_assignment(self, assignment: torch.Tensor) -> None:
        """Install the program of every gene.

        Args:
            assignment: Program index per gene, shape ``(n_genes,)``.

        Raises:
            ValueError: If the shape is wrong, an index is out of range, or a
                program has no gene.
        """
        assignment = torch.as_tensor(assignment, dtype=torch.long)
        if assignment.shape != self.assignment.shape:
            raise ValueError(
                f"expected {tuple(self.assignment.shape)} assignments, "
                f"got {tuple(assignment.shape)}"
            )
        counts = torch.bincount(assignment, minlength=self.n_programs)
        if len(counts) != self.n_programs or (counts == 0).any():
            raise ValueError(
                f"every program in [0, {self.n_programs}) needs a gene; "
                f"counts {counts.tolist()}"
            )
        self.assignment.copy_(assignment.to(self.assignment.device))

    def membership_mask(self) -> torch.Tensor:
        """Return the attention mask, True where program ``k`` may not read gene ``g``.

        Returns:
            Boolean mask of shape ``(K, n_genes)``.
        """
        programs = torch.arange(self.n_programs, device=self.assignment.device)
        return self.assignment[None, :] != programs[:, None]

    @torch.no_grad()
    def attention_weights(self, genes: torch.Tensor) -> torch.Tensor:
        """Recompute how much each program's query attends to each gene.

        The forward pass never stores attention; this replays it for
        interpretation, without dropout, and returns the last layer's
        weights averaged over heads.

        Args:
            genes: Gene states the read-out reads, shape ``(batch, n_genes, d)``.

        Returns:
            Weights of shape ``(batch, K, n_genes)``; zero outside each
            program, summing to one within it.
        """
        mask = self.membership_mask()
        state = self.queries.expand(genes.shape[0], -1, -1)
        weights = torch.zeros(0)
        for q_norm, g_norm, attention, f_norm, ffn in zip(
            self.query_norms,
            self.gene_norms,
            self.attention,
            self.ffn_norms,
            self.ffn,
            strict=True,
        ):
            keys = g_norm(genes)
            read, weights = attention(
                q_norm(state),
                keys,
                keys,
                attn_mask=mask,
                need_weights=True,
                average_attn_weights=True,
            )
            state = state + read
            state = state + ffn(f_norm(state))
        return weights

    def forward(self, genes: torch.Tensor) -> torch.Tensor:
        """Read the programs from the genes' final states.

        Args:
            genes: Gene states, shape ``(batch, n_genes, d)``.

        Returns:
            Program embeddings, shape ``(batch, K, width)``.
        """
        mask = self.membership_mask()
        state = self.queries.expand(genes.shape[0], -1, -1)
        for q_norm, g_norm, attention, f_norm, ffn in zip(
            self.query_norms,
            self.gene_norms,
            self.attention,
            self.ffn_norms,
            self.ffn,
            strict=True,
        ):
            keys = g_norm(genes)
            read, _ = attention(
                q_norm(state), keys, keys, attn_mask=mask, need_weights=False
            )
            state = state + read
            state = state + ffn(f_norm(state))
        return self.projection(state)
