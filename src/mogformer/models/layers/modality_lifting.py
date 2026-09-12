"""Lift scalar per-gene measurements into modality-aware tokens.

Each gene carries three scalars — expression, copy number, methylation — and the
transformer needs tokens. This module turns each scalar into a ``d``-dimensional
token, tags it with a learnable modality embedding so the model knows which
measurement it is reading, and optionally adds a per-gene identity embedding so
two genes with the same scaled value remain distinguishable.

Two numerical tokenizers are available. The default two-layer perceptron is the
original; the periodic tokenizer of Gorishniy et al. (NeurIPS 2022) is the
alternative, motivated by biology acting on thresholds rather than linearly.

Masking for self-supervised pretraining is applied here, at the value-token
level: a masked token becomes ``mask_k + modality_k + gene_i``, so the model
still knows *which* measurement of *which* gene it is being asked to
reconstruct, and only the value itself is hidden.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

#: Numerical tokenizers accepted by :class:`ModalityLifting`.
NUMERICAL_TOKENIZERS: tuple[str, ...] = ("mlp", "plr")


class GeneEmbeddingAdapter(nn.Module):
    """Project frozen pretrained gene embeddings, optionally adapting them first.

    The pretrained matrix stays frozen — it is registered as a buffer, so it
    moves with the module across devices but receives no gradient. With
    ``adapter_rank > 0`` a low-rank residual correction is learned on top,
    initialised to zero so training starts exactly at the frozen embedding.

    Attributes:
        adapter_rank: Rank of the residual correction; 0 means projection only.
    """

    def __init__(
        self, pretrained: torch.Tensor, d_out: int, adapter_rank: int = 0
    ) -> None:
        """Register the frozen embeddings and build the projection.

        Args:
            pretrained: Embeddings for the selected genes, shape
                ``(n_selected, d_embedding)``.
            d_out: Token width to project onto.
            adapter_rank: Rank of the low-rank residual correction. 0 disables it.
        """
        super().__init__()
        d_embedding = pretrained.shape[1]
        self.register_buffer("base", pretrained.float())
        self.adapter_rank = adapter_rank

        if adapter_rank > 0:
            self.w_down = nn.Linear(d_embedding, adapter_rank, bias=False)
            self.w_up = nn.Linear(adapter_rank, d_embedding, bias=False)
            nn.init.xavier_normal_(self.w_down.weight)
            # Zero init makes the residual an identity at initialisation.
            nn.init.zeros_(self.w_up.weight)

        self.proj = nn.Linear(d_embedding, d_out, bias=False)
        nn.init.xavier_normal_(self.proj.weight)

    def forward(self) -> torch.Tensor:
        """Return the projected gene embeddings.

        Takes no input: the embeddings are a property of the gene set, not of
        the patient.

        Returns:
            Tensor of shape ``(1, n_selected, d_out)``, broadcastable over batch.
        """
        embeddings = self.base
        if self.adapter_rank > 0:
            embeddings = embeddings + self.w_up(
                torch.nn.functional.gelu(self.w_down(embeddings))
            )
        return self.proj(embeddings).unsqueeze(0)


class PeriodicLinearTokenizer(nn.Module):
    """Tokenize a scalar through a learnable periodic basis.

    Implements the periodic-linear-ReLU embedding of Gorishniy et al.,
    *On Embeddings for Numerical Features in Tabular Deep Learning*
    (NeurIPS 2022). A scalar is expanded onto sine and cosine bases at learnable
    frequencies, then passed through two linear layers::

        periodic(x) = [sin(2 pi v_k x), cos(2 pi v_k x)] for k in 1..n_frequencies
        token(x)    = linear_2(relu(linear_1(periodic(x))))

    One instance per modality; frequencies are shared across genes because gene
    identity is carried separately, leaving the tokenizer to model only the
    modality's value distribution.

    Attributes:
        n_frequencies: Number of learnable frequencies.
        frequencies: The learnable frequencies themselves.
    """

    def __init__(self, d: int, n_frequencies: int = 16, sigma: float = 1.0) -> None:
        """Build the periodic basis and the two projections.

        Args:
            d: Output token width.
            n_frequencies: Size of the periodic basis.
            sigma: Standard deviation of the frequency initialisation. This is
                the sensitive hyperparameter — it sets the spectrum of the
                basis. For standardised features ``1.0`` is a reasonable start;
                tune over ``{0.1, 0.5, 1.0, 3.0}`` in the inner loop.
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * sigma)
        self.linear1 = nn.Linear(2 * n_frequencies, d)
        self.linear2 = nn.Linear(d, d)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a batch of per-gene scalars.

        Args:
            x: Scalars, shape ``(batch, n_genes)``.

        Returns:
            Tokens of shape ``(batch, n_genes, d)``.
        """
        projected = x.unsqueeze(-1) * self.frequencies * 2.0 * math.pi
        periodic = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        return self.linear2(torch.relu(self.linear1(periodic)))


class ModalityLifting(nn.Module):
    """Turn three per-gene scalars into three modality-tagged token streams.

    Attributes:
        d: Token width.
        gene_id_embedding: Whether a per-gene identity embedding is added.
        numerical_tokenizer: Active tokenizer, one of
            :data:`NUMERICAL_TOKENIZERS`.
    """

    def __init__(
        self,
        d: int = 64,
        gene_id_embedding: bool = False,
        n_universe: int = 0,
        gene_ids: Sequence[int] | None = None,
        gene_id_std: float = 0.02,
        numerical_tokenizer: str = "mlp",
        plr_n_frequencies: int = 16,
        plr_sigma: float = 1.0,
        mask_token_std: float = 0.10,
        pretrained_gene_emb: torch.Tensor | None = None,
        pretrained_emb_adapter_rank: int = 0,
    ) -> None:
        """Build the tokenizers, embeddings and mask tokens.

        Args:
            d: Token width.
            gene_id_embedding: Add a per-gene identity embedding to all three
                modality tokens, so genes with equal scaled values stay
                distinguishable.
            n_universe: Size of the gene universe, for the learnable identity
                embedding table. Required when ``gene_id_embedding`` is set and
                no pretrained matrix is supplied.
            gene_ids: Universe positions of the selected genes. Required under
                the same condition as ``n_universe``.
            gene_id_std: Initialisation scale of the learnable identity table.
            numerical_tokenizer: One of :data:`NUMERICAL_TOKENIZERS`.
            plr_n_frequencies: Basis size for the periodic tokenizer.
            plr_sigma: Frequency initialisation scale for the periodic tokenizer.
            mask_token_std: Initialisation scale of the pretraining mask tokens.
            pretrained_gene_emb: Pretrained embeddings for the selected genes,
                shape ``(n_selected, d_embedding)``. Takes precedence over the
                learnable identity table.
            pretrained_emb_adapter_rank: Adapter rank for the pretrained path.

        Raises:
            ValueError: If ``numerical_tokenizer`` is unknown, or if identity
                embeddings are requested without either a pretrained matrix or
                both ``n_universe`` and ``gene_ids``.
        """
        super().__init__()
        if numerical_tokenizer not in NUMERICAL_TOKENIZERS:
            raise ValueError(
                f"unknown numerical tokenizer {numerical_tokenizer!r}; "
                f"available: {list(NUMERICAL_TOKENIZERS)}"
            )

        self.d = d
        self.gene_id_embedding = gene_id_embedding
        self.numerical_tokenizer = numerical_tokenizer

        self.tok_rna: nn.Module
        self.tok_cnv: nn.Module
        self.tok_methy: nn.Module
        if numerical_tokenizer == "plr":
            self.tok_rna = PeriodicLinearTokenizer(d, plr_n_frequencies, plr_sigma)
            self.tok_cnv = PeriodicLinearTokenizer(d, plr_n_frequencies, plr_sigma)
            self.tok_methy = PeriodicLinearTokenizer(d, plr_n_frequencies, plr_sigma)
        else:
            self.tok_rna = self._build_mlp_tokenizer(d)
            self.tok_cnv = self._build_mlp_tokenizer(d)
            self.tok_methy = self._build_mlp_tokenizer(d)

        self.emb_rna = nn.Parameter(torch.randn(1, 1, d))
        self.emb_cnv = nn.Parameter(torch.randn(1, 1, d))
        self.emb_methy = nn.Parameter(torch.randn(1, 1, d))

        # Pretraining mask tokens, in modality order.
        self.mask_rna = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_cnv = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_methy = nn.Parameter(torch.zeros(1, 1, d))
        self._mask_token_std = mask_token_std

        if gene_id_embedding:
            if pretrained_gene_emb is not None:
                self.gene_adapter = GeneEmbeddingAdapter(
                    pretrained=pretrained_gene_emb,
                    d_out=d,
                    adapter_rank=pretrained_emb_adapter_rank,
                )
            else:
                if not n_universe or gene_ids is None:
                    raise ValueError(
                        "gene_id_embedding requires either pretrained_gene_emb, "
                        "or both n_universe > 0 and gene_ids"
                    )
                self.gene_emb = nn.Embedding(n_universe, d)
                self.register_buffer(
                    "gene_ids", torch.as_tensor(gene_ids, dtype=torch.long)
                )
                self._gene_id_std = gene_id_std

        self._init_weights()

    @staticmethod
    def _build_mlp_tokenizer(d: int) -> nn.Sequential:
        """Return the default two-layer scalar tokenizer."""
        return nn.Sequential(
            nn.Linear(1, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )

    def _init_weights(self) -> None:
        """Initialise tokenizer weights, modality embeddings and mask tokens."""
        if self.numerical_tokenizer == "mlp":
            for tokenizer in (self.tok_rna, self.tok_cnv, self.tok_methy):
                sequential = cast(nn.Sequential, tokenizer)
                for layer in (sequential[0], sequential[3]):
                    linear = cast(nn.Linear, layer)
                    nn.init.xavier_normal_(linear.weight)
                    nn.init.zeros_(linear.bias)
        # Periodic tokenizers initialise themselves.

        for embedding in (self.emb_rna, self.emb_cnv, self.emb_methy):
            nn.init.normal_(embedding, mean=0.0, std=0.02)
        for token in (self.mask_rna, self.mask_cnv, self.mask_methy):
            nn.init.normal_(token, mean=0.0, std=self._mask_token_std)

        if self.gene_id_embedding and hasattr(self, "gene_emb"):
            nn.init.normal_(self.gene_emb.weight, mean=0.0, std=self._gene_id_std)

    def get_gene_embedding(self) -> torch.Tensor | None:
        """Return the per-gene identity embeddings currently in use.

        The reconstruction decoder addresses its predictions by gene and
        modality, and must use the *same* embeddings as the encoder for that
        address to mean anything — hence this accessor rather than a second
        table on the decoder.

        Returns:
            Tensor of shape ``(n_selected, d)``, or None when identity
            embeddings are disabled.
        """
        if hasattr(self, "gene_adapter"):
            return self.gene_adapter().squeeze(0)
        if self.gene_id_embedding:
            return self.gene_emb(self.gene_ids)
        return None

    def get_modality_embeddings(self) -> torch.Tensor:
        """Return the three modality embeddings, shared with the decoder address.

        Returns:
            Tensor of shape ``(3, d)`` in modality order.
        """
        return torch.cat([self.emb_rna, self.emb_cnv, self.emb_methy], dim=1).squeeze(0)

    def forward(
        self,
        rna: torch.Tensor,
        cnv: torch.Tensor,
        methy: torch.Tensor,
        mask_bool: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lift the three scalar streams into token streams.

        Args:
            rna: Expression values, shape ``(batch, n_genes)``.
            cnv: Copy-number values, shape ``(batch, n_genes)``.
            methy: Methylation values, shape ``(batch, n_genes)``.
            mask_bool: Pretraining mask, shape ``(batch, n_genes, 3)`` in
                modality order, True where the value is hidden. Masking replaces
                the value token only, so modality and gene identity survive.

        Returns:
            Three tensors of shape ``(batch, n_genes, d)``, in modality order.
        """
        if self.numerical_tokenizer == "plr":
            value_rna = self.tok_rna(rna)
            value_cnv = self.tok_cnv(cnv)
            value_methy = self.tok_methy(methy)
        else:
            value_rna = self.tok_rna(rna.unsqueeze(-1))
            value_cnv = self.tok_cnv(cnv.unsqueeze(-1))
            value_methy = self.tok_methy(methy.unsqueeze(-1))

        if mask_bool is not None:
            value_rna = torch.where(
                mask_bool[..., 0:1], self.mask_rna.expand_as(value_rna), value_rna
            )
            value_cnv = torch.where(
                mask_bool[..., 1:2], self.mask_cnv.expand_as(value_cnv), value_cnv
            )
            value_methy = torch.where(
                mask_bool[..., 2:3],
                self.mask_methy.expand_as(value_methy),
                value_methy,
            )

        z_rna = value_rna + self.emb_rna
        z_cnv = value_cnv + self.emb_cnv
        z_methy = value_methy + self.emb_methy

        if self.gene_id_embedding:
            identity = self.get_gene_embedding()
            if identity is not None:
                if identity.dim() == 2:
                    identity = identity.unsqueeze(0)
                z_rna = z_rna + identity
                z_cnv = z_cnv + identity
                z_methy = z_methy + identity

        return z_rna, z_cnv, z_methy
