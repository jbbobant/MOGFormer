"""Attention biased by biological graph structure.

Standard self-attention over a gene sequence is permutation-invariant and knows
nothing about which genes interact. These layers add a learnable bias indexed by
shortest-path distance, so attention is encouraged along known pathways while
remaining free to discover interactions absent from the reference network.

Two bias formulations are supported, because the project measures them against
each other: ``inside`` places the bias within the softmax, as Graphormer does,
and ``dual`` averages two separately normalised distributions.

Optionally a second bias term keyed on signed regulatory edges is added, with
separate learnable magnitudes for activation and repression.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

#: Ways the structural bias can enter the attention computation.
ATTENTION_BIAS_MODES: tuple[str, ...] = ("inside", "dual")

#: Initial value of the raw per-head gate. ``softplus(0.5413) ~ 1.0``, so the
#: gate starts at neutral scale and is free to grow or shrink during training.
_NEUTRAL_GATE_LOGIT = 0.5413


class StructuralGraphAttention(nn.Module):
    """Multi-head attention with a distance bias and an optional regulatory bias.

    The distance bias table is sized ``max_distance + 3`` to cover shortest-path
    values in ``[0, max_distance + 1]`` plus one dedicated bucket for the tumor
    summary token, which the enclosing
    :class:`~mogformer.models.layers.global_transformer.GlobalGraphTransformer`
    assigns the sentinel distance ``max_distance + 2``. Giving that token its own
    bucket matters: padding its row with zero instead would place it in the same
    bias bucket as every gene's own diagonal, conflating "this is me" with "this
    is the whole-tumor summary".

    Attributes:
        num_heads: Number of attention heads.
        d_head: Width of each head.
        mode: Active bias formulation, one of :data:`ATTENTION_BIAS_MODES`.
        use_grn: Whether the regulatory bias is applied. Set by the enclosing
            transformer once it knows a regulatory graph was supplied.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_distance: int = 5,
        mode: str = "inside",
        dropout: float = 0.1,
        lambda_gate: bool = False,
    ) -> None:
        """Build the projections and the bias tables.

        Args:
            d_model: Token width; must divide evenly by ``num_heads``.
            num_heads: Number of attention heads.
            max_distance: Largest hop count represented exactly by the bias.
            mode: Bias formulation, one of :data:`ATTENTION_BIAS_MODES`.
            dropout: Dropout applied to the attention weights.
            lambda_gate: Give each head a learnable scalar scaling its distance
                bias, letting heads specialise into structural and semantic
                roles. Only meaningful in ``inside`` mode.

        Raises:
            ValueError: If ``mode`` is unknown or ``d_model`` is not divisible
                by ``num_heads``.
        """
        super().__init__()
        if mode not in ATTENTION_BIAS_MODES:
            raise ValueError(
                f"unknown attention bias mode {mode!r}; "
                f"available: {list(ATTENTION_BIAS_MODES)}"
            )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not divisible by num_heads {num_heads}"
            )

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.mode = mode

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # +3 covers distances 0..max_distance, the unreachable bucket at
        # max_distance + 1, and the summary-token bucket at max_distance + 2.
        self.spatial_embedding = nn.Embedding(max_distance + 3, num_heads)

        # Regulatory bias, activated by the enclosing transformer.
        self.use_grn = False
        self.b_grn_activation = nn.Parameter(torch.zeros(num_heads))
        self.b_grn_repression = nn.Parameter(torch.zeros(num_heads))

        self.attn_drop = nn.Dropout(dropout)

        self.lambda_gate = lambda_gate
        if lambda_gate:
            self.lambda_raw = nn.Parameter(
                torch.full((num_heads,), _NEUTRAL_GATE_LOGIT)
            )

    def _structural_bias(
        self, spd_matrix: torch.Tensor, grn_matrix: torch.Tensor | None
    ) -> torch.Tensor:
        """Build the additive bias from distances and regulatory edges.

        Args:
            spd_matrix: Integer distances, shape ``(seq_len, seq_len)``.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.

        Returns:
            Bias of shape ``(num_heads, seq_len, seq_len)``.
        """
        # tanh bounds the distance bias so it cannot swamp the semantic scores.
        bias = torch.tanh(self.spatial_embedding(spd_matrix).permute(2, 0, 1))

        if self.use_grn and grn_matrix is not None:
            activation = (grn_matrix > 0).float().unsqueeze(0)
            repression = (grn_matrix < 0).float().unsqueeze(0)
            bias = bias + (
                self.b_grn_activation.view(-1, 1, 1) * activation
                + self.b_grn_repression.view(-1, 1, 1) * repression
            )
        return bias

    def forward(
        self,
        h: torch.Tensor,
        spd_matrix: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        structural_bias: bool = True,
    ) -> torch.Tensor:
        """Attend over the token sequence.

        Args:
            h: Token states, shape ``(batch, seq_len, d_model)``.
            spd_matrix: Integer distances, shape ``(seq_len, seq_len)``, already
                padded for any prepended summary token.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.
            structural_bias: When False, fall back to plain scaled dot-product
                attention. This is the switch behind the no-structure ablation,
                and it bypasses the regulatory bias too.

        Returns:
            Updated token states, shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = h.shape

        query = (
            self.q_proj(h)
            .view(batch, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(h)
            .view(batch, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(h)
            .view(batch, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)

        if not structural_bias:
            attention = F.softmax(scores, dim=-1)
        else:
            bias = self._structural_bias(spd_matrix, grn_matrix)
            bias = bias.unsqueeze(0).expand(batch, -1, -1, -1)

            if self.mode == "inside":
                if self.lambda_gate:
                    scale = F.softplus(self.lambda_raw).view(1, self.num_heads, 1, 1)
                    bias = scale * bias
                attention = F.softmax(scores + bias, dim=-1)
            else:
                # Two separately normalised distributions, averaged so the rows
                # still sum to one.
                attention = (F.softmax(scores, dim=-1) + F.softmax(bias, dim=-1)) / 2.0

        attention = self.attn_drop(attention)
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)


class StructuralAttentionBlock(nn.Module):
    """Pre-layer-norm transformer block wrapping :class:`StructuralGraphAttention`.

    Pre-norm rather than post-norm because the deeper configurations of this
    model train unstably otherwise at the small batch sizes the cohort allows.

    Attributes:
        attn: The structural attention layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        max_distance: int,
        mode: str,
        dropout: float,
        lambda_gate: bool = False,
    ) -> None:
        """Build the attention and feed-forward sublayers.

        Args:
            d_model: Token width.
            num_heads: Number of attention heads.
            dim_feedforward: Hidden width of the feed-forward sublayer.
            max_distance: Largest hop count represented exactly by the bias.
            mode: Bias formulation, one of :data:`ATTENTION_BIAS_MODES`.
            dropout: Dropout for attention weights and the feed-forward sublayer.
            lambda_gate: Per-head learnable scaling of the distance bias.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StructuralGraphAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_distance=max_distance,
            mode=mode,
            dropout=dropout,
            lambda_gate=lambda_gate,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        spd_matrix: torch.Tensor,
        grn_matrix: torch.Tensor | None = None,
        structural_bias: bool = True,
    ) -> torch.Tensor:
        """Apply structural attention then the feed-forward sublayer.

        Args:
            x: Token states, shape ``(batch, seq_len, d_model)``.
            spd_matrix: Integer distances, shape ``(seq_len, seq_len)``.
            grn_matrix: Signed regulatory adjacency of the same shape, or None.
            structural_bias: When False, attention ignores both graph biases.

        Returns:
            Updated token states of the same shape as ``x``.
        """
        x = x + self.attn(
            self.norm1(x), spd_matrix, grn_matrix, structural_bias=structural_bias
        )
        return x + self.ffn(self.norm2(x))
