"""Training-time modules behind the embedding regularisers.

Both regularisers act on the patient summary, which a 32-patient batch samples
too thinly: a 128-dimensional covariance estimated from 32 rows has rank at
most 31, and a uniformity term over 32 points sees a sixth of the cohort. A
memory bank of recent, detached embeddings supplies the missing rows, and the
negatives of the contrastive loss.

A projection head maps the embedding to the space where a regulariser acts:
VICReg's expander (wide, decorrelation imposed there, as in the original
method) and the contrastive head (narrow, as in SimCLR). The head absorbs much
of what the loss demands, so the embedding before it keeps more structure.
Neither module is part of the frozen encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingMemoryBank:
    """First-in, first-out store of recent embeddings, detached.

    Attributes:
        size: Maximum number of rows kept.
    """

    def __init__(self, size: int) -> None:
        """Create an empty bank.

        Args:
            size: Maximum number of rows kept; 0 keeps none.
        """
        self.size = size
        self._rows: torch.Tensor | None = None

    @property
    def values(self) -> torch.Tensor | None:
        """The stored rows, newest last, or None while empty."""
        return self._rows

    @torch.no_grad()
    def push(self, embeddings: torch.Tensor) -> None:
        """Add a batch, dropping the oldest rows beyond ``size``.

        Args:
            embeddings: Rows to store, shape ``(batch, d)``.
        """
        if self.size <= 0:
            return
        rows = embeddings.detach()
        if self._rows is not None:
            rows = torch.cat([self._rows, rows], 0)
        self._rows = rows[-self.size :]


class ProjectionHead(nn.Module):
    """Two-layer projection head: Linear → BatchNorm → ReLU → Linear.

    Attributes:
        net: The two-layer network.
    """

    def __init__(
        self, d_in: int, width: int = 512, out_width: int | None = None
    ) -> None:
        """Build the head.

        Args:
            d_in: Width of the patient embedding.
            width: Width of the hidden layer.
            out_width: Width of the output; ``width`` when None (VICReg's
                expander).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, out_width or width),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project a batch of embeddings, shape ``(batch, d_in)``."""
        return self.net(embeddings)
