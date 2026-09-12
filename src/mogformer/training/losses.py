"""Objectives for supervised classification and masked reconstruction.

The supervised objective has to survive an extreme class imbalance — in the
breast cancer cohort the rarest subtype is 35 patients out of 949 — without
collapsing into the safe majority guess. Focal loss down-weights examples the
model already classifies confidently, and square-root-dampened class weights
reweight the rare classes without the instability that raw inverse frequencies
produce at that ratio.

The reconstruction objective is Huber rather than squared error because omics
matrices carry genuine biological outliers that a squared penalty would let
dominate the gradient.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MultiClassFocalLoss(nn.Module):
    """Focal cross-entropy with optional per-class weights.

    Attributes:
        gamma: Focusing exponent. Zero recovers weighted cross-entropy; larger
            values concentrate the gradient on poorly classified examples.
        reduction: One of ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 3.0,
        reduction: str = "mean",
    ) -> None:
        """Store the weighting and focusing configuration.

        Args:
            alpha: Per-class weights, shape ``(num_classes,)``, or None for
                unweighted. Registered as a buffer so it follows the module
                across devices.
            gamma: Focusing exponent.
            reduction: One of ``"mean"``, ``"sum"`` or ``"none"``.

        Raises:
            ValueError: If ``reduction`` is unknown or ``gamma`` is negative.
        """
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"unknown reduction {reduction!r}; expected mean, sum or none"
            )
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha: torch.Tensor | None = None
            self.register_buffer("_alpha", None)
        else:
            self.register_buffer("_alpha", alpha.float())

    @property
    def alpha_weights(self) -> torch.Tensor | None:
        """Return the class weights, already on the module's device."""
        weights = self._alpha
        return None if weights is None else cast(torch.Tensor, weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            inputs: Unnormalised logits, shape ``(batch, num_classes)``.
            targets: Integer class labels, shape ``(batch,)``.

        Returns:
            Scalar loss, or per-example losses when ``reduction`` is ``"none"``.
        """
        cross_entropy = F.cross_entropy(inputs, targets, reduction="none")
        # exp(-CE) is the probability assigned to the true class.
        true_class_prob = torch.exp(-cross_entropy)
        focal_term = (1.0 - true_class_prob) ** self.gamma

        weights = self.alpha_weights
        if weights is not None:
            loss = weights.gather(0, targets) * focal_term * cross_entropy
        else:
            loss = focal_term * cross_entropy

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def sqrt_dampened_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute class weights as the square root of inverse frequency.

    Raw inverse frequency at a 480-to-35 ratio makes the rare class dominate the
    gradient and the model swings to over-predicting it. The square root keeps
    the ordering while compressing the range.

    Args:
        y: Integer class labels of the training split, shape ``(n_samples,)``.
        num_classes: Total number of classes, so absent classes still receive an
            entry.

    Returns:
        Weights of shape ``(num_classes,)``, normalised to mean one. Classes
        absent from ``y`` receive weight zero.

    Raises:
        ValueError: If ``num_classes`` is not positive.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")

    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = np.sqrt(1.0 / counts[present])

    mean_weight = weights[present].mean() if present.any() else 1.0
    return weights / mean_weight


def masked_huber_dual(
    xhat_global: torch.Tensor,
    xhat_local: torch.Tensor,
    targets: torch.Tensor,
    mask_bool: torch.Tensor,
    lambda_global: float = 1.0,
    lambda_local: float = 1.0,
    delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score both reconstruction heads on the masked entries only.

    Unmasked entries are excluded: the model can see them, so scoring them would
    reward copying rather than inference.

    Args:
        xhat_global: Global-head predictions, shape ``(batch, n_genes, 3)``.
        xhat_local: Local-head predictions, same shape.
        targets: True values, same shape.
        mask_bool: True where a value was hidden, same shape.
        lambda_global: Weight of the global head's term.
        lambda_local: Weight of the local head's term.
        delta: Huber transition point, in standardised units.

    Returns:
        Tuple of the combined loss and the two detached per-head losses, the
        latter for logging.
    """
    n_masked = mask_bool.sum().clamp(min=1)
    loss_global = (
        F.huber_loss(
            xhat_global[mask_bool], targets[mask_bool], delta=delta, reduction="sum"
        )
        / n_masked
    )
    loss_local = (
        F.huber_loss(
            xhat_local[mask_bool], targets[mask_bool], delta=delta, reduction="sum"
        )
        / n_masked
    )
    total = lambda_global * loss_global + lambda_local * loss_local
    return total, loss_global.detach(), loss_local.detach()


@torch.no_grad()
def participation_ratio(embeddings: torch.Tensor) -> float:
    """Estimate how many dimensions an embedding actually uses.

    Defined as ``(sum of eigenvalues)^2 / sum of squared eigenvalues`` of the
    covariance. A value near the full width means variance is spread evenly; a
    value near one means the representation has collapsed onto a single
    direction, which is the failure mode to watch during pretraining.

    Args:
        embeddings: Embedding matrix, shape ``(n_samples, d)``.

    Returns:
        Effective dimensionality, between 1 and ``d``.
    """
    centred = embeddings - embeddings.mean(0, keepdim=True)
    covariance = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0)
    total = eigenvalues.sum()
    squared = (eigenvalues * eigenvalues).sum()
    return (total * total / squared.clamp(min=1e-12)).item()
