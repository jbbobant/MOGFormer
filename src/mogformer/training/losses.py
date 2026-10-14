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

import math
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


def inverse_frequency_weights(
    y: np.ndarray, num_classes: int, power: float = 0.5
) -> np.ndarray:
    """Compute class weights as inverse frequency raised to a power.

    Power one is full inverse frequency, which at a 480-to-35 ratio makes the
    rare class dominate the gradient and the model swings to over-predicting
    it. The default square root keeps the ordering while compressing the
    range; zero gives every class the same weight.

    Args:
        y: Integer class labels of the training split, shape ``(n_samples,)``.
        num_classes: Total number of classes, so absent classes still receive an
            entry.
        power: Exponent applied to the inverse class frequency.

    Returns:
        Weights of shape ``(num_classes,)``, normalised to mean one. Classes
        absent from ``y`` receive weight zero.

    Raises:
        ValueError: If ``num_classes`` is not positive or ``power`` negative.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if power < 0:
        raise ValueError(f"power must be non-negative, got {power}")

    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = (1.0 / counts[present]) ** power

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
    per_modality: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score both reconstruction heads on the masked entries only.

    Unmasked entries are excluded: the model can see them, so scoring them would
    reward copying rather than inference.

    Pooled over all masked entries, each modality's weight in the loss is its
    share of the masked entries, which the masking weights decide (expression
    is masked twice as often). Averaging within each modality first and then
    summing gives every modality the same weight whatever its masking rate.

    Args:
        xhat_global: Global-head predictions, shape ``(batch, n_genes, 3)``.
        xhat_local: Local-head predictions, same shape.
        targets: True values, same shape.
        mask_bool: True where a value was hidden, same shape.
        lambda_global: Weight of the global head's term.
        lambda_local: Weight of the local head's term.
        delta: Huber transition point, in standardised units.
        per_modality: Average within each modality, then sum over the
            modalities that have masked entries.

    Returns:
        Tuple of the combined loss and the two detached per-head losses, the
        latter for logging.
    """
    if per_modality:
        weights = mask_bool.to(targets.dtype)
        counts = weights.sum(dim=(0, 1))
        present = counts > 0

        def modality_mean_sum(prediction: torch.Tensor) -> torch.Tensor:
            elementwise = F.huber_loss(
                prediction, targets, delta=delta, reduction="none"
            )
            per_modality_loss = (elementwise * weights).sum(dim=(0, 1))
            return (per_modality_loss[present] / counts[present]).sum()

        loss_global = modality_mean_sum(xhat_global)
        loss_local = modality_mean_sum(xhat_local)
        total = lambda_global * loss_global + lambda_local * loss_local
        return total, loss_global.detach(), loss_local.detach()

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
def participation_ratio(embeddings: torch.Tensor | np.ndarray) -> float:
    """Estimate how many dimensions an embedding actually uses.

    Defined as ``(sum of eigenvalues)^2 / sum of squared eigenvalues`` of the
    covariance. A value near the full width means variance is spread evenly; a
    value near one means the representation has collapsed onto a single
    direction, which is the failure mode to watch during pretraining.

    Args:
        embeddings: Embedding matrix, shape ``(n_samples, d)``, as a tensor or
            an array; computed in double precision either way.

    Returns:
        Effective dimensionality, between 1 and ``d``.
    """
    values = torch.as_tensor(embeddings).detach().to(torch.float64)
    centred = values - values.mean(0, keepdim=True)
    covariance = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0)
    total = eigenvalues.sum()
    squared = (eigenvalues * eigenvalues).sum()
    return (total * total / squared.clamp(min=1e-12)).item()


def embedding_correlation_penalty(
    embeddings: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Penalise correlated embedding dimensions within a batch.

    ``(1/d) Σ_{k≠l} R_kl²`` over the batch correlation matrix ``R``. This is
    VICReg's covariance term (Bardes, Ponce & LeCun 2022) computed on
    standardised dimensions. On raw covariances the penalty can be met by
    shrinking the embedding, which costs nothing when a normalisation follows
    it (the cross-attention head layer-normalises the summary tokens); VICReg
    blocks that with a variance term, and correlations are immune to it.

    Args:
        embeddings: Batch embeddings, shape ``(batch, d)``.
        eps: Added to each dimension's variance, so a constant dimension does
            not divide by zero. Kept far below any real variance: a larger
            value would let shrinking the embedding lower the penalty again.

    Returns:
        The penalty, a scalar tensor carrying gradient.
    """
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    scaled = centred / (centred.pow(2).mean(dim=0, keepdim=True) + eps).sqrt()
    correlation = scaled.T @ scaled / embeddings.shape[0]
    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    return off_diagonal.pow(2).sum() / embeddings.shape[1]


def uniformity_loss(
    embeddings: torch.Tensor,
    bank: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Penalise embeddings that crowd together on the unit sphere.

    ``log mean_{i≠j} exp(−t ‖z_i − z_j‖²)`` over the L2-normalised embeddings
    (Wang & Isola 2020). It is lowest when the embeddings spread evenly over
    the sphere, which a reconstruction objective alone does not ask for: added
    to a masked autoencoder, it raised the effective rank of the features
    (U-MAE; Zhang, Wang & Wang 2022). Pairs run from each batch embedding to
    every other batch embedding and to every bank entry, so a small batch
    still sees most of the cohort.

    Args:
        embeddings: Batch embeddings, shape ``(batch, d)``; normalised here.
        bank: Earlier embeddings, shape ``(n_bank, d)``, already normalised
            and detached, or None.
        temperature: The ``t`` above.

    Returns:
        The loss, a scalar tensor carrying gradient through ``embeddings``.
    """
    z = F.normalize(embeddings, dim=1)
    candidates = z if bank is None or len(bank) == 0 else torch.cat([z, bank], 0)
    squared_distance = (2.0 - 2.0 * z @ candidates.T).clamp(min=0)
    logits = -temperature * squared_distance
    batch = z.shape[0]
    self_pairs = torch.zeros_like(logits, dtype=torch.bool)
    self_pairs[:, :batch] = torch.eye(batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_pairs, float("-inf"))
    n_pairs = logits.numel() - batch
    return torch.logsumexp(logits.flatten(), 0) - math.log(n_pairs)


def vicreg_variance(embeddings: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg's variance hinge: keep every dimension's spread at least one.

    ``mean_k max(0, 1 − sqrt(var_k + eps))`` (Bardes, Ponce & LeCun 2022). It is
    what stops a covariance penalty from being met by shrinking dimensions.

    Args:
        embeddings: Expanded embeddings, shape ``(n, D)``.
        eps: Added to each variance before the square root.

    Returns:
        The penalty, a scalar tensor.
    """
    std = (embeddings.var(dim=0) + eps).sqrt()
    return F.relu(1.0 - std).mean()


def vicreg_covariance(embeddings: torch.Tensor) -> torch.Tensor:
    """VICReg's covariance term: squared off-diagonal covariances over ``D``.

    Estimating a ``D``-by-``D`` covariance needs more than ``D`` rows; pass the
    batch concatenated with a memory bank, not a 32-patient batch alone.

    Args:
        embeddings: Expanded embeddings, shape ``(n, D)``.

    Returns:
        The penalty, a scalar tensor.
    """
    centred = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / max(embeddings.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.pow(2).sum() / embeddings.shape[1]


@torch.no_grad()
def rank_me(embeddings: torch.Tensor | np.ndarray, eps: float = 1e-7) -> float:
    """Entropy-based effective rank of an embedding matrix.

    ``exp(−Σ p_k log p_k)`` with ``p_k = s_k / Σs + eps`` over the singular
    values ``s`` of the uncentred embeddings (Garrido et al. 2023). It tracks
    linear-probe quality across self-supervised methods without labels, so it
    can select a checkpoint outcome-blind.

    Args:
        embeddings: Embedding matrix, shape ``(n_samples, d)``.
        eps: Keeps the logarithm finite.

    Returns:
        The effective rank, between 1 and ``min(n_samples, d)``.
    """
    values = torch.as_tensor(embeddings).detach().to(torch.float64)
    singular = torch.linalg.svdvals(values)
    p = singular / singular.sum().clamp(min=1e-12) + eps
    return float(torch.exp(-(p * p.log()).sum()).item())


def info_nce(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    bank: torch.Tensor | None = None,
    temperature: float = 0.3,
) -> torch.Tensor:
    """InfoNCE: each anchor must pick its own positive among all candidates.

    Candidates are every positive in the batch plus the bank's rows, compared
    by cosine similarity over ``temperature`` (van den Oord et al. 2018;
    SimCLR). A low temperature concentrates the loss on the nearest
    negatives, which in a cohort are often patients of the same subtype, and
    pushes them apart; a moderate one (0.3–0.5) tolerates similar patients
    and keeps clusters (Wang & Liu 2021).

    Args:
        anchors: Projections of the first view, shape ``(batch, p)``.
        positives: Projections of the second view of the same patients, same
            shape and row order.
        bank: Earlier projections, already normalised and detached, shape
            ``(n_bank, p)``, used as extra negatives; or None.
        temperature: Softmax temperature.

    Returns:
        The mean loss over anchors, a scalar tensor.
    """
    a = F.normalize(anchors, dim=1)
    b = F.normalize(positives, dim=1)
    candidates = b if bank is None or len(bank) == 0 else torch.cat([b, bank], 0)
    logits = a @ candidates.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)
