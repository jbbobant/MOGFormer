"""Objectives, seeding and the per-fold training loop."""

from __future__ import annotations

from mogformer.training.losses import (
    MultiClassFocalLoss,
    embedding_correlation_penalty,
    info_nce,
    inverse_frequency_weights,
    masked_huber_dual,
    participation_ratio,
    rank_me,
    uniformity_loss,
    vicreg_covariance,
    vicreg_variance,
)
from mogformer.training.regularisers import EmbeddingMemoryBank, ProjectionHead
from mogformer.training.seed import seed_everything
from mogformer.training.trainer import (
    ARCHIVED_GATE_LR_MULTIPLIER,
    GATE_LR_MULTIPLIER,
    MaskedReconstructionTrainer,
    TrainingHistory,
)

__all__ = [
    "ARCHIVED_GATE_LR_MULTIPLIER",
    "GATE_LR_MULTIPLIER",
    "EmbeddingMemoryBank",
    "MaskedReconstructionTrainer",
    "MultiClassFocalLoss",
    "ProjectionHead",
    "TrainingHistory",
    "embedding_correlation_penalty",
    "info_nce",
    "inverse_frequency_weights",
    "masked_huber_dual",
    "participation_ratio",
    "rank_me",
    "seed_everything",
    "uniformity_loss",
    "vicreg_covariance",
    "vicreg_variance",
]
