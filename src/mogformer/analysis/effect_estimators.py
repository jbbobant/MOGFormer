"""Model-free estimates of a transcription factor's effect on a target.

The trans probe reads a trained encoder. Whether its answers are worth more
than simple statistics of the same matrix is an empirical question, so every
probe estimate is set against two comparators computed on the same inputs.

**Conditional regression.** A ridge regression of the target's expression on
every channel left visible when the target is fully hidden, read at the
factor's expression. This is the linear version of what the probe estimates: a
conditional dependence given everything else observed. A deep model that cannot
beat it has learned nothing about regulation that linear algebra did not.

**Copy number as an instrument.** A factor's own copy number moves its
expression and — if it touches the target only through that expression — is a
natural experiment. The Wald ratio ``cov(c_f, x_t) / cov(c_f, x_f)`` then
estimates the causal effect even under hidden confounding, which neither the
probe nor the regression can do. The exclusion restriction fails when factor
and target share an amplicon, so instruments are only as good as that
assumption; weak ones are withheld via the first-stage F statistic.
"""

from __future__ import annotations

import logging

import numpy as np

from mogformer.analysis.probe_cis import MODALITY_INDEX

logger = logging.getLogger(__name__)

#: Conventional first-stage F below which an instrument is treated as weak.
MIN_FIRST_STAGE_F = 10.0

#: Ridge penalties searched, per sample, on standardised inputs.
RIDGE_PENALTIES: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def wald_ratio_effects(
    inputs: np.ndarray,
    factors: np.ndarray,
    targets: np.ndarray,
    min_first_stage_f: float = MIN_FIRST_STAGE_F,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate factor-to-target effects with the factor's copy number as instrument.

    Args:
        inputs: Standardised values, shape ``(n_patients, n_genes, 3)``.
        factors: Factor gene index per pair, shape ``(n_pairs,)``.
        targets: Target gene index per pair, same shape.
        min_first_stage_f: Instruments weaker than this return NaN.

    Returns:
        The Wald-ratio effect per pair, in standardised units, and the
        first-stage F statistic per pair.
    """
    rna, cnv = MODALITY_INDEX["rna"], MODALITY_INDEX["cnv"]
    n_patients = inputs.shape[0]
    centred = inputs - inputs.mean(axis=0, keepdims=True)
    instrument = centred[:, factors, cnv]
    exposure = centred[:, factors, rna]
    outcome = centred[:, targets, rna]

    first_stage = (instrument * exposure).mean(axis=0)
    reduced_form = (instrument * outcome).mean(axis=0)
    correlation = first_stage / (instrument.std(axis=0) * exposure.std(axis=0) + 1e-12)
    first_stage_f = (
        (n_patients - 2) * correlation**2 / np.maximum(1.0 - correlation**2, 1e-12)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        effect = reduced_form / first_stage
    effect = np.where(first_stage_f >= min_first_stage_f, effect, np.nan)
    return effect, first_stage_f


def _visible_columns(n_genes: int, target: int) -> np.ndarray:
    """Flattened ``(gene, channel)`` columns visible while a target is hidden."""
    columns = np.arange(n_genes * 3).reshape(n_genes, 3)
    return np.delete(columns, target, axis=0).ravel()


def _ridge_coefficients(
    covariance: np.ndarray, target: int, n_genes: int, penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one target's ridge regression from the shared covariance matrix."""
    visible = _visible_columns(n_genes, target)
    response = target * 3 + MODALITY_INDEX["rna"]
    gram = covariance[np.ix_(visible, visible)] + penalty * np.eye(len(visible))
    return visible, np.linalg.solve(gram, covariance[visible, response])


def select_ridge_penalty(
    inputs: np.ndarray,
    penalties: tuple[float, ...] = RIDGE_PENALTIES,
    n_folds: int = 5,
    n_targets: int = 40,
    seed: int = 0,
) -> float:
    """Choose the ridge penalty by cross-validated expression prediction.

    Args:
        inputs: Standardised values, shape ``(n_patients, n_genes, 3)``.
        penalties: Candidate penalties, per sample.
        n_folds: Folds over patients.
        n_targets: Targets the error is averaged over.
        seed: Seed for the folds and the target subset.

    Returns:
        The penalty with the lowest held-out squared error.
    """
    rng = np.random.default_rng(seed)
    n_patients, n_genes, _ = inputs.shape
    flat = inputs.reshape(n_patients, -1).astype(np.float64)
    folds = rng.permutation(n_patients) % n_folds
    chosen = rng.choice(n_genes, size=min(n_targets, n_genes), replace=False)
    errors = np.zeros(len(penalties))
    for fold in range(n_folds):
        train, test = flat[folds != fold], flat[folds == fold]
        covariance = train.T @ train / len(train)
        for target in chosen:
            response = target * 3 + MODALITY_INDEX["rna"]
            for position, penalty in enumerate(penalties):
                visible, beta = _ridge_coefficients(
                    covariance, target, n_genes, penalty
                )
                residual = test[:, response] - test[:, visible] @ beta
                errors[position] += float((residual**2).mean())
    best = float(penalties[int(np.argmin(errors))])
    logger.info("ridge penalty %.3g chosen by %d-fold cross-validation", best, n_folds)
    return best


def ridge_conditional_effects(
    inputs: np.ndarray,
    factors: np.ndarray,
    targets: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Estimate each pair's conditional effect by ridge regression.

    The target's three channels are hidden, as in the probe's ``whole_gene``
    mode, and its expression is regressed on every other gene's three channels.
    The coefficient at the factor's expression is the pair's estimate. One
    regression per distinct target serves all of its factors, and all
    regressions share one covariance matrix.

    Args:
        inputs: Standardised values, shape ``(n_patients, n_genes, 3)``.
        factors: Factor gene index per pair, shape ``(n_pairs,)``.
        targets: Target gene index per pair, same shape.
        penalty: Ridge penalty, per sample.

    Returns:
        Coefficient per pair, in standardised units.
    """
    n_patients, n_genes, _ = inputs.shape
    flat = inputs.reshape(n_patients, -1).astype(np.float64)
    covariance = flat.T @ flat / n_patients
    effects = np.full(len(factors), np.nan)
    for target in np.unique(targets):
        visible, beta = _ridge_coefficients(covariance, int(target), n_genes, penalty)
        position_of = {int(column): i for i, column in enumerate(visible)}
        for pair in np.nonzero(targets == target)[0]:
            column = int(factors[pair]) * 3 + MODALITY_INDEX["rna"]
            effects[pair] = beta[position_of[column]]
    return effects
