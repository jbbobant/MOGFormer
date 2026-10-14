"""Removal of processing-batch effects with ComBat, fitted on the fitting patients.

A cohort assembled from several processing batches can carry per-feature shifts
in location and spread that follow the batch, not the tumour. An encoder trained
on such inputs learns the batch as faithfully as it learns biology: in METABRIC
it made one processing batch a cluster of its own. ComBat (Johnson, Li &
Rabinovic 2007) estimates each batch's additive and multiplicative effect per
feature, and shrinks those estimates towards the batch's average across
features with an empirical-Bayes prior. Many noisy per-feature estimates thus
borrow strength from each other, which matters when a batch holds 150 patients
and 10,000 features.

Two choices follow the project's rules:

* **fit on the fitting patients only.** The batch effects are estimated from
  the training and monitoring patients and applied to every patient, including
  held-out ones, as any other preprocessing step is;
* **no biological covariates.** ComBat can protect covariates such as subtype,
  but they would carry labels into an encoder trained without them, and into
  the probes that judge it. Batches here differ only modestly in subtype mix, so
  little biology is removed with the batch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from mogformer.data.omics import OmicsData

logger = logging.getLogger(__name__)

#: Batch-correction methods accepted by the preprocessing configuration.
BATCH_CORRECTIONS: tuple[str, ...] = ("combat",)


@dataclass
class CombatCorrector:
    """Parametric empirical-Bayes ComBat, location and scale, without covariates.

    Attributes:
        max_iterations: Iterations of the posterior updates per batch.
        tolerance: Relative change at which the updates stop.
        grand_mean: Per-feature mean over the fitting patients.
        pooled_sd: Per-feature pooled within-batch standard deviation.
        batch_shift: Batch label to its shrunken additive effect per feature, in
            standardised units.
        batch_scale: Batch label to its shrunken multiplicative effect per
            feature (a standard-deviation ratio).
    """

    max_iterations: int = 100
    tolerance: float = 1e-4
    grand_mean: np.ndarray | None = None
    pooled_sd: np.ndarray | None = None
    batch_shift: dict[object, np.ndarray] = field(default_factory=dict)
    batch_scale: dict[object, np.ndarray] = field(default_factory=dict)

    def fit(
        self, values: np.ndarray, batches: np.ndarray | Sequence[object]
    ) -> CombatCorrector:
        """Estimate every batch's effect per feature on the fitting patients.

        Args:
            values: Patients x features; missing values are allowed and ignored.
            batches: Batch label per patient.

        Returns:
            The fitted corrector.

        Raises:
            ValueError: If a batch has fewer than two patients.
        """
        values = np.asarray(values, dtype=float)
        labels = np.asarray(batches)
        levels, counts = np.unique(labels, return_counts=True)
        if (counts < 2).any():
            raise ValueError(
                "every batch needs two patients; sizes "
                f"{dict(zip(levels, counts, strict=True))}"
            )

        batch_means = {b: np.nanmean(values[labels == b], axis=0) for b in levels}
        weights = counts / counts.sum()
        self.grand_mean = np.sum(
            [w * batch_means[b] for b, w in zip(levels, weights, strict=True)], axis=0
        )
        residual = values - np.stack([batch_means[b] for b in labels])
        self.pooled_sd = np.sqrt(np.nanmean(residual**2, axis=0))
        self.pooled_sd[~np.isfinite(self.pooled_sd) | (self.pooled_sd == 0)] = 1.0
        z = (values - self.grand_mean) / self.pooled_sd

        for b in levels:
            zb = z[labels == b]
            shift_hat = np.nanmean(zb, axis=0)
            scale_hat = np.nanvar(zb, axis=0, ddof=1)
            scale_hat[~np.isfinite(scale_hat) | (scale_hat <= 0)] = 1.0
            n = np.sum(~np.isnan(zb), axis=0).astype(float)
            # Priors across features: normal for the shift, inverse gamma for
            # the variance, both by the method of moments.
            gamma_bar, tau2 = np.nanmean(shift_hat), np.nanvar(shift_hat, ddof=1)
            m, s2 = np.nanmean(scale_hat), np.nanvar(scale_hat, ddof=1)
            lam = (2 * s2 + m**2) / s2
            theta = (m * s2 + m**3) / s2
            shift, scale = shift_hat.copy(), scale_hat.copy()
            for _ in range(self.max_iterations):
                new_shift = (n * tau2 * shift_hat + scale * gamma_bar) / (
                    n * tau2 + scale
                )
                squares = np.nansum((zb - new_shift) ** 2, axis=0)
                new_scale = (theta + 0.5 * squares) / (n / 2.0 + lam - 1.0)
                change = max(
                    np.nanmax(np.abs(new_shift - shift) / (np.abs(shift) + 1e-12)),
                    np.nanmax(np.abs(new_scale - scale) / (np.abs(scale) + 1e-12)),
                )
                shift, scale = new_shift, new_scale
                if change < self.tolerance:
                    break
            self.batch_shift[b] = shift
            self.batch_scale[b] = scale
            logger.info(
                "ComBat batch %s: %d patients, mean |shift| %.3f SD, mean scale %.3f",
                b,
                int((labels == b).sum()),
                float(np.nanmean(np.abs(shift))),
                float(np.nanmean(np.sqrt(scale))),
            )
        return self

    def transform(
        self, values: np.ndarray, batches: np.ndarray | Sequence[object]
    ) -> np.ndarray:
        """Remove the fitted batch effects.

        Args:
            values: Patients x features, on the scale the corrector was fitted.
            batches: Batch label per patient; every label must have been seen
                at fitting.

        Returns:
            The corrected values; missing values stay missing.

        Raises:
            RuntimeError: If the corrector has not been fitted.
            ValueError: If a batch was not seen at fitting.
        """
        if self.grand_mean is None or self.pooled_sd is None:
            raise RuntimeError("fit the corrector before transforming")
        values = np.asarray(values, dtype=float)
        labels = np.asarray(batches)
        unseen = set(np.unique(labels)) - set(self.batch_shift)
        if unseen:
            raise ValueError(f"labels not seen at fitting: {sorted(unseen)}")
        shift = np.stack([self.batch_shift[b] for b in labels])
        scale = np.stack([self.batch_scale[b] for b in labels])
        z = (values - self.grand_mean) / self.pooled_sd
        return (z - shift) / np.sqrt(scale) * self.pooled_sd + self.grand_mean


def load_batch_labels(
    path: str | Path, column: str, patients: Sequence[str]
) -> np.ndarray:
    """Read each patient's processing batch from a per-patient table.

    Args:
        path: CSV with patients in its first column.
        column: Column holding the batch.
        patients: Patients, in the order of the data.

    Returns:
        Batch label per patient, as strings.

    Raises:
        ValueError: If a patient has no batch.
    """
    table = pd.read_csv(path, index_col=0)
    labels = table[column].reindex(list(patients))
    missing = labels.isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} patients have no {column!r} in {path}, "
            f"e.g. {list(labels[missing].index[:3])}"
        )
    return labels.astype(str).to_numpy()


def correct_batches(
    data: OmicsData,
    batches: np.ndarray,
    fit_rows: np.ndarray,
    modalities: Sequence[str],
) -> tuple[OmicsData, dict[str, CombatCorrector]]:
    """Remove processing-batch effects from chosen omics, fitted on some patients.

    Args:
        data: The cohort, raw values.
        batches: Batch label per patient, in the data's row order.
        fit_rows: Rows the batch effects are estimated on.
        modalities: Omics blocks to correct; the others are left as they are.

    Returns:
        A copy of the cohort with the named blocks corrected, and the fitted
        corrector per omics.
    """
    values = data.X.astype(float).copy()
    correctors = {}
    for modality in modalities:
        block = data.block_slice(modality)
        corrector = CombatCorrector().fit(values[fit_rows, block], batches[fit_rows])
        values[:, block] = corrector.transform(values[:, block], batches)
        correctors[modality] = corrector
        logger.info(
            "ComBat corrected %s across %d batches",
            modality,
            len(corrector.batch_shift),
        )
    return replace(data, X=values), correctors
