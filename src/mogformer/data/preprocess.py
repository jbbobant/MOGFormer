"""Leakage-free preprocessing as a single scikit-learn transformer.

:class:`MultiOmicsTransformer` carries every statistic that must not see the
held-out fold: variable-gene selection, imputation medians, the RNA log
transform and per-modality scaling. Placed in front of a classifier inside a
:class:`sklearn.pipeline.Pipeline`, all of it refits on the training fold and is
merely applied to the test fold — including inside the inner loop of a
hyperparameter search.

Selected genes therefore differ from fold to fold. That is correct, not a bug:
selection is a fitted parameter like any other.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from mogformer.data.omics import MODALITY_ORDER

logger = logging.getLogger(__name__)


#: Scales expression can arrive on. Counts are log-transformed here; values
#: already on a log or standardised scale are not, because ``log1p`` after
#: clipping at zero would erase every below-average value.
RNA_SCALES: tuple[str, ...] = ("counts", "log")


class MultiOmicsTransformer(BaseEstimator, TransformerMixin):
    """Select variable genes, impute, log RNA and standardise, all per fold.

    Genes are ranked by median absolute deviation within each active modality
    and combined into a consensus rank by summing the per-modality ranks. The
    top ``top_k`` of that pool is kept, and curated genes are force-included
    afterwards so canonical markers survive regardless of their variance.

    Attributes:
        selected_idx_: Column indices of the selected genes, set by ``fit``.
        selected_genes_: Names of the selected genes, set by ``fit``.
        n_curated_selected_: How many selected genes came from the curated list.
        medians_: Per-modality imputation medians, set by ``fit``.
        scalers_: Per-modality fitted scalers, set by ``fit``.
        n_features_out_: Width of the transformed matrix.
    """

    def __init__(
        self,
        n_genes: int,
        gene_names: Sequence[str],
        top_k: int = 1000,
        curated_genes: Sequence[str] | None = None,
        active_modalities: Sequence[str] = MODALITY_ORDER,
        mad_on_log_rna: bool = False,
        impute: bool = True,
        rna_scale: str = "counts",
        clip_z: float | None = None,
    ) -> None:
        """Store hyperparameters verbatim.

        ``sklearn.base.clone`` reconstructs an estimator from these attributes
        and requires them to be stored without coercion, so no validation or
        conversion happens here; both are deferred to ``fit`` and to the private
        coercion properties.

        Args:
            n_genes: Genes per modality block in the input matrix.
            gene_names: Gene symbols, in column order within each block.
            top_k: Size of the variance-ranked pool kept before curated genes
                are added on top.
            curated_genes: Genes to force-include after ranking, or None.
            active_modalities: Subset of :data:`MODALITY_ORDER` to use. Drives
                both the consensus rank and which blocks are emitted, so it is
                the switch behind the modality-contribution experiments.
            mad_on_log_rna: Rank RNA by the deviation of ``log1p(rna)`` rather
                than of raw counts. False matches the original pipeline.
            impute: Replace missing values with per-gene training medians.
            rna_scale: One of :data:`RNA_SCALES`. ``"counts"`` takes
                ``log1p`` of expression before scaling; ``"log"`` scales it
                as it is, for cohorts distributed already logged or
                standardised (METABRIC's microarray z-scores).
            clip_z: Clip standardised values to ``[-clip_z, clip_z]``. Heavy
                tails — rare high-level amplifications, outlying expression —
                otherwise reach dozens of standard deviations, as model inputs
                and as reconstruction targets. None leaves them unbounded.
        """
        self.n_genes = n_genes
        self.gene_names = gene_names
        self.top_k = top_k
        self.curated_genes = curated_genes
        self.active_modalities = active_modalities
        self.mad_on_log_rna = mad_on_log_rna
        self.impute = impute
        self.rna_scale = rna_scale
        self.clip_z = clip_z

    @property
    def _active(self) -> tuple[str, ...]:
        """Return the active modalities as a tuple, without touching the param."""
        return tuple(self.active_modalities)

    @property
    def _log_rna(self) -> bool:
        """Return whether expression is log-transformed before scaling."""
        return self.rna_scale == "counts"

    @property
    def _curated(self) -> list[str]:
        """Return the curated gene list, without touching the stored param."""
        return list(self.curated_genes) if self.curated_genes is not None else []

    def _block(self, X: np.ndarray, modality: str) -> np.ndarray:
        """Return the columns of ``X`` belonging to one modality.

        Args:
            X: Feature matrix, shape ``(n_samples, 3 * n_genes)``.
            modality: One of :data:`MODALITY_ORDER`.

        Returns:
            View of shape ``(n_samples, n_genes)``.

        Raises:
            KeyError: If ``modality`` is not a known modality.
        """
        if modality not in MODALITY_ORDER:
            raise KeyError(
                f"unknown modality {modality!r}; expected one of {list(MODALITY_ORDER)}"
            )
        offset = MODALITY_ORDER.index(modality) * self.n_genes
        return X[:, offset : offset + self.n_genes]

    @staticmethod
    def _median_absolute_deviation(block: np.ndarray) -> np.ndarray:
        """Return the per-gene median absolute deviation, ignoring missing values.

        Args:
            block: One modality's values, shape ``(n_samples, n_genes)``.

        Returns:
            Deviation per gene, shape ``(n_genes,)``.
        """
        median = np.nanmedian(block, axis=0)
        return np.nanmedian(np.abs(block - median), axis=0)

    @staticmethod
    def _apply_impute(block: np.ndarray, medians: np.ndarray) -> np.ndarray:
        """Fill missing values with the supplied per-gene medians.

        Args:
            block: Values to fill, shape ``(n_samples, n_genes)``.
            medians: Training medians, shape ``(n_genes,)``.

        Returns:
            A filled copy; the input is not modified.
        """
        filled = block.copy()
        missing = np.isnan(filled)
        if missing.any():
            filled[missing] = medians[np.where(missing)[1]]
        return filled

    def fit(self, X: np.ndarray, y: Any = None) -> MultiOmicsTransformer:
        """Select genes and fit imputation medians and scalers on ``X``.

        Args:
            X: Training features, shape ``(n_samples, 3 * n_genes)``.
            y: Ignored; present for the scikit-learn API.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If ``X`` does not carry three blocks of ``n_genes``
                columns, if ``top_k`` is not positive, if ``rna_scale`` is
                unknown, or if expression declared as counts is negative.
        """
        expected_width = len(MODALITY_ORDER) * self.n_genes
        if X.shape[1] != expected_width:
            raise ValueError(
                f"X has {X.shape[1]} columns but {len(MODALITY_ORDER)} blocks of "
                f"{self.n_genes} genes require {expected_width}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.rna_scale not in RNA_SCALES:
            raise ValueError(
                f"unknown rna_scale {self.rna_scale!r}; available: {list(RNA_SCALES)}"
            )
        if self._log_rna and "rna" in self._active:
            rna = self._block(X, "rna")
            if np.nanmin(rna) < 0:
                raise ValueError(
                    "expression has negative values but rna_scale is 'counts'; "
                    "log1p would clip them to zero. Set preprocess.rna_scale: log "
                    "for expression that is already logged or standardised."
                )

        genes = np.asarray(self.gene_names)
        curated = set(self._curated)

        # Consensus rank: sum the descending per-modality deviation ranks, so a
        # low total means a gene is variable across the active modalities.
        rank_total = np.zeros(self.n_genes, dtype=np.float64)
        for modality in self._active:
            block = self._block(X, modality).astype(np.float64)
            if modality == "rna" and self.mad_on_log_rna and self._log_rna:
                block = np.log1p(np.clip(block, 0, None))
            rank_total += rankdata(
                -self._median_absolute_deviation(block), method="average"
            )

        ordered = np.argsort(rank_total, kind="stable")
        curated_positions = [i for i, gene in enumerate(genes) if gene in curated]
        pool = [int(i) for i in ordered if genes[i] not in curated]
        selected = list(
            dict.fromkeys(curated_positions + pool[: min(self.top_k, len(pool))])
        )

        self.selected_idx_ = np.asarray(selected, dtype=int)
        self.selected_genes_ = genes[self.selected_idx_].tolist()
        self.n_curated_selected_ = len(curated_positions)

        self.medians_: dict[str, np.ndarray] = {}
        self.scalers_: dict[str, StandardScaler] = {}
        for modality in self._active:
            block = self._block(X, modality)[:, self.selected_idx_].astype(np.float64)
            if self.impute:
                medians = np.nanmedian(block, axis=0)
                # A gene missing in every training patient has no median.
                medians = np.where(np.isnan(medians), 0.0, medians)
                self.medians_[modality] = medians
                block = self._apply_impute(block, medians)
            if modality == "rna" and self._log_rna:
                block = np.log1p(np.clip(block, 0, None))
            self.scalers_[modality] = StandardScaler().fit(block)

        self.n_features_out_ = len(self.selected_idx_) * len(self._active)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted selection, imputation, log and scaling.

        Args:
            X: Features to transform, shape ``(n_samples, 3 * n_genes)``.

        Returns:
            Transformed matrix of shape ``(n_samples, n_features_out_)``, with
            active modality blocks concatenated in :data:`MODALITY_ORDER`.
        """
        parts = []
        for modality in self._active:
            block = self._block(X, modality)[:, self.selected_idx_].astype(np.float64)
            if self.impute:
                block = self._apply_impute(block, self.medians_[modality])
            if modality == "rna" and self._log_rna:
                block = np.log1p(np.clip(block, 0, None))
            scaled = self.scalers_[modality].transform(block)
            if self.clip_z is not None:
                scaled = np.clip(scaled, -self.clip_z, self.clip_z)
            parts.append(scaled)
        return np.hstack(parts)

    def observed_mask(self, X: np.ndarray) -> np.ndarray:
        """Return which transformed entries were measured rather than imputed.

        Imputed entries hold a training median, not a measurement, so a
        reconstruction objective must not score them: it would reward the
        model for predicting the imputation rule.

        Args:
            X: Features, shape ``(n_samples, 3 * n_genes)``, as given to
                :meth:`transform`.

        Returns:
            Boolean array of shape ``(n_samples, n_features_out_)``, True where
            the value was observed, laid out like :meth:`transform`'s output.
        """
        return np.hstack(
            [
                ~np.isnan(self._block(X, modality)[:, self.selected_idx_])
                for modality in self._active
            ]
        )

    def get_selected_gene_names(self) -> list[str]:
        """Return the genes selected by the last ``fit``.

        Exposed so downstream importance aggregation can align by gene name
        across folds, where the selected set differs.

        Returns:
            Selected gene symbols, curated genes first.
        """
        return list(self.selected_genes_)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return output column names as ``modality:gene``.

        Args:
            input_features: Ignored; present for the scikit-learn API.

        Returns:
            Object array of length ``n_features_out_``.
        """
        names = [
            f"{modality}:{gene}"
            for modality in self._active
            for gene in self.selected_genes_
        ]
        return np.asarray(names, dtype=object)
