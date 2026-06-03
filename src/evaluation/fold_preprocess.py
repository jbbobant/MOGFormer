"""
preprocess.py — Leakage-free preprocessing as a single sklearn transformer.

This is the heart of the anti-leakage requirement. 
MultiOmicsTransformer implements fit/transform so that, when placed in front of a classifier inside
a Pipeline, ALL of the following refit on the training fold ONLY and are merely
applied to the held-out fold:

    1. MAD-based HVG selection via consensus rank across ACTIVE modalities
    2. force-include of curated genes (added AFTER ranking)
    3. median imputation (per gene)
    4. ln(x+1) on RNA only
    5. StandardScaler on every active modality

Block layout of input X: [ RNA | CNV | methy ], each block of width G with
identical gene ordering (as produced by data.load_omics). `active_modalities`
lets the modality-contribution experiments (spec 6a) switch blocks on/off; the
consensus rank is then computed over active modalities only, and only active
blocks are emitted.

"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

MODALITY_ORDER = ("rna", "cnv", "methy")


class MultiOmicsTransformer(BaseEstimator, TransformerMixin):
    """Per-fold MAD selection + impute + log1p(RNA) + standardize.

    Parameters
    ----------
    n_genes : int
        Number of genes G per modality block in the input X.
    gene_names : sequence of str, length G
        Column (gene) names within each block.
    top_k : int
        Size of the MAD-ranked pool to keep (curated genes are added on top of
        this pool, matching the original preprocess.py behaviour).
    curated_genes : sequence of str
        Genes to force-include after ranking.
    active_modalities : sequence of str
        Subset of ("rna","cnv","methy") to use. Drives both the consensus rank
        and which blocks are emitted.
    mad_on_log_rna : bool
        If True, compute RNA MAD on log1p(RNA). Default False to match the
        original pipeline (MAD on raw RSEM). Surfaced as a flag, not a silent
        choice.
    impute : bool
        Median imputation (fit on train). Default True.
    """

    def __init__(
        self,
        n_genes: int,
        gene_names: Sequence[str],
        top_k: int = 1000,
        curated_genes: Optional[Sequence[str]] = None,
        active_modalities: Sequence[str] = MODALITY_ORDER,
        mad_on_log_rna: bool = False,
        impute: bool = True,
    ):
        
        self.n_genes = n_genes
        self.gene_names = gene_names
        self.top_k = top_k
        self.curated_genes = curated_genes
        self.active_modalities = active_modalities
        self.mad_on_log_rna = mad_on_log_rna
        self.impute = impute

    # lazily coerced views (do NOT touch the stored params)
    @property
    def _active(self):
        return tuple(self.active_modalities)

    @property
    def _curated(self):
        return list(self.curated_genes) if self.curated_genes is not None else []

    # ---- block helpers -------------------------------------------------
    def _block(self, X: np.ndarray, modality: str) -> np.ndarray:
        g = self.n_genes
        off = {"rna": 0, "cnv": g, "methy": 2 * g}[modality]
        return X[:, off:off + g]

    @staticmethod
    def _mad(block: np.ndarray) -> np.ndarray:
        """Median absolute deviation per column (gene), ignoring NaNs."""
        med = np.nanmedian(block, axis=0)
        return np.nanmedian(np.abs(block - med), axis=0)

    # ---- fit -----------------------------------------------------------
    def fit(self, X: np.ndarray, y=None):
        gene_arr = np.asarray(self.gene_names)
        curated_set = set(self._curated)

        # 1) consensus MAD rank over ACTIVE modalities (rank 1 = most variable)
        rank_sum = np.zeros(self.n_genes, dtype=np.float64)
        for m in self._active:
            blk = self._block(X, m).astype(np.float64)
            if m == "rna" and self.mad_on_log_rna:
                blk = np.log1p(np.clip(blk, 0, None))
            mad = self._mad(blk)
            # descending rank: highest MAD -> smallest rank value
            rank_sum += rankdata(-mad, method="average")

        # 2) pool = genes by ascending consensus, curated removed, take top_k
        order = np.argsort(rank_sum, kind="stable")          # most variable first
        curated_idx = [i for i, g in enumerate(gene_arr) if g in curated_set]
        pool_idx = [i for i in order if gene_arr[i] not in curated_set]
        k = min(self.top_k, len(pool_idx))
        pool_top = pool_idx[:k]

        # 3) force-include curated AFTER ranking (curated first, stable order)
        selected = list(dict.fromkeys(curated_idx + pool_top))  # dedup, keep order
        self.selected_idx_ = np.asarray(selected, dtype=int)
        self.selected_genes_ = gene_arr[self.selected_idx_].tolist()
        self.n_curated_selected_ = len(curated_idx)

        # 4) per-modality impute medians + scalers, fit on TRAIN block @ selected
        self.medians_ = {}
        self.scalers_ = {}
        for m in self._active:
            blk = self._block(X, m)[:, self.selected_idx_].astype(np.float64)
            if self.impute:
                med = np.nanmedian(blk, axis=0)
                med = np.where(np.isnan(med), 0.0, med)        # all-NaN gene -> 0
                self.medians_[m] = med
                blk = self._apply_impute(blk, med)
            if m == "rna":
                blk = np.log1p(np.clip(blk, 0, None))
            sc = StandardScaler().fit(blk)
            self.scalers_[m] = sc

        self.n_features_out_ = len(self.selected_idx_) * len(self._active)
        return self

    # ---- transform -----------------------------------------------------
    @staticmethod
    def _apply_impute(blk: np.ndarray, med: np.ndarray) -> np.ndarray:
        out = blk.copy()
        nan_mask = np.isnan(out)
        if nan_mask.any():
            cols = np.where(nan_mask)[1]
            out[nan_mask] = med[cols]
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        parts = []
        for m in self._active:
            blk = self._block(X, m)[:, self.selected_idx_].astype(np.float64)
            if self.impute:
                blk = self._apply_impute(blk, self.medians_[m])
            if m == "rna":
                blk = np.log1p(np.clip(blk, 0, None))
            parts.append(self.scalers_[m].transform(blk))
        return np.hstack(parts)

    # ---- introspection -------------------------------------------------
    def get_selected_gene_names(self) -> List[str]:
        return list(self.selected_genes_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        for m in self._active:
            names += [f"{m}:{g}" for g in self.selected_genes_]
        return np.asarray(names, dtype=object)
