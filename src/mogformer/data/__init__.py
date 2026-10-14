"""Cohort loading, leakage-free preprocessing and shared cross-validation folds."""

from __future__ import annotations

from mogformer.data.folds import (
    FoldSpec,
    assert_folds_match,
    assert_no_leakage,
    load_folds,
    make_folds,
    remap_fold_indices,
    save_folds,
    select_folds,
)
from mogformer.data.gene_embeddings import (
    align_gene_embeddings,
    load_gene_embedding_bundle,
    load_gene_embedding_table,
    save_gene_embedding_table,
)
from mogformer.data.omics import (
    MODALITY_ORDER,
    OmicsData,
    beta_to_m_values,
    exclude_classes,
    load_curated_genes,
    load_omics,
    load_omics_bundle,
    save_omics_bundle,
)
from mogformer.data.preprocess import MultiOmicsTransformer

__all__ = [
    "MODALITY_ORDER",
    "FoldSpec",
    "MultiOmicsTransformer",
    "OmicsData",
    "align_gene_embeddings",
    "assert_folds_match",
    "assert_no_leakage",
    "beta_to_m_values",
    "exclude_classes",
    "load_curated_genes",
    "load_folds",
    "load_gene_embedding_bundle",
    "load_gene_embedding_table",
    "load_omics",
    "load_omics_bundle",
    "make_folds",
    "remap_fold_indices",
    "save_folds",
    "save_gene_embedding_table",
    "save_omics_bundle",
    "select_folds",
]
