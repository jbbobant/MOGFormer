"""
data.py — Raw multi-omics loading, patient-level alignment, label encoding.

CRITICAL DESIGN POINT (anti-leakage):
This module loads the RAW omics matrices (genes x patients) and returns them
UNTOUCHED by any statistic that must be fit per-fold. $

The output X is a single (n_patients, 3 * G) matrix with a fixed block layout
[ RNA | CNV | methy ], each block ordered identically by `gene_names`. The
preprocess transformer is told G and gene_names so it can subset/scale blocks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MODALITIES = ("rna", "cnv", "methy")


@dataclass
class OmicsData:
    """Container for aligned, RAW (un-leaked) multi-omics data."""
    X: np.ndarray                  # (n_patients, 3 * G), blocks [rna|cnv|methy]
    y: np.ndarray                  # (n_patients,) integer-encoded labels
    patient_ids: List[str]         # length n_patients, row order
    gene_names: List[str]          # length G, column order within each block
    label_map: Dict[str, int]      # subtype string -> int
    inverse_label_map: Dict[int, str] = field(default_factory=dict)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    def block_slice(self, modality: str) -> slice:
        """Column slice for a modality block inside X."""
        g = self.n_genes
        offset = {"rna": 0, "cnv": g, "methy": 2 * g}[modality]
        return slice(offset, offset + g)

    def class_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.y, minlength=len(self.label_map))
        return {self.inverse_label_map[i]: int(counts[i]) for i in range(len(counts))}


def _load_omics_raw(filepath: str) -> pd.DataFrame:
    """Load a raw omics CSV (genes x patients). Mirrors the dedup/NaN-index
    handling in the original preprocess.py so behaviour is consistent."""
    df = pd.read_csv(filepath, index_col=0)
    df = df[df.index.notnull()]
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        print(f"  -> {os.path.basename(filepath)}: aggregating {n_dupes} "
              f"duplicate gene rows via median.")
        df = df.groupby(level=0).median()
    return df


def load_omics(
    raw_dir: str,
    rna_file: str = "data_rna_seq_v2_rsem.csv",
    cnv_file: str = "data_cnv.csv",
    methy_file: str = "data_methylation_M.csv",
    clin_file: str = "data_clinical.csv",
    label_col: str = "SUBTYPE",
) -> OmicsData:
    """Load and align raw multi-omics + clinical labels at the patient level.

    Returns an OmicsData with RAW values. No fold-dependent statistic is applied.
    """
    rna = _load_omics_raw(os.path.join(raw_dir, rna_file))
    cnv = _load_omics_raw(os.path.join(raw_dir, cnv_file))
    methy = _load_omics_raw(os.path.join(raw_dir, methy_file))
    clin = pd.read_csv(os.path.join(raw_dir, clin_file), index_col=0)

    # Patients that have a label AND data in all three modalities.
    clin_idx = clin.dropna(subset=[label_col]).index
    common_patients = sorted(
        set(clin_idx) & set(rna.columns) & set(cnv.columns) & set(methy.columns)
    )
    if len(common_patients) == 0:
        raise ValueError("No patients shared across clinical + all 3 omics. "
                         "Check that omics columns are patient barcodes.")

    # Genes shared across all three modalities (consensus universe).
    common_genes = sorted(set(rna.index) & set(cnv.index) & set(methy.index))
    if len(common_genes) == 0:
        raise ValueError("No genes shared across the three omics matrices.")

    clin = clin.loc[common_patients]
    rna = rna.loc[common_genes, common_patients]
    cnv = cnv.loc[common_genes, common_patients]
    methy = methy.loc[common_genes, common_patients]

    # --- Patient-level dedup SAFETY ASSERTION (user: one sample per patient) ---
    if len(set(common_patients)) != len(common_patients):
        raise AssertionError("Duplicate patient IDs detected after alignment.")

    print(f"Aligned: {len(common_patients)} patients x {len(common_genes)} "
          f"common genes across {len(MODALITIES)} modalities.")

    # Transpose each to (patients x genes) and concatenate to [rna|cnv|methy].
    rna_t = rna.T.to_numpy(dtype=np.float64)
    cnv_t = cnv.T.to_numpy(dtype=np.float64)
    methy_t = methy.T.to_numpy(dtype=np.float64)
    X = np.hstack([rna_t, cnv_t, methy_t])

    # Label encoding: sorted-unique for a stable, reproducible mapping.
    subtypes = clin[label_col].astype(str).to_numpy()
    classes = sorted(np.unique(subtypes).tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    inverse = {i: c for c, i in label_map.items()}
    y = np.asarray([label_map[s] for s in subtypes], dtype=np.int64)

    print(f"Classes ({len(label_map)}): {label_map}")

    return OmicsData(
        X=X, y=y,
        patient_ids=list(common_patients),
        gene_names=list(common_genes),
        label_map=label_map,
        inverse_label_map=inverse,
    )


def load_curated_genes(path: Optional[str], gene_universe: List[str]) -> List[str]:
    """Read a one-gene-per-line curated file and keep only genes present in the
    aligned universe. Returns [] if path is None/missing."""
    if not path or not os.path.exists(path):
        if path:
            print(f"  -> curated gene file not found at {path}; none force-included.")
        return []
    with open(path) as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    present = [g for g in raw if g in set(gene_universe)]
    missing = len(raw) - len(present)
    print(f"  -> curated genes: {len(present)} present in universe "
          f"({missing} not found and dropped).")
    return present