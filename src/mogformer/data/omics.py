"""Raw multi-omics loading, patient alignment and label encoding.

This module deliberately returns **raw** values. Every statistic that must be
fit per fold — variable-gene selection, imputation, log transform, scaling —
lives in :mod:`mogformer.data.preprocess` as a scikit-learn transformer so it
refits inside each fold. Loading and preprocessing are kept apart because
mixing them is the classic leakage path.

Cohort composition is configuration, never a literal. A loader that hardcodes
``!= "BRCA_Normal"`` silently produces a 914-patient cohort while another path
reads 949, and the two are never scored on comparable folds. Use ``exclude``
and record the value.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical modality order. Every ``X`` matrix in the project is laid out as
#: three horizontally stacked blocks in this order, so index arithmetic
#: everywhere depends on it. Defined once here and imported elsewhere.
MODALITY_ORDER: tuple[str, ...] = ("rna", "cnv", "methy")


@dataclass
class OmicsData:
    """Aligned, raw multi-omics matrices with encoded labels.

    Attributes:
        X: Feature matrix of shape ``(n_patients, 3 * n_genes)``, laid out as
            three blocks in :data:`MODALITY_ORDER`, each block sharing the same
            gene ordering.
        y: Integer-encoded labels, shape ``(n_patients,)``.
        patient_ids: Patient barcodes in row order of ``X``.
        gene_names: Gene symbols in column order within each block.
        label_map: Class name to integer code.
        inverse_label_map: Integer code to class name.
    """

    X: np.ndarray
    y: np.ndarray
    patient_ids: list[str]
    gene_names: list[str]
    label_map: dict[str, int]
    inverse_label_map: dict[int, str] = field(default_factory=dict)

    @property
    def n_genes(self) -> int:
        """Return the number of genes per modality block."""
        return len(self.gene_names)

    @property
    def n_patients(self) -> int:
        """Return the number of patients."""
        return len(self.patient_ids)

    def block_slice(self, modality: str) -> slice:
        """Return the column slice of ``X`` holding one modality.

        Args:
            modality: One of :data:`MODALITY_ORDER`.

        Returns:
            Slice selecting that modality's block from ``X``.

        Raises:
            KeyError: If ``modality`` is not a known modality.
        """
        if modality not in MODALITY_ORDER:
            raise KeyError(
                f"unknown modality {modality!r}; expected one of {list(MODALITY_ORDER)}"
            )
        offset = MODALITY_ORDER.index(modality) * self.n_genes
        return slice(offset, offset + self.n_genes)

    def class_counts(self) -> dict[str, int]:
        """Return the number of patients per class, keyed by class name."""
        counts = np.bincount(self.y, minlength=len(self.label_map))
        return {self.inverse_label_map[i]: int(counts[i]) for i in range(len(counts))}


#: Clipping bound for beta values before the logit, so a fully methylated or
#: unmethylated gene maps to a finite M-value (about ±10) instead of infinity.
M_VALUE_EPS = 1e-3


def beta_to_m_values(beta: pd.DataFrame, eps: float = M_VALUE_EPS) -> pd.DataFrame:
    """Convert methylation beta values to M-values, ``log2(β / (1 − β))``.

    Beta values are bounded proportions whose variance shrinks towards 0 and 1;
    the logit makes them approximately homoscedastic, which suits the linear
    and distance-based models downstream. It is a per-value transform with no
    fitted statistic, so applying it before the fold split leaks nothing.

    Args:
        beta: Beta values in ``[0, 1]``, any shape; missing values stay missing.
        eps: Values are clipped to ``[eps, 1 − eps]`` first.

    Returns:
        M-values, same index and columns.
    """
    clipped = beta.clip(eps, 1.0 - eps)
    return (clipped / (1.0 - clipped)).apply(np.log2)


#: Scales copy number can be presented to a model on. ``absolute`` leaves the
#: values as loaded (absolute copies, GISTIC calls or ordinal codes);
#: ``log_ratio`` is ``log2(CN / 2)``, which keeps absolute dosage and therefore
#: ploidy; ``ploidy_relative`` is ``log2(CN / ploidy)``, blind to a uniform
#: genome doubling; ``ordinal`` maps ordinal codes 0–4 onto −2…2.
CNV_REPRESENTATIONS: tuple[str, ...] = (
    "absolute",
    "log_ratio",
    "ploidy_relative",
    "ordinal",
)

#: Copies a homozygous deletion is floored at before the logarithm, so that it
#: maps to the bottom of the bounded scale rather than to minus infinity.
COPY_NUMBER_FLOOR = 0.5

#: Bound of the log-ratio scales: ±2 is a quarter or four times the reference.
COPY_NUMBER_LOG_BOUND = 2.0

#: Ordinal code of a copy-neutral gene in a 0–4 (deep loss … amplification)
#: coding, such as METABRIC's.
ORDINAL_NEUTRAL_CODE = 2.0

#: Label given to patients without a class label when a cohort is loaded for
#: label-free use (self-supervision), so that they are kept and identifiable.
UNLABELLED = "unlabelled"


def ploidy_estimate(cnv_absolute: pd.DataFrame) -> pd.Series:
    """Estimate each patient's ploidy as the mean absolute copy number.

    Ploidy is a property of the genome, so the mean is taken over every gene of
    the absolute copy-number matrix, never over a selected subset: a fold's
    gene selection must not change it. The estimate is weighted by gene density
    rather than genome length, which is an approximation, but it is computed
    per patient from that patient's values alone, so it fits nothing across
    patients and leaks nothing.

    Args:
        cnv_absolute: Absolute copy number, genes by patients. Missing values
            are ignored.

    Returns:
        Ploidy per patient, indexed by the matrix's columns.

    Raises:
        ValueError: If a copy number is negative.
    """
    if (cnv_absolute < 0).any().any():
        raise ValueError("absolute copy number must be non-negative")
    return cnv_absolute.mean(axis=0, skipna=True).rename("ploidy")


def relative_copy_number(
    cnv_absolute: pd.DataFrame,
    ploidy: pd.Series,
    floor: float = COPY_NUMBER_FLOOR,
    bound: float = COPY_NUMBER_LOG_BOUND,
) -> pd.DataFrame:
    """Express copy number relative to each patient's own ploidy.

    Computes ``clip(log2(max(CN, floor) / ploidy), -bound, bound)``. A genome
    doubled uniformly gives the same profile as the undoubled one, so whole-
    genome doubling no longer dominates the inputs; focal gains and losses
    relative to the tumour's own baseline remain.

    Args:
        cnv_absolute: Absolute copy number, genes by patients.
        ploidy: Ploidy per patient, indexed like ``cnv_absolute``'s columns.
        floor: Copies a homozygous deletion is raised to before the logarithm.
        bound: Symmetric clip of the log ratio; ``2`` bounds high-level
            amplifications at four times the patient's ploidy.

    Returns:
        Bounded log ratios, same shape and labels as ``cnv_absolute``; missing
        values stay missing.

    Raises:
        ValueError: If a patient has no ploidy estimate or a non-positive one.
    """
    reference = ploidy.reindex(cnv_absolute.columns)
    if reference.isna().any() or (reference <= 0).any():
        missing = list(reference[reference.isna() | (reference <= 0)].index[:5])
        raise ValueError(f"no positive ploidy estimate for patients {missing}")
    ratio = (cnv_absolute.clip(lower=floor) / reference).apply(np.log2)
    return ratio.clip(-bound, bound)


def load_ploidy(path: str | Path) -> pd.Series:
    """Read a per-patient ploidy table written beside a cohort archive.

    Args:
        path: CSV with ``patient_id`` and ``ploidy`` columns.

    Returns:
        Ploidy per patient, indexed by patient barcode.
    """
    table = pd.read_csv(path, index_col="patient_id")
    return table["ploidy"].astype(float)


def apply_copy_number_representation(
    data: OmicsData,
    representation: str,
    ploidy: pd.Series | None = None,
) -> OmicsData:
    """Rewrite a cohort's copy-number block on the requested scale.

    Each scale is a per-value or per-patient transform with no statistic fitted
    across patients, so it is applied before any fold split without leakage.

    Args:
        data: Cohort whose copy-number block is absolute copies (for
            ``log_ratio`` and ``ploidy_relative``) or ordinal codes 0–4 (for
            ``ordinal``).
        representation: One of :data:`CNV_REPRESENTATIONS`.
        ploidy: Per-patient ploidy, required by ``ploidy_relative``; estimated
            over the full absolute copy-number matrix, not this cohort's
            aligned genes.

    Returns:
        A cohort sharing everything with ``data`` except the copy-number block.

    Raises:
        ValueError: If ``representation`` is unknown, or ``ploidy_relative`` is
            asked for without a ploidy estimate.
    """
    if representation not in CNV_REPRESENTATIONS:
        raise ValueError(
            f"unknown copy-number representation {representation!r}; "
            f"available: {list(CNV_REPRESENTATIONS)}"
        )
    if representation == "absolute":
        return data

    block = data.block_slice("cnv")
    cnv = pd.DataFrame(
        data.X[:, block].T.astype(np.float64),
        index=data.gene_names,
        columns=data.patient_ids,
    )
    if representation == "ploidy_relative":
        if ploidy is None:
            raise ValueError("ploidy_relative copy number needs a ploidy estimate")
        rewritten = relative_copy_number(cnv, ploidy)
    elif representation == "log_ratio":
        rewritten = relative_copy_number(
            cnv, pd.Series(2.0, index=cnv.columns), floor=COPY_NUMBER_FLOOR
        )
    else:
        rewritten = cnv - ORDINAL_NEUTRAL_CODE

    features = data.X.copy()
    features[:, block] = rewritten.to_numpy().T.astype(features.dtype)
    logger.info("copy number rewritten on the %s scale", representation)
    return replace(data, X=features)


def _load_omics_raw(filepath: Path) -> pd.DataFrame:
    """Read one genes-by-patients matrix, collapsing duplicate gene rows.

    Rows with a null index are dropped and duplicated gene symbols are
    aggregated by median, which is how the original pipeline behaved.

    Args:
        filepath: Path to a CSV whose first column holds gene symbols.

    Returns:
        Data frame indexed by gene symbol with patients as columns.
    """
    frame = pd.read_csv(filepath, index_col=0)
    frame = frame[frame.index.notna()]
    if frame.index.duplicated().any():
        n_duplicates = int(frame.index.duplicated().sum())
        logger.info(
            "%s: aggregating %d duplicate gene rows by median",
            filepath.name,
            n_duplicates,
        )
        frame = frame.groupby(level=0).median()
    return frame


def load_omics(
    raw_dir: str | Path,
    rna_file: str,
    cnv_file: str,
    methy_file: str,
    clin_file: str,
    label_col: str,
    exclude: Iterable[str] = (),
    keep_unlabelled: bool = False,
) -> OmicsData:
    """Load and align raw multi-omics matrices with clinical labels.

    Patients are kept when they carry a label and appear in all three omics
    matrices; genes are kept when they appear in all three. No fold-dependent
    statistic is applied.

    File names, the label column and the excluded classes are all required or
    explicit rather than defaulted, because a silently defaulted cohort is how
    two models end up scored on different patients.

    Args:
        raw_dir: Directory holding the four input files.
        rna_file: RNA matrix filename, genes by patients.
        cnv_file: Copy-number matrix filename, genes by patients.
        methy_file: Methylation matrix filename, genes by patients.
        clin_file: Clinical table filename, indexed by patient barcode.
        label_col: Column of the clinical table holding the class label.
        exclude: Class names to drop before encoding. Empty by default, which
            keeps every labelled patient.
        keep_unlabelled: Keep patients without a label (or absent from the
            clinical table) under the class :data:`UNLABELLED`, for label-free
            uses such as self-supervision, whose cohort is defined by the
            omics measured and not by the label.

    Returns:
        Aligned raw data with contiguous integer labels.

    Raises:
        ValueError: If no patient or no gene is shared across all inputs, or if
            excluding classes would leave fewer than two classes.
        AssertionError: If duplicate patient barcodes survive alignment.
    """
    root = Path(raw_dir)
    rna = _load_omics_raw(root / rna_file)
    cnv = _load_omics_raw(root / cnv_file)
    methy = _load_omics_raw(root / methy_file)
    clinical = pd.read_csv(root / clin_file, index_col=0)

    if keep_unlabelled:
        measured = sorted(set(rna.columns) & set(cnv.columns) & set(methy.columns))
        clinical = clinical[~clinical.index.duplicated()].reindex(measured)
        clinical[label_col] = clinical[label_col].fillna(UNLABELLED)
    labelled = clinical.dropna(subset=[label_col])
    excluded = set(exclude)
    if excluded:
        before = len(labelled)
        labelled = labelled[~labelled[label_col].astype(str).isin(excluded)]
        logger.info(
            "excluded %d patient(s) in classes %s",
            before - len(labelled),
            sorted(excluded),
        )

    common_patients = sorted(
        set(labelled.index) & set(rna.columns) & set(cnv.columns) & set(methy.columns)
    )
    if not common_patients:
        raise ValueError(
            "No patients shared across the clinical table and all three omics "
            "matrices. Check that omics columns are patient barcodes."
        )

    common_genes = sorted(set(rna.index) & set(cnv.index) & set(methy.index))
    if not common_genes:
        raise ValueError("No genes shared across the three omics matrices.")

    clinical = labelled.loc[common_patients]
    rna = rna.loc[common_genes, common_patients]
    cnv = cnv.loc[common_genes, common_patients]
    methy = methy.loc[common_genes, common_patients]

    if len(set(common_patients)) != len(common_patients):
        raise AssertionError("Duplicate patient barcodes detected after alignment.")

    logger.info(
        "aligned %d patients x %d common genes across %d modalities",
        len(common_patients),
        len(common_genes),
        len(MODALITY_ORDER),
    )

    blocks = [frame.T.to_numpy(dtype=np.float64) for frame in (rna, cnv, methy)]
    features = np.hstack(blocks)

    subtypes = clinical[label_col].astype(str).to_numpy()
    classes = sorted(np.unique(subtypes).tolist())
    if len(classes) < 2:
        raise ValueError(
            f"Only {len(classes)} class remains after exclusion {sorted(excluded)}; "
            "at least two are required."
        )
    label_map = {name: code for code, name in enumerate(classes)}
    inverse_label_map = {code: name for name, code in label_map.items()}
    labels = np.asarray([label_map[name] for name in subtypes], dtype=np.int64)

    logger.info("classes (%d): %s", len(label_map), label_map)

    return OmicsData(
        X=features,
        y=labels,
        patient_ids=list(common_patients),
        gene_names=list(common_genes),
        label_map=label_map,
        inverse_label_map=inverse_label_map,
    )


def load_curated_genes(
    path: str | Path | None, gene_universe: Sequence[str]
) -> list[str]:
    """Read a curated gene list and keep the genes present in the universe.

    Curated genes are force-included by the preprocessing transformer after
    variable-gene ranking, guaranteeing that canonical markers survive every
    fold regardless of their variance.

    Args:
        path: One-symbol-per-line text file, or None to force-include nothing.
        gene_universe: Gene symbols available after alignment.

    Returns:
        Curated symbols present in ``gene_universe``, in file order. Empty when
        ``path`` is None or does not exist.
    """
    if path is None:
        return []
    curated_path = Path(path)
    if not curated_path.exists():
        logger.warning(
            "curated gene file not found at %s; none force-included", curated_path
        )
        return []

    requested = [
        line.strip()
        for line in curated_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe = set(gene_universe)
    present = [gene for gene in requested if gene in universe]
    logger.info(
        "curated genes: %d of %d present in the universe",
        len(present),
        len(requested),
    )
    return present


def exclude_classes(
    data: OmicsData, classes: Iterable[str]
) -> tuple[OmicsData, np.ndarray]:
    """Drop whole classes from a loaded dataset and re-encode labels.

    Returns the boolean keep mask alongside the filtered data so that existing
    folds can be remapped rather than regenerated — see
    :func:`mogformer.data.folds.remap_fold_indices`. Preserving fold membership
    is what makes a four-class run a controlled comparison against a five-class
    run instead of an independent experiment on a different partition.

    Args:
        data: Loaded dataset.
        classes: Class names to drop. Names absent from the dataset are ignored.

    Returns:
        Tuple of the filtered dataset and a boolean mask over the original
        patients, True where the patient was kept.

    Raises:
        ValueError: If fewer than two classes would remain.
    """
    excluded = set(classes)
    keep_mask = np.array(
        [data.inverse_label_map[int(code)] not in excluded for code in data.y],
        dtype=bool,
    )
    kept = np.where(keep_mask)[0]

    remaining = [
        data.inverse_label_map[code]
        for code in range(len(data.label_map))
        if data.inverse_label_map[code] not in excluded
    ]
    if len(remaining) < 2:
        raise ValueError(
            f"Excluding {sorted(excluded)} would leave {len(remaining)} class(es); "
            "at least two are required."
        )

    label_map = {name: code for code, name in enumerate(remaining)}
    inverse_label_map = {code: name for name, code in label_map.items()}
    labels = np.asarray(
        [label_map[data.inverse_label_map[int(code)]] for code in data.y[kept]],
        dtype=np.int64,
    )

    logger.info(
        "excluded %d patient(s) in %s; %d remain across %d classes",
        data.n_patients - len(kept),
        sorted(excluded),
        len(kept),
        len(label_map),
    )

    filtered = OmicsData(
        X=data.X[kept],
        y=labels,
        patient_ids=[data.patient_ids[i] for i in kept],
        gene_names=data.gene_names,
        label_map=label_map,
        inverse_label_map=inverse_label_map,
    )
    return filtered, keep_mask


def save_omics_bundle(data: OmicsData, path: str | Path) -> Path:
    """Serialise an aligned cohort to a single compressed archive.

    The three raw matrices together are roughly half a gigabyte of CSV, and
    parsing and aligning them dominates the start-up of every run. Alignment,
    however, is a cohort-level operation with no fold dependence, so the
    *result* of :func:`load_omics` can be cached without touching the leakage
    boundary: every statistic that must be fit per fold still is.

    Args:
        data: Cohort to serialise.
        path: Destination ``.npz`` file. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X=data.X.astype(np.float32),
        y=data.y.astype(np.int64),
        patient_ids=np.asarray(data.patient_ids, dtype=object),
        gene_names=np.asarray(data.gene_names, dtype=object),
        label_names=np.asarray(
            [name for name, _ in sorted(data.label_map.items(), key=lambda kv: kv[1])],
            dtype=object,
        ),
    )
    logger.info(
        "wrote cohort bundle %s (%d patients, %d genes, %.1f MB)",
        destination,
        data.n_patients,
        data.n_genes,
        destination.stat().st_size / 1e6,
    )
    return destination


def load_omics_bundle(path: str | Path) -> OmicsData:
    """Read a cohort written by :func:`save_omics_bundle`.

    Args:
        path: Archive to read.

    Returns:
        The cohort, identical to what :func:`load_omics` produced apart from
        ``X`` being stored at single precision.
    """
    with np.load(Path(path), allow_pickle=True) as archive:
        label_names = [str(name) for name in archive["label_names"]]
        label_map = {name: code for code, name in enumerate(label_names)}
        return OmicsData(
            X=archive["X"],
            y=archive["y"],
            patient_ids=[str(pid) for pid in archive["patient_ids"]],
            gene_names=[str(gene) for gene in archive["gene_names"]],
            label_map=label_map,
            inverse_label_map={code: name for name, code in label_map.items()},
        )
