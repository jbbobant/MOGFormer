"""Repeated stratified cross-validation folds, persisted so every model shares them.

The evaluation protocol compares models with paired statistics, which is only
valid when every model is scored on identical partitions. Folds are therefore
built once, written to disk with both positional indices and patient barcodes,
and loaded by every run — baselines and transformer alike.

Barcodes are persisted alongside indices so a partition survives any later
reordering of the data matrix, and so a mismatch between a loaded fold and the
current cohort is detectable rather than silent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class FoldSpec:
    """One outer cross-validation fold.

    Attributes:
        repeat: Zero-based repeat index.
        fold: Zero-based fold index within the repeat.
        train_idx: Positional indices of the training patients.
        test_idx: Positional indices of the held-out patients.
    """

    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_folds(
    y: np.ndarray,
    patient_ids: Sequence[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> list[FoldSpec]:
    """Build repeated stratified patient-level folds.

    Five splits rather than ten keeps roughly seven Normal-like patients in each
    held-out fold, which is the smallest class in the breast cancer cohort.

    Args:
        y: Integer class labels, shape ``(n_patients,)``.
        patient_ids: Patient barcodes in the same order as ``y``.
        n_splits: Folds per repeat.
        n_repeats: Number of repeats; total estimates is the product.
        seed: Seed for the repeated splitter.

    Returns:
        ``n_splits * n_repeats`` folds in repeat-major order.

    Raises:
        ValueError: If ``y`` and ``patient_ids`` differ in length.
    """
    if len(y) != len(patient_ids):
        raise ValueError(
            f"y has {len(y)} entries but patient_ids has {len(patient_ids)}"
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    return [
        FoldSpec(
            repeat=i // n_splits,
            fold=i % n_splits,
            train_idx=np.asarray(train, dtype=int),
            test_idx=np.asarray(test, dtype=int),
        )
        for i, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y))
    ]


def assert_no_leakage(folds: Sequence[FoldSpec], patient_ids: Sequence[str]) -> None:
    """Verify that no patient appears in both halves of any fold.

    Checked by barcode rather than by index so the guarantee still holds if a
    cohort ever carries more than one sample per patient.

    Args:
        folds: Folds to check.
        patient_ids: Patient barcodes indexed by the folds.

    Raises:
        ValueError: If any patient appears in both train and test of a fold.
    """
    barcodes = np.asarray(patient_ids)
    for spec in folds:
        overlap = set(barcodes[spec.train_idx].tolist()) & set(
            barcodes[spec.test_idx].tolist()
        )
        if overlap:
            raise ValueError(
                f"Leakage: {len(overlap)} patient(s) in both train and test of "
                f"repeat {spec.repeat} fold {spec.fold}: "
                f"{sorted(overlap)[:5]}"
            )


def save_folds(
    folds: Sequence[FoldSpec], patient_ids: Sequence[str], path: str | Path
) -> None:
    """Write folds to JSON, recording both indices and patient barcodes.

    Args:
        folds: Folds to persist.
        patient_ids: Patient barcodes in the order the indices refer to.
        path: Destination file; parent directories are created as needed.
    """
    barcodes = np.asarray(patient_ids)
    payload = {
        "n_estimates": len(folds),
        "patient_order": list(patient_ids),
        "folds": [
            {
                "repeat": spec.repeat,
                "fold": spec.fold,
                "train_idx": spec.train_idx.tolist(),
                "test_idx": spec.test_idx.tolist(),
                "train_patients": barcodes[spec.train_idx].tolist(),
                "test_patients": barcodes[spec.test_idx].tolist(),
            }
            for spec in folds
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %d folds to %s", len(folds), destination)


def load_folds(path: str | Path) -> tuple[list[FoldSpec], list[str]]:
    """Read folds previously written by :func:`save_folds`.

    Args:
        path: JSON file to read.

    Returns:
        Tuple of the folds and the patient order the indices refer to.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = [
        FoldSpec(
            repeat=entry["repeat"],
            fold=entry["fold"],
            train_idx=np.asarray(entry["train_idx"], dtype=int),
            test_idx=np.asarray(entry["test_idx"], dtype=int),
        )
        for entry in payload["folds"]
    ]
    return folds, list(payload["patient_order"])


def assert_folds_match(
    loaded_patient_order: Sequence[str], current_patient_ids: Sequence[str]
) -> None:
    """Verify that loaded folds were built over the current cohort.

    This is the guard that would have caught the 914-versus-949 divergence:
    positional indices are meaningless against a different patient order, and
    without this check the mismatch produces plausible-looking but incomparable
    scores.

    Args:
        loaded_patient_order: Patient order recorded in the fold file.
        current_patient_ids: Patient order of the dataset about to be scored.

    Raises:
        ValueError: If the two orders differ, with a diagnosis of how.
    """
    if list(loaded_patient_order) == list(current_patient_ids):
        return

    loaded, current = set(loaded_patient_order), set(current_patient_ids)
    if loaded == current:
        raise ValueError(
            "Fold file covers the same patients in a different order; "
            "positional indices would be silently wrong."
        )
    raise ValueError(
        f"Fold file covers {len(loaded)} patients but the dataset has "
        f"{len(current)}. Only in folds: {sorted(loaded - current)[:5]}; "
        f"only in dataset: {sorted(current - loaded)[:5]}."
    )


def remap_fold_indices(
    folds: Sequence[FoldSpec], keep_mask: np.ndarray
) -> list[FoldSpec]:
    """Rewrite fold indices after patients have been dropped from a cohort.

    Patients that survive keep their fold membership, so a reduced-class run is
    a controlled comparison against the full run rather than an independent
    experiment on a freshly stratified partition. Regenerating folds over the
    smaller cohort instead is what made the four-class baselines unpairable with
    the five-class transformer results.

    Args:
        folds: Folds indexed against the original cohort.
        keep_mask: Boolean mask over the original patients, True where kept, as
            returned by :func:`mogformer.data.omics.exclude_classes`.

    Returns:
        Folds indexed against the compacted cohort.

    Raises:
        ValueError: If ``keep_mask`` is not a boolean array.
    """
    if keep_mask.dtype != bool:
        raise ValueError(f"keep_mask must be boolean, got {keep_mask.dtype}")

    remapped_positions = -np.ones(len(keep_mask), dtype=int)
    remapped_positions[np.where(keep_mask)[0]] = np.arange(int(keep_mask.sum()))

    return [
        FoldSpec(
            repeat=spec.repeat,
            fold=spec.fold,
            train_idx=remapped_positions[spec.train_idx[keep_mask[spec.train_idx]]],
            test_idx=remapped_positions[spec.test_idx[keep_mask[spec.test_idx]]],
        )
        for spec in folds
    ]


def select_folds(
    folds: Sequence[FoldSpec],
    n_repeats: int | None = None,
    max_folds: int | None = None,
) -> list[FoldSpec]:
    """Take a prefix of the fold list, for cheap partial runs.

    Any result produced from a subset must report its own estimate count: a
    single-repeat run yields five estimates, not twenty-five, and its confidence
    interval is correspondingly wider.

    Args:
        folds: Full fold list.
        n_repeats: Keep only folds from the first this-many repeats. None keeps
            every repeat.
        max_folds: Keep only the first this-many folds within each repeat. None
            keeps every fold.

    Returns:
        The selected folds, in their original order.
    """
    selected = list(folds)
    if n_repeats is not None:
        selected = [spec for spec in selected if spec.repeat < n_repeats]
    if max_folds is not None:
        selected = [spec for spec in selected if spec.fold < max_folds]
    return selected
