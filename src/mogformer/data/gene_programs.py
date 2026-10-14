"""Gene programs: a partition of the gene axis for program-scoped read-outs.

The default encoder's summary uses about seven directions, and every gene's
state shares them: proliferation, ER signalling and copy-number events are
superposed in the same few coordinates. Program-scoped read-outs give each
group of co-varying genes a token of its own, which alone reads and alone
reconstructs those genes. Two tokens can then only be redundant where their
programs truly co-vary across patients.

Programs are fitted on the fitting patients only (the same rows that fit the
gene axis and the scalers), so nothing about held-out patients enters them.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

#: Sources of gene programs accepted by the encoder configuration.
PROGRAM_SOURCES: tuple[str, ...] = ("coexpression",)


def fit_coexpression_programs(inputs: np.ndarray, n_programs: int) -> np.ndarray:
    """Partition genes into programs of co-varying standardised profiles.

    Each gene's profile is its three omics across the fitting patients,
    concatenated and z-scored. Ward linkage on these profiles is clustering by
    correlation (the squared Euclidean distance between z-scored profiles is
    ``2m(1 − r)``), and unlike average linkage on ``|r|`` it does not leave
    singletons. Anti-correlated genes land in different programs.

    Args:
        inputs: Standardised inputs of the fitting patients, shape
            ``(n_patients, n_genes, 3)``.
        n_programs: Number of programs ``K``.

    Returns:
        Program index per gene, shape ``(n_genes,)``, in ``[0, K)``. Programs
        are numbered by decreasing size, so the numbering does not depend on
        the linkage's internal order.

    Raises:
        ValueError: If ``n_programs`` is not between 1 and the number of genes.
    """
    n_genes = inputs.shape[1]
    if not 1 <= n_programs <= n_genes:
        raise ValueError(f"n_programs must be in [1, {n_genes}], got {n_programs}")
    profiles = np.concatenate([inputs[:, :, k].T for k in range(inputs.shape[2])], 1)
    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    profiles = profiles / (profiles.std(axis=1, keepdims=True) + 1e-8)
    if n_programs == 1:
        return np.zeros(n_genes, dtype=np.int64)
    labels = fcluster(linkage(profiles, "ward"), n_programs, "maxclust") - 1
    order = np.argsort(-np.bincount(labels, minlength=n_programs), kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    return renumber[labels].astype(np.int64)
