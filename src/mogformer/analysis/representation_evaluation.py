"""Measure what a frozen patient representation knows.

The yardstick for every change to the self-supervised encoder. A
representation is any patients x dimensions matrix: the encoder's embedding,
or a comparator built from the same inputs (PCA of the concatenated omics, PCA
per omics, the raw inputs). Each is scored the same way:

* **linear probes** — cross-validated logistic or ridge read-outs of known
  labels, so "knows PAM50" means "PAM50 is linearly decodable";
* **effective dimension** — the participation ratio of its covariance, which
  falls towards 1 as a single axis takes over;
* **dominant-axis share** — how much of it a single covariate (copy-number
  burden, ploidy) explains;
* **clustering checkpoint** — consensus stability on fixed populations, and
  whether the split is a known one (burden or ploidy).

Probes use a repeated stratified partition drawn per target, and report the
Nadeau–Bengio corrected interval like every other comparison in the project.
No survival information is read here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, f1_score, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mogformer.analysis.clustering import run_consensus_clustering
from mogformer.evaluation.metrics import aggregate_across_folds
from mogformer.evaluation.stats import corrected_resampled_ttest
from mogformer.training.losses import participation_ratio

logger = logging.getLogger(__name__)

#: Folds and repeats of every probe partition, as in the supervised benchmark.
N_SPLITS = 5
N_REPEATS = 5

#: Ridge penalties searched inside each training fold for continuous targets.
RIDGE_ALPHAS = np.logspace(-2, 4, 13)

#: Proliferation genes of the PAM50 risk-of-recurrence proliferation score
#: (Nielsen et al. 2010), in current symbols.
PROLIFERATION_GENES: tuple[str, ...] = (
    "BIRC5",
    "CCNB1",
    "CDC20",
    "NUF2",
    "CEP55",
    "NDC80",
    "MKI67",
    "PTTG1",
    "RRM2",
    "TYMS",
    "UBE2C",
)


@dataclass(frozen=True)
class ProbeTarget:
    """One label a representation is probed for.

    Attributes:
        name: Display name.
        values: One value per patient, aligned with the representations;
            missing values are left out of that target's probe.
        task: ``"classification"`` (scored by macro-F1) or ``"regression"``
            (scored by R²).
        nuisance: True for labels a good representation should *not* carry,
            such as the tissue source site.
    """

    name: str
    values: pd.Series
    task: str
    nuisance: bool = False


#: Most factors MOFA+ is fitted with: it switches off factors it cannot
#: support, and its run time grows with the factor count.
MOFA_MAX_FACTORS = 128


def mofa_factors(inputs: np.ndarray, n_factors: int, seed: int = 42) -> np.ndarray:
    """Fit MOFA+ with one view per omics and return the patient factors.

    Args:
        inputs: Standardised inputs, patients x genes x omics.
        n_factors: Factors requested (capped at :data:`MOFA_MAX_FACTORS`).
        seed: Seed of the variational fit.

    Returns:
        Factor values, patients x factors.
    """
    import contextlib
    import io

    from mofapy2.run.entry_point import entry_point

    # mofapy2 reports progress on stdout; the project logs instead.
    with contextlib.redirect_stdout(io.StringIO()):
        model = entry_point()
        model.set_data_options(scale_views=False)
        model.set_data_matrix(
            [[inputs[:, :, omics]] for omics in range(inputs.shape[2])],
            likelihoods=["gaussian"] * inputs.shape[2],
        )
        model.set_model_options(factors=min(n_factors, MOFA_MAX_FACTORS))
        model.set_train_options(
            iter=1000, convergence_mode="fast", seed=seed, verbose=False, dropR2=None
        )
        model.build()
        model.run()
    factors = np.asarray(model.model.nodes["Z"].getExpectation())
    logger.info("MOFA+: %d factors fitted", factors.shape[1])
    return factors


def comparator_representations(
    inputs: np.ndarray, n_components: int, seed: int = 42, with_mofa: bool = False
) -> dict[str, np.ndarray]:
    """Build the comparators the encoder's embedding must beat.

    Like the encoder, they are fitted on every patient without any label, so
    probing them is as fair as probing the embedding.

    Args:
        inputs: The encoder's standardised inputs, patients x genes x omics.
        n_components: Dimension to match, usually the embedding's width.
        seed: Seed for the randomised PCA solver and MOFA+.
        with_mofa: Add ``mofa`` (MOFA+ factors, at most
            :data:`MOFA_MAX_FACTORS`); minutes rather than seconds.

    Returns:
        ``pca_concatenated`` (PCA of all omics together), ``pca_per_omics``
        (an equal share of components per omics, concatenated) and
        ``raw_inputs`` (every input value, an upper reference), and
        ``mofa`` when asked for.
    """
    n_patients, _, n_omics = inputs.shape
    flat = inputs.reshape(n_patients, -1)
    components = min(n_components, n_patients - 1, flat.shape[1])
    per_omics = max(1, components // n_omics)
    blocks = [
        PCA(n_components=per_omics, random_state=seed).fit_transform(
            inputs[:, :, omics]
        )
        for omics in range(n_omics)
    ]
    comparators = {
        "pca_concatenated": PCA(
            n_components=components, random_state=seed
        ).fit_transform(flat),
        "pca_per_omics": np.hstack(blocks),
        "raw_inputs": flat,
    }
    if with_mofa:
        comparators["mofa"] = mofa_factors(inputs, components, seed=seed)
    return comparators


def linear_probe_scores(
    features: np.ndarray,
    target: ProbeTarget,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = 42,
) -> np.ndarray:
    """Score a linear read-out of one label over a repeated partition.

    Scaling and the read-out are fitted on each training fold only. Classes
    with fewer members than folds are left out, since they cannot be split.

    Args:
        features: Patients x dimensions, aligned with ``target.values``.
        target: The label and how to score it.
        n_splits: Folds per repeat.
        n_repeats: Repeats.
        seed: Seed of the partition, shared by every representation so that
            their scores are paired.

    Returns:
        One score per fold (macro-F1 or R²), ordered by repeat then fold.
    """
    values = target.values.to_numpy()
    observed = pd.notna(target.values).to_numpy()
    if target.task == "classification":
        labels = pd.Series(values[observed]).astype(str)
        counts = labels.value_counts()
        keep_classes = set(counts[counts >= n_splits].index)
        rows = np.flatnonzero(observed)[labels.isin(keep_classes).to_numpy()]
        y = pd.Series(values[rows]).astype(str).to_numpy()
        splitter: RepeatedStratifiedKFold | RepeatedKFold = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
    else:
        rows = np.flatnonzero(observed)
        y = values[rows].astype(float)
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
    x = features[rows]

    scores = []
    for train, test in splitter.split(x, y):
        if target.task == "classification":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            ).fit(x[train], y[train])
            scores.append(f1_score(y[test], model.predict(x[test]), average="macro"))
        else:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)).fit(
                x[train], y[train]
            )
            scores.append(r2_score(y[test], model.predict(x[test])))
    return np.asarray(scores)


def probe_panel(
    representations: Mapping[str, np.ndarray],
    targets: Sequence[ProbeTarget],
    seed: int = 42,
) -> pd.DataFrame:
    """Probe every representation for every target on shared partitions.

    Args:
        representations: Name to patients x dimensions, rows aligned.
        targets: The labels to probe for.
        seed: Seed of the partitions.

    Returns:
        Long table with ``representation``, ``target``, ``task``,
        ``nuisance``, ``estimate`` and ``score``.
    """
    rows: list[dict[str, object]] = []
    for target in targets:
        for name, features in representations.items():
            scores = linear_probe_scores(features, target, seed=seed)
            logger.info(
                "probe %-22s %-18s %.3f", name, target.name, float(np.mean(scores))
            )
            rows.extend(
                {
                    "representation": name,
                    "target": target.name,
                    "task": target.task,
                    "nuisance": target.nuisance,
                    "estimate": index,
                    "score": float(score),
                }
                for index, score in enumerate(scores)
            )
    return pd.DataFrame(rows)


def summarise_probe_panel(per_fold: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Summarise each probe and compare it with the reference representation.

    Args:
        per_fold: Output of :func:`probe_panel`.
        reference: Representation every other one is compared against,
            usually the encoder's embedding.

    Returns:
        One row per representation and target: mean, corrected interval, and
        the paired difference to the reference with its corrected p-value.
    """
    rows = []
    for (target, name), group in per_fold.groupby(["target", "representation"]):
        scores = group.sort_values("estimate")["score"].to_numpy()
        summary = aggregate_across_folds(scores, n_splits=N_SPLITS)
        base = (
            per_fold[
                (per_fold["target"] == target)
                & (per_fold["representation"] == reference)
            ]
            .sort_values("estimate")["score"]
            .to_numpy()
        )
        test = (
            corrected_resampled_ttest(scores - base, n_splits=N_SPLITS)
            if name != reference and len(base) == len(scores)
            else {"mean_diff": float("nan"), "p_value": float("nan")}
        )
        rows.append(
            {
                "target": target,
                "representation": name,
                "task": group["task"].iloc[0],
                "nuisance": bool(group["nuisance"].iloc[0]),
                "mean": summary["mean"],
                "nb_ci95_lo": summary["nb_ci95_lo"],
                "nb_ci95_hi": summary["nb_ci95_hi"],
                f"minus_{reference}": test["mean_diff"],
                "corrected_p": test["p_value"],
            }
        )
    return pd.DataFrame(rows)


def dominant_axis_share(
    embedding: np.ndarray, covariates: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Measure how much of a representation one covariate explains.

    Args:
        embedding: Patients x dimensions.
        covariates: Name to one value per patient (NaN allowed).

    Returns:
        Per covariate: ``pc1_r2``, the squared correlation of the first
        principal component with the covariate; ``variance_share``, the
        fraction of the representation's total variance a linear function of
        the covariate explains; and the share of variance in the first
        principal component itself.
    """
    centred = embedding - embedding.mean(axis=0)
    pca = PCA(n_components=1).fit(centred)
    pc1 = pca.transform(centred)[:, 0]
    rows: list[dict[str, object]] = []
    for name, values in covariates.items():
        keep = ~np.isnan(values)
        x = values[keep].reshape(-1, 1)
        fitted = LinearRegression().fit(x, centred[keep]).predict(x)
        explained = ((fitted - centred[keep].mean(axis=0)) ** 2).sum()
        total = ((centred[keep] - centred[keep].mean(axis=0)) ** 2).sum()
        rows.append(
            {
                "covariate": name,
                "pc1_r2": float(np.corrcoef(pc1[keep], values[keep])[0, 1] ** 2),
                "variance_share": float(explained / total),
                "pc1_variance_share": float(pca.explained_variance_ratio_[0]),
            }
        )
    return pd.DataFrame(rows)


def clustering_checkpoint(
    embedding: np.ndarray,
    populations: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    n_clusters_list: Sequence[int] = (2, 3, 4, 5, 6),
    n_resample: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster fixed populations and compare each split with known ones.

    Structure only: no outcome is read. The question it answers is whether
    the most stable split of a population is simply the burden or ploidy
    split, which would veto a change however well it probes.

    Args:
        embedding: Patients x dimensions for the whole cohort.
        populations: Name to the row indices of a population.
        references: Name to a cohort-wide partition to compare against (for
            example the ploidy split); ``-1`` marks a patient without a value.
        n_clusters_list: Numbers of clusters tried.
        n_resample: Consensus resamples per number of clusters.
        seed: Seed of the resampling.

    Returns:
        One row per population and number of clusters: PAC, and the ARI with
        each reference partition on the patients it covers.
    """
    rows = []
    for population, indices in populations.items():
        values = embedding[indices]
        results = run_consensus_clustering(
            values, n_clusters_list=n_clusters_list, n_resample=n_resample, seed=seed
        )
        for n_clusters, result in results.items():
            row: dict[str, object] = {
                "population": population,
                "n_patients": len(indices),
                "k": n_clusters,
                "pac": result.pac,
            }
            for name, partition in references.items():
                labels = partition[indices]
                covered = labels >= 0
                row[f"ari_{name}"] = float(
                    adjusted_rand_score(labels[covered], result.labels[covered])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Return the linear centred kernel alignment of two representations.

    ``‖Yᵀ X‖²_F / (‖Xᵀ X‖_F ‖Yᵀ Y‖_F)`` on column-centred matrices (Kornblith
    et al. 2019). It is invariant to rotation and isotropic scaling, so two
    encoders trained from different seeds score 1 when they learned the same
    geometry, whatever basis their dimensions landed in.

    Args:
        first: Patients x dimensions.
        second: Patients x dimensions, same patients in the same order.

    Returns:
        The alignment, between 0 and 1.

    Raises:
        ValueError: If the two cover a different number of patients.
    """
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"representations cover {first.shape[0]} and {second.shape[0]} patients"
        )
    x = first - first.mean(axis=0)
    y = second - second.mean(axis=0)
    cross = np.linalg.norm(y.T @ x, "fro") ** 2
    scale = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(cross / scale)


def per_gene_rank_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the Spearman correlation of two measurements, gene by gene.

    Args:
        first: Patients x genes.
        second: Patients x genes, same genes in the same order.

    Returns:
        One coefficient per gene; NaN where either measurement is constant.
    """
    from scipy.stats import rankdata

    ranks_first = rankdata(first, axis=0)
    ranks_second = rankdata(second, axis=0)
    ranks_first -= ranks_first.mean(axis=0)
    ranks_second -= ranks_second.mean(axis=0)
    numerator = (ranks_first * ranks_second).sum(axis=0)
    denominator = np.sqrt((ranks_first**2).sum(axis=0) * (ranks_second**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


__all__ = [
    "MOFA_MAX_FACTORS",
    "N_REPEATS",
    "N_SPLITS",
    "PROLIFERATION_GENES",
    "ProbeTarget",
    "clustering_checkpoint",
    "comparator_representations",
    "dominant_axis_share",
    "linear_cka",
    "linear_probe_scores",
    "mofa_factors",
    "participation_ratio",
    "per_gene_rank_correlation",
    "probe_panel",
    "summarise_probe_panel",
]
