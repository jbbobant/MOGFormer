"""Consensus clustering of the frozen embedding, and its confound screen.

A single run of k-means always returns clusters. The question is whether the
same patients group together when the cohort is resampled, so every partition
here is built by consensus: cluster many subsamples, count how often each pair of
patients lands together, and read the stability off that co-assignment matrix.

The proportion of ambiguous clustering summarises it in one number — the fraction
of patient pairs that co-cluster neither reliably nor never. Low is stable.

Two guards live here alongside the clustering itself, and both were decisive for
this project. The confound screen asks whether the split tracks a technical
artefact rather than biology. The representation ladder asks whether the split
needs the model at all — and for the Luminal A cohort the answer was that a
copy-number burden score alone reproduces it, which is a finding rather than a
disappointment. Report it.

This module computes; it does not draw. Figures live in
:mod:`mogformer.analysis.plots`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

#: Co-assignment frequencies strictly inside this band count as ambiguous.
PAC_BOUNDS: tuple[float, float] = (0.1, 0.9)

#: Consensus index at or above which a patient is a core member of its cluster.
CORE_THRESHOLD = 0.8

#: Clustering algorithms accepted by the consensus functions.
CLUSTERING_ALGORITHMS: tuple[str, ...] = ("kmeans", "gmm")


def fit_partition(
    values: np.ndarray,
    n_clusters: int,
    algorithm: str = "kmeans",
    seed: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """Partition patients with one clustering algorithm.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        n_clusters: Number of clusters.
        algorithm: One of :data:`CLUSTERING_ALGORITHMS`. ``"gmm"`` fits a
            full-covariance Gaussian mixture, with a small ridge on the
            covariances so that small clusters stay well conditioned.
        seed: Random state.
        n_init: Initialisations; the best fit is kept.

    Returns:
        Cluster label per patient.

    Raises:
        ValueError: If ``algorithm`` is unknown.
    """
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        return model.fit_predict(values)
    if algorithm == "gmm":
        mixture = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=n_init,
            random_state=seed,
        )
        return mixture.fit_predict(values)
    raise ValueError(
        f"unknown algorithm {algorithm!r}; available: {list(CLUSTERING_ALGORITHMS)}"
    )


@dataclass
class ConsensusResult:
    """The outcome of consensus clustering at one number of clusters.

    Attributes:
        n_clusters: Clusters requested.
        pac: Proportion of ambiguous clustering; lower is more stable.
        consensus: Co-assignment matrix, shape ``(n_samples, n_samples)``.
        labels: Final partition from clustering the full cohort.
        consensus_index: Per-patient mean co-assignment with its own cluster.
    """

    n_clusters: int
    pac: float
    consensus: np.ndarray
    labels: np.ndarray
    consensus_index: np.ndarray

    @property
    def core_mask(self) -> np.ndarray:
        """Return True for patients whose cluster membership is unambiguous."""
        return self.consensus_index >= CORE_THRESHOLD


def as_array(embedding: object) -> np.ndarray:
    """Coerce a torch tensor or array-like into a float array.

    Args:
        embedding: Tensor, frame or array of shape ``(n_samples, n_features)``.

    Returns:
        A NumPy array.
    """
    detach = getattr(embedding, "numpy", None)
    if callable(detach):
        return np.asarray(detach())
    if isinstance(embedding, pd.DataFrame):
        return embedding.to_numpy()
    return np.asarray(embedding)


def top_principal_components(
    values: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the leading principal components.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        n_components: Components requested; clipped to what the data supports.

    Returns:
        Tuple of the projection, shape ``(n_samples, k)``, and the explained
        variance ratio of each retained component.
    """
    centred = values - values.mean(0, keepdims=True)
    k = min(n_components, values.shape[1], values.shape[0] - 1)
    model = PCA(n_components=k).fit(centred)
    return model.transform(centred), model.explained_variance_ratio_


def consensus_matrix(
    values: np.ndarray,
    n_clusters: int,
    n_resample: int = 200,
    subsample_frac: float = 0.8,
    seed: int = 42,
    algorithm: str = "kmeans",
) -> np.ndarray:
    """Build the co-assignment matrix over resampled clusterings.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        n_clusters: Clusters per resampled run.
        n_resample: Number of resamples.
        subsample_frac: Fraction of patients drawn per resample.
        seed: Seed for the resampling and for each clustering run.
        algorithm: One of :data:`CLUSTERING_ALGORITHMS`.

    Returns:
        Matrix of shape ``(n_samples, n_samples)`` whose entries are the
        fraction of resamples in which both patients were drawn and grouped
        together.

    Raises:
        ValueError: If ``subsample_frac`` is not in ``(0, 1]`` or fewer than two
            clusters are requested.
    """
    if not 0 < subsample_frac <= 1:
        raise ValueError(f"subsample_frac must lie in (0, 1], got {subsample_frac}")
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    n_samples = values.shape[0]
    rng = np.random.default_rng(seed)
    together = np.zeros((n_samples, n_samples))
    drawn = np.zeros((n_samples, n_samples))
    size = max(int(subsample_frac * n_samples), n_clusters)

    for _ in range(n_resample):
        selected = rng.choice(n_samples, size, replace=False)
        labels = fit_partition(
            values[selected],
            n_clusters,
            algorithm,
            seed=int(rng.integers(1_000_000)),
            n_init=5 if algorithm == "kmeans" else 1,
        )

        one_hot = np.zeros((len(selected), n_clusters))
        one_hot[np.arange(len(selected)), labels] = 1
        together[np.ix_(selected, selected)] += one_hot @ one_hot.T
        drawn[np.ix_(selected, selected)] += 1

    return np.divide(together, drawn, out=np.zeros_like(together), where=drawn > 0)


def proportion_ambiguous_clustering(
    consensus: np.ndarray, bounds: tuple[float, float] = PAC_BOUNDS
) -> float:
    """Summarise a consensus matrix as its fraction of ambiguous pairs.

    A perfectly stable partition puts every off-diagonal entry at zero or one,
    so nothing falls inside the band and the value is zero.

    Args:
        consensus: Co-assignment matrix.
        bounds: Exclusive band counted as ambiguous.

    Returns:
        Fraction of off-diagonal pairs inside the band.
    """
    low, high = bounds
    off_diagonal = consensus[np.triu_indices(consensus.shape[0], k=1)]
    return float(np.mean((off_diagonal > low) & (off_diagonal < high)))


def consensus_index(consensus: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute each patient's mean co-assignment with its own cluster.

    A patient near one is a core member; a patient near the boundary between two
    clusters sits far lower, and reporting how many such patients exist is what
    stops a clean-looking partition from hiding a fragile margin.

    Args:
        consensus: Co-assignment matrix.
        labels: Cluster assignment per patient.

    Returns:
        Per-patient consensus index, shape ``(n_samples,)``.
    """
    scores = np.zeros(len(labels))
    for i, label in enumerate(labels):
        peers = np.where(labels == label)[0]
        peers = peers[peers != i]
        scores[i] = consensus[i, peers].mean() if len(peers) else 1.0
    return scores


def run_consensus_clustering(
    embedding: object,
    n_clusters_list: Sequence[int] = (2, 3, 4, 5),
    n_resample: int = 200,
    subsample_frac: float = 0.8,
    seed: int = 42,
    algorithm: str = "kmeans",
) -> dict[int, ConsensusResult]:
    """Cluster by consensus across a range of cluster counts.

    Args:
        embedding: Patient embedding, shape ``(n_samples, n_features)``.
        n_clusters_list: Cluster counts to evaluate.
        n_resample: Resamples per cluster count.
        subsample_frac: Fraction of patients per resample.
        seed: Seed for resampling and clustering.
        algorithm: One of :data:`CLUSTERING_ALGORITHMS`.

    Returns:
        Mapping of cluster count to its result.
    """
    values = as_array(embedding)
    results: dict[int, ConsensusResult] = {}

    for n_clusters in n_clusters_list:
        matrix = consensus_matrix(
            values, n_clusters, n_resample, subsample_frac, seed, algorithm
        )
        labels = fit_partition(values, n_clusters, algorithm, seed=seed)
        pac = proportion_ambiguous_clustering(matrix)
        results[n_clusters] = ConsensusResult(
            n_clusters=n_clusters,
            pac=pac,
            consensus=matrix,
            labels=labels,
            consensus_index=consensus_index(matrix, labels),
        )
        logger.info(
            "K=%d: PAC %.3f (lower is more stable) | %d/%d core members",
            n_clusters,
            pac,
            int(results[n_clusters].core_mask.sum()),
            len(labels),
        )
    return results


def order_by_linkage(consensus: np.ndarray) -> np.ndarray:
    """Order patients so a consensus heatmap shows its block structure.

    Args:
        consensus: Co-assignment matrix.

    Returns:
        Permutation of patient indices from average-linkage clustering of the
        implied distances.
    """
    distances = 1 - consensus
    np.fill_diagonal(distances, 0.0)
    return np.asarray(
        leaves_list(linkage(squareform(distances, checks=False), "average"))
    )


def silhouette_full_versus_pcs(
    embedding: object,
    n_clusters_list: Sequence[int] = (2, 3),
    n_components: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare silhouette in the full space against a leading-component space.

    Silhouette deflates as dimensionality grows, so a low value in the full
    embedding can understate real structure. If the projected value is markedly
    higher, the full-dimensional number was pessimistic rather than truthful.

    Args:
        embedding: Patient embedding.
        n_clusters_list: Cluster counts to evaluate.
        n_components: Components retained for the projected comparison.
        seed: Seed for the clustering.

    Returns:
        Frame with ``n_clusters``, ``silhouette_full`` and ``silhouette_pc``.
    """
    values = as_array(embedding)
    projected, _ = top_principal_components(values, n_components)

    rows = []
    for n_clusters in n_clusters_list:
        full_labels = KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(
            values
        )
        pc_labels = KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(
            projected
        )
        rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_full": float(silhouette_score(values, full_labels)),
                "silhouette_pc": float(silhouette_score(projected, pc_labels)),
            }
        )
        logger.info(
            "K=%d silhouette: full %.3f | top-%d PC %.3f",
            n_clusters,
            rows[-1]["silhouette_full"],
            n_components,
            rows[-1]["silhouette_pc"],
        )
    return pd.DataFrame(rows)


def per_cluster_silhouette(embedding: object, labels: np.ndarray) -> np.ndarray:
    """Return each patient's silhouette against its assigned cluster.

    Args:
        embedding: Patient embedding.
        labels: Cluster assignment per patient.

    Returns:
        Per-patient silhouette, shape ``(n_samples,)``.
    """
    return silhouette_samples(as_array(embedding), labels)


def zca_whiten(embedding: object, ridge: float = 1e-3) -> np.ndarray:
    """Remove the dominant scale from an embedding, keeping its axes.

    Whitening makes the variance isotropic by construction, so a scree plot of
    the result says nothing. It is worth doing only to ask one question: does
    separability survive once the largest-variance direction stops dominating?

    Args:
        embedding: Patient embedding.
        ridge: Added to the eigenvalues before inversion, for stability.

    Returns:
        Whitened data of the same shape.
    """
    values = as_array(embedding).astype(np.float64)
    centred = values - values.mean(0, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred, rowvar=False))
    whitener = (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + ridge)) @ eigenvectors.T
    )
    return centred @ whitener


def covariate_association(
    scores: np.ndarray, covariate: pd.Series
) -> tuple[float, float, str]:
    """Measure how strongly one covariate tracks a continuous score.

    Continuous covariates are summarised by absolute rank correlation and
    categorical ones by the correlation ratio, so effect sizes from the two
    kinds sit on a comparable zero-to-one scale.

    Args:
        scores: Continuous score per patient, such as a principal component.
        covariate: Covariate values, aligned to ``scores``. Missing values are
            dropped pairwise.

    Returns:
        Tuple of the effect size, its p-value and the test used: one of
        ``"spearman"``, ``"eta_squared"`` or ``"skip"``.
    """
    series = pd.Series(covariate).reset_index(drop=True)
    present = ~series.isna()
    values = np.asarray(scores)[present.to_numpy()]
    series = series[present]

    if len(values) < 5 or series.nunique() < 2:
        return float("nan"), float("nan"), "skip"

    is_continuous = pd.api.types.is_numeric_dtype(series) and series.nunique() > 8
    if is_continuous:
        rho, p_value = stats.spearmanr(values, series.astype(float))
        return abs(float(rho)), float(p_value), "spearman"

    groups = [
        values[(series == level).to_numpy()]
        for level in series.unique()
        if (series == level).sum() >= 2
    ]
    if len(groups) < 2:
        return float("nan"), float("nan"), "skip"

    grand_mean = values.mean()
    between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    total = float(((values - grand_mean) ** 2).sum())
    eta_squared = between / total if total > 0 else float("nan")
    try:
        _, p_value = stats.f_oneway(*groups)
    except ValueError:
        p_value = float("nan")
    return float(eta_squared), float(p_value), "eta_squared"


def screen_covariates(
    scores: np.ndarray,
    covariates: pd.DataFrame,
    roles: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Test every covariate against a score, with a multiplicity correction.

    The purpose is to decide whether a partition is molecular or technical
    before anything is claimed about it. Labelling each covariate's role —
    biology, clinical or technical — is what makes the verdict readable: a
    partition dominated by a biological covariate is a result, one dominated by
    a sequencing site is an artefact.

    Args:
        scores: Continuous score per patient.
        covariates: One column per covariate, aligned to ``scores``.
        roles: Optional mapping of covariate name to role, carried through to
            the output for plotting.

    Returns:
        Frame with the effect size, raw and Holm-corrected p-values, the test
        used and the role, sorted by descending effect size.
    """
    rows = []
    for name in covariates.columns:
        effect, p_value, test = covariate_association(scores, covariates[name])
        rows.append(
            {
                "covariate": name,
                "effect_size": effect,
                "p_value": p_value,
                "test": test,
                "role": (roles or {}).get(name, "unspecified"),
            }
        )

    frame = pd.DataFrame(rows)
    frame["p_holm"] = _holm_correct(frame["p_value"].to_numpy())
    return frame.sort_values("effect_size", ascending=False).reset_index(drop=True)


def _holm_correct(p_values: np.ndarray) -> np.ndarray:
    """Apply the Holm step-down correction, ignoring missing values.

    Args:
        p_values: Raw p-values, possibly containing ``nan``.

    Returns:
        Corrected p-values, with ``nan`` preserved in place.
    """
    corrected = np.full(len(p_values), np.nan)
    testable = np.where(~np.isnan(p_values))[0]
    if len(testable) == 0:
        return corrected

    order = testable[np.argsort(p_values[testable])]
    n_tests = len(order)
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min((n_tests - rank) * p_values[index], 1.0)
        running_max = max(running_max, adjusted)
        corrected[index] = running_max
    return corrected


@dataclass
class RepresentationRung:
    """One rung of the representation ablation ladder.

    Attributes:
        name: Label for the representation, such as ``"cnv_only"``.
        pac: Stability of the partition it produces.
        labels: The partition itself.
        ari_vs_reference: Agreement with the frozen reference partition.
    """

    name: str
    pac: float
    labels: np.ndarray
    ari_vs_reference: float = field(default=float("nan"))


def compare_representations(
    representations: Mapping[str, object],
    reference_labels: np.ndarray,
    n_clusters: int = 2,
    n_resample: int = 200,
    subsample_frac: float = 0.8,
    seed: int = 42,
) -> list[RepresentationRung]:
    """Ask whether a partition needs the model that produced it.

    Each representation is clustered under the identical consensus protocol and
    compared to the reference partition. High agreement from a simpler
    representation means the structure was never the model's contribution — the
    honest conclusion, and one worth stating plainly rather than burying.

    Args:
        representations: Mapping of name to data, shape
            ``(n_samples, n_features)``, all over the same patients in the same
            order.
        reference_labels: The frozen partition to compare against.
        n_clusters: Clusters per representation.
        n_resample: Resamples per representation.
        subsample_frac: Fraction of patients per resample.
        seed: Seed for resampling and clustering.

    Returns:
        One rung per representation, in input order.

    Raises:
        ValueError: If a representation has a different number of patients from
            the reference partition.
    """
    rungs = []
    for name, data in representations.items():
        values = as_array(data)
        if values.shape[0] != len(reference_labels):
            raise ValueError(
                f"representation {name!r} covers {values.shape[0]} patients "
                f"but the reference partition covers {len(reference_labels)}"
            )
        matrix = consensus_matrix(values, n_clusters, n_resample, subsample_frac, seed)
        labels = fit_partition(values, n_clusters, seed=seed)
        rung = RepresentationRung(
            name=name,
            pac=proportion_ambiguous_clustering(matrix),
            labels=labels,
            ari_vs_reference=float(adjusted_rand_score(reference_labels, labels)),
        )
        rungs.append(rung)
        logger.info(
            "%s: PAC %.3f | agreement with the reference partition ARI %.3f",
            name,
            rung.pac,
            rung.ari_vs_reference,
        )
    return rungs


def partition_agreement(partitions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise agreement between partitions of the same patients.

    Args:
        partitions: Mapping of name to cluster labels, all over the same
            patients in the same order.

    Returns:
        Square frame of adjusted Rand indices, indexed and columned by name.
    """
    names = list(partitions)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for row in names:
        for column in names:
            matrix.loc[row, column] = adjusted_rand_score(
                partitions[row], partitions[column]
            )
    return matrix


def bootstrap_agreement(
    values: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    algorithm: str = "kmeans",
    n_boot: int = 100,
    seed: int = 42,
) -> float:
    """Mean ARI between a partition and its refits on bootstrap resamples.

    Each resample draws patients with replacement, refits the partition on
    them, and compares the refit with the original labels on the distinct
    patients drawn. A partition that the data support reappears; one that
    depends on which patients happened to be present does not.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        labels: The partition of all patients.
        n_clusters: Number of clusters.
        algorithm: One of :data:`CLUSTERING_ALGORITHMS`.
        n_boot: Bootstrap resamples.
        seed: Seed of the resampling.

    Returns:
        The mean ARI over resamples.
    """
    rng = np.random.default_rng(seed)
    n_samples = values.shape[0]
    agreement = []
    for _ in range(n_boot):
        drawn = np.unique(rng.choice(n_samples, n_samples, replace=True))
        refit = fit_partition(
            values[drawn],
            n_clusters,
            algorithm,
            seed=int(rng.integers(1_000_000)),
            n_init=5 if algorithm == "kmeans" else 1,
        )
        agreement.append(adjusted_rand_score(labels[drawn], refit))
    return float(np.mean(agreement))


@dataclass(frozen=True)
class Reduction:
    """A fitted standardise-then-project map from features to components.

    Attributes:
        mean: Per-feature mean of the kept features.
        scale: Per-feature standard deviation of the kept features.
        kept: Boolean mask of the features with non-zero spread.
        components: Principal axes, shape ``(n_kept_components, n_kept)``; an
            identity row for a single feature.
    """

    mean: np.ndarray
    scale: np.ndarray
    kept: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map data onto the kept components.

        Args:
            values: Data of shape ``(n_samples, n_features)``.

        Returns:
            Component scores, shape ``(n_samples, n_kept_components)``.
        """
        z = (as_array(values)[:, self.kept] - self.mean) / self.scale
        return z @ self.components.T

    def weights_on_features(self, coefficients: np.ndarray) -> np.ndarray:
        """Express a linear read-out of the components as weights on the features.

        A score ``components · coefficients`` equals ``features · w`` up to a
        constant, with the ``w`` returned here, which is what attributing the
        score back to the features needs.

        Args:
            coefficients: One coefficient per kept component.

        Returns:
            One weight per original feature (zero for constant features).
        """
        weights = np.zeros(self.kept.shape[0])
        weights[self.kept] = self.components.T @ coefficients / self.scale
        return weights


def fit_reduction(
    values: np.ndarray, variance: float = 0.9, max_components: int = 32
) -> Reduction:
    """Fit the standardise-then-PCA reduction used before every downstream model.

    Every representation is reduced the same way before clustering or a
    survival model, so that a 512-d block-structured embedding, a PCA
    comparator and a single score are treated on comparable terms, and a
    Gaussian mixture has a covariance it can estimate. Fitting uses no label
    and no outcome.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        variance: Share of variance the kept components must explain.
        max_components: Upper bound on the components kept.

    Returns:
        The fitted reduction. A single feature is kept, standardised.
    """
    values = as_array(values)
    spread = values.std(axis=0)
    kept = spread > 0
    mean, scale = values[:, kept].mean(axis=0), spread[kept]
    if int(kept.sum()) == 1:
        return Reduction(mean, scale, kept, np.ones((1, 1)))
    z = (values[:, kept] - mean) / scale
    limit = min(max_components, z.shape[0] - 1, z.shape[1])
    pca = PCA(n_components=limit).fit(z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_kept = int(min(np.searchsorted(cumulative, variance) + 1, limit))
    # PCA centres internally; z is already centred, so the projection is exact.
    return Reduction(mean, scale, kept, pca.components_[:n_kept])


def reduce_for_clustering(
    values: np.ndarray, variance: float = 0.9, max_components: int = 32
) -> np.ndarray:
    """Standardise each feature, then keep the leading principal components.

    Shorthand for fitting :func:`fit_reduction` and applying it to the same
    data.

    Args:
        values: Data of shape ``(n_samples, n_features)``.
        variance: Share of variance the kept components must explain.
        max_components: Upper bound on the components kept.

    Returns:
        Principal-component scores, shape ``(n_samples, n_kept)``. A single
        feature is returned standardised.
    """
    return fit_reduction(values, variance, max_components).transform(values)


def partition_audit(
    labels: np.ndarray, axes: pd.DataFrame, roles: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Compare a partition with every known axis, Holm-corrected.

    Categorical axes are compared by adjusted mutual information and adjusted
    Rand index, with a chi-squared test. Continuous axes are compared by the
    share of their variance that lies between strata (the correlation ratio),
    with a one-way ANOVA, through :func:`covariate_association`.

    Args:
        labels: Stratum per patient.
        axes: One column per known axis, aligned to ``labels``.
        roles: Optional role per axis (``"biology"``, ``"clinical"``,
            ``"technical"``), carried through.

    Returns:
        One row per axis: ``kind``, ``ami``, ``ari``, ``eta_squared``, the
        raw and Holm-corrected p-values and the role, sorted by the main
        effect size (AMI or the correlation ratio).
    """
    rows = []
    for name in axes.columns:
        series = axes[name].reset_index(drop=True)
        present = ~series.isna()
        is_continuous = pd.api.types.is_numeric_dtype(series) and series.nunique() > 8
        row: dict[str, object] = {
            "axis": name,
            "n": int(present.sum()),
            "role": (roles or {}).get(name, "unspecified"),
        }
        if is_continuous:
            effect, p_value, _ = covariate_association(
                series.to_numpy(dtype=float)[present.to_numpy()],
                pd.Series(labels[present.to_numpy()]).astype(str),
            )
            row.update(kind="continuous", eta_squared=effect, effect=effect)
        else:
            known = series[present].astype(str).to_numpy()
            strata = labels[present.to_numpy()]
            table = pd.crosstab(strata, known)
            p_value = (
                float(stats.chi2_contingency(table)[1])
                if table.shape[0] > 1 and table.shape[1] > 1
                else float("nan")
            )
            ami = float(adjusted_mutual_info_score(known, strata))
            row.update(
                kind="categorical",
                ami=ami,
                ari=float(adjusted_rand_score(known, strata)),
                effect=ami,
            )
        row["p_value"] = p_value
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["p_holm"] = _holm_correct(frame["p_value"].fillna(1.0).to_numpy())
    return frame.sort_values("effect", ascending=False).reset_index(drop=True)


def block_separation(
    embedding: np.ndarray, labels: np.ndarray, block_width: int
) -> pd.DataFrame:
    """Share of each embedding block's variance that lies between strata.

    Args:
        embedding: Patients x dimensions, made of consecutive blocks.
        labels: Stratum per patient.
        block_width: Width of each block (a gene program's embedding).

    Returns:
        One row per block: its index and the between-strata share of its
        variance, sorted from most to least separating.
    """
    rows = []
    for block in range(embedding.shape[1] // block_width):
        values = embedding[:, block * block_width : (block + 1) * block_width]
        centred = values - values.mean(axis=0)
        total = float((centred**2).sum())
        between = sum(
            float((labels == level).sum())
            * float(
                (
                    (values[labels == level].mean(axis=0) - values.mean(axis=0)) ** 2
                ).sum()
            )
            for level in np.unique(labels)
        )
        rows.append(
            {"block": block, "between_share": between / total if total else 0.0}
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("between_share", ascending=False).reset_index(drop=True)
