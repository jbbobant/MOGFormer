"""Unit tests for clustering, survival and the interventional probes.

Everything runs on constructed data with a known answer, so the tests check the
statistics rather than re-deriving them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.analysis.clustering import (
    CORE_THRESHOLD,
    compare_representations,
    consensus_index,
    consensus_matrix,
    covariate_association,
    partition_agreement,
    proportion_ambiguous_clustering,
    run_consensus_clustering,
    screen_covariates,
    silhouette_full_versus_pcs,
    top_principal_components,
    zca_whiten,
)
from mogformer.analysis.probe_cis import (
    DEFAULT_GRID,
    GridRankCorrelation,
    make_gene_batches,
    patient_slopes,
    summarise_response,
)
from mogformer.analysis.probe_trans import (
    magnitude_tests,
    permuted_graph_retention,
    sign_permutation_test,
)
from mogformer.analysis.survival import (
    MIN_EVENTS_PER_VARIABLE,
    check_events_per_variable,
    months_from_days,
)


@pytest.fixture()
def two_blobs() -> np.ndarray:
    """Return 60 points in two well-separated clusters."""
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.normal(-5.0, 0.3, size=(30, 4)),
            rng.normal(+5.0, 0.3, size=(30, 4)),
        ]
    )


# --------------------------------------------------------------------------
# Consensus clustering
# --------------------------------------------------------------------------
def test_consensus_matrix_is_symmetric_and_bounded(two_blobs) -> None:
    """Co-assignment frequencies are symmetric probabilities."""
    matrix = consensus_matrix(two_blobs, n_clusters=2, n_resample=30, seed=0)

    assert matrix.shape == (60, 60)
    assert np.allclose(matrix, matrix.T)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_well_separated_clusters_give_near_zero_pac(two_blobs) -> None:
    """A partition that always recurs leaves no ambiguous pairs."""
    matrix = consensus_matrix(two_blobs, n_clusters=2, n_resample=50, seed=0)
    assert proportion_ambiguous_clustering(matrix) < 0.05


def test_pac_rises_on_structureless_data() -> None:
    """Clustering noise produces the ambiguity the statistic is meant to catch."""
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(60, 4))
    structured = np.vstack(
        [rng.normal(-5.0, 0.3, (30, 4)), rng.normal(5.0, 0.3, (30, 4))]
    )

    noise_pac = proportion_ambiguous_clustering(
        consensus_matrix(noise, 2, n_resample=50, seed=0)
    )
    structured_pac = proportion_ambiguous_clustering(
        consensus_matrix(structured, 2, n_resample=50, seed=0)
    )
    assert noise_pac > structured_pac


def test_consensus_index_marks_core_members(two_blobs) -> None:
    """Members of a clean partition are all core."""
    matrix = consensus_matrix(two_blobs, n_clusters=2, n_resample=50, seed=0)
    labels = np.array([0] * 30 + [1] * 30)
    index = consensus_index(matrix, labels)

    assert index.shape == (60,)
    assert (index >= CORE_THRESHOLD).all()


def test_run_consensus_clustering_reports_each_k(two_blobs) -> None:
    """Every requested cluster count comes back with its own result."""
    results = run_consensus_clustering(
        two_blobs, n_clusters_list=(2, 3), n_resample=20, seed=0
    )

    assert set(results) == {2, 3}
    assert results[2].pac <= results[3].pac
    assert results[2].labels.shape == (60,)
    assert results[2].core_mask.dtype == bool


def test_consensus_matrix_rejects_bad_arguments(two_blobs) -> None:
    """A single cluster or an out-of-range subsample fraction is refused."""
    with pytest.raises(ValueError, match="n_clusters must be"):
        consensus_matrix(two_blobs, n_clusters=1)
    with pytest.raises(ValueError, match="subsample_frac"):
        consensus_matrix(two_blobs, n_clusters=2, subsample_frac=1.5)


def test_top_principal_components_are_clipped_to_the_data(two_blobs) -> None:
    """Requesting more components than exist returns what is available."""
    projected, ratio = top_principal_components(two_blobs, n_components=50)
    assert projected.shape[0] == 60
    assert projected.shape[1] <= 4
    assert len(ratio) == projected.shape[1]


def test_silhouette_comparison_returns_both_spaces(two_blobs) -> None:
    """Full-dimensional and projected silhouettes are reported side by side."""
    frame = silhouette_full_versus_pcs(two_blobs, n_clusters_list=(2,), n_components=2)
    assert list(frame.columns) == [
        "n_clusters",
        "silhouette_full",
        "silhouette_pc",
    ]
    assert frame["silhouette_full"].iloc[0] > 0.5


def test_zca_whitening_produces_isotropic_covariance(two_blobs) -> None:
    """Whitened data has near-identity covariance, by construction."""
    whitened = zca_whiten(two_blobs, ridge=1e-6)
    covariance = np.cov(whitened, rowvar=False)
    assert np.allclose(covariance, np.eye(4), atol=1e-2)


# --------------------------------------------------------------------------
# Confound screening
# --------------------------------------------------------------------------
def test_covariate_association_detects_a_continuous_relationship() -> None:
    """A covariate that tracks the score yields a large rank correlation."""
    scores = np.arange(50, dtype=float)
    effect, p_value, test = covariate_association(scores, pd.Series(scores * 2))

    assert test == "spearman"
    assert effect == pytest.approx(1.0, abs=1e-9)
    assert p_value < 1e-6


def test_covariate_association_handles_categories() -> None:
    """A categorical covariate is summarised by the correlation ratio."""
    scores = np.concatenate([np.zeros(25), np.ones(25)])
    groups = pd.Series(["a"] * 25 + ["b"] * 25)
    effect, _, test = covariate_association(scores, groups)

    assert test == "eta_squared"
    assert effect == pytest.approx(1.0, abs=1e-9)


def test_covariate_association_skips_degenerate_input() -> None:
    """A constant or near-empty covariate is skipped rather than fabricated."""
    scores = np.arange(50, dtype=float)
    _, _, test = covariate_association(scores, pd.Series(["x"] * 50))
    assert test == "skip"


def test_screen_covariates_corrects_for_multiplicity() -> None:
    """Holm-corrected p-values are never smaller than the raw ones."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=80)
    covariates = pd.DataFrame(
        {
            "real": scores * 3 + rng.normal(scale=0.1, size=80),
            "noise_a": rng.normal(size=80),
            "noise_b": rng.normal(size=80),
        }
    )
    frame = screen_covariates(scores, covariates, roles={"real": "biology"})

    assert frame.iloc[0]["covariate"] == "real"
    assert frame.iloc[0]["role"] == "biology"
    testable = frame.dropna(subset=["p_value"])
    assert (testable["p_holm"] >= testable["p_value"] - 1e-12).all()


# --------------------------------------------------------------------------
# Representation ladder
# --------------------------------------------------------------------------
def test_a_simpler_representation_can_reproduce_the_partition(two_blobs) -> None:
    """A single informative column recovers the reference partition exactly.

    The property behind the project's finding that a copy-number burden score
    alone reproduces the embedding's Luminal A split.
    """
    reference = np.array([0] * 30 + [1] * 30)
    rungs = compare_representations(
        {
            "full": two_blobs,
            "one_column": two_blobs[:, [0]],
        },
        reference_labels=reference,
        n_clusters=2,
        n_resample=20,
        seed=0,
    )

    by_name = {rung.name: rung for rung in rungs}
    assert by_name["one_column"].ari_vs_reference == pytest.approx(1.0)
    assert by_name["full"].ari_vs_reference == pytest.approx(1.0)


def test_representation_ladder_rejects_mismatched_cohorts(two_blobs) -> None:
    """A representation covering different patients is refused."""
    with pytest.raises(ValueError, match="covers"):
        compare_representations(
            {"short": two_blobs[:10]},
            reference_labels=np.zeros(60, dtype=int),
        )


def test_partition_agreement_is_one_on_the_diagonal() -> None:
    """Every partition agrees perfectly with itself."""
    matrix = partition_agreement(
        {"a": np.array([0, 0, 1, 1]), "b": np.array([1, 1, 0, 0])}
    )
    assert matrix.loc["a", "a"] == pytest.approx(1.0)
    # A relabelling is the same partition, and the index knows it.
    assert matrix.loc["a", "b"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Probe statistics
# --------------------------------------------------------------------------
def test_patient_slopes_recover_a_known_gradient() -> None:
    """A response built with a known slope returns that slope."""
    grid = DEFAULT_GRID
    slopes = np.array([-0.5, 0.0, 2.0])
    response = grid[:, None] * slopes[None, :] + 7.0

    assert np.allclose(patient_slopes(response, grid), slopes)


def test_patient_slopes_ignore_an_offset() -> None:
    """A per-patient intercept does not change the slope."""
    grid = DEFAULT_GRID
    base = grid[:, None] * np.array([1.0, 1.0])
    shifted = base + np.array([100.0, -100.0])

    assert np.allclose(patient_slopes(base, grid), patient_slopes(shifted, grid))


def test_grid_rank_correlation_is_signed_correctly() -> None:
    """A silencing response — injection up, prediction down — is negative."""
    grid = DEFAULT_GRID
    correlator = GridRankCorrelation(n_patients=4, grid=grid)

    silencing = grid[:, None] * np.full(4, -1.0)
    activating = grid[:, None] * np.full(4, +1.0)

    assert correlator(silencing.reshape(-1)) == pytest.approx(-1.0, abs=1e-6)
    assert correlator(activating.reshape(-1)) == pytest.approx(1.0, abs=1e-6)


def test_summarise_response_intervals_bracket_the_slope() -> None:
    """The bootstrap interval contains the point estimate."""
    grid = DEFAULT_GRID
    rng = np.random.default_rng(0)
    response = grid[:, None] * rng.normal(-0.3, 0.05, size=50) + rng.normal(
        scale=0.01, size=(len(grid), 50)
    )
    correlator = GridRankCorrelation(50, grid)

    stats = summarise_response(response, correlator, rng, grid, n_boot=200)

    assert stats.slope_lo <= stats.slope <= stats.slope_hi
    assert stats.slope < 0
    assert stats.rho < 0
    assert stats.frac_patients_negative > 0.9
    assert stats.interval_excludes_zero


def test_gene_batches_never_group_close_or_regulated_genes() -> None:
    """Batch members are far apart and share no regulatory edge."""
    n_genes = 6
    spd = np.full((n_genes, n_genes), 9)
    np.fill_diagonal(spd, 0)
    spd[0, 1] = spd[1, 0] = 1  # adjacent, must not share a batch
    grn = np.zeros((n_genes, n_genes))
    grn[3, 2] = 1.0  # regulated, must not share a batch

    batches = make_gene_batches(spd, range(n_genes), batch_size=6, grn=grn)

    for batch in batches:
        for i, gene in enumerate(batch):
            for other in batch[i + 1 :]:
                assert (gene, other) not in {(0, 1), (1, 0)}
                assert grn[gene, other] == 0
                assert grn[other, gene] == 0


def test_batch_size_one_disables_batching() -> None:
    """Exact mode probes one gene per pass."""
    spd = np.zeros((4, 4))
    assert make_gene_batches(spd, range(4), batch_size=1) == [[0], [1], [2], [3]]


# --------------------------------------------------------------------------
# Trans sign concordance — the corrected test
# --------------------------------------------------------------------------
def build_edges(
    n_activating: int, n_repressing: int, positive_frac: float
) -> pd.DataFrame:
    """Build an edge table with a chosen imbalance and slope-sign bias."""
    rng = np.random.default_rng(0)
    signs = np.array([1] * n_activating + [-1] * n_repressing)
    n_edges = len(signs)
    slopes = np.where(
        rng.random(n_edges) < positive_frac,
        rng.uniform(0.001, 0.01, n_edges),
        -rng.uniform(0.001, 0.01, n_edges),
    )
    return pd.DataFrame({"edge_sign": signs, "slope": slopes})


def test_sign_test_reports_the_trivial_baseline() -> None:
    """The majority-class rate is reported, not just the p-value.

    Regression test for the project's most consequential analysis error: with
    697 activating and 206 repressing edges, always guessing "activating" scores
    77%, so a 60% concordance is worse than a constant predictor even though a
    binomial test against 0.5 calls it overwhelmingly significant.
    """
    edges = build_edges(n_activating=697, n_repressing=206, positive_frac=0.65)
    result = sign_permutation_test(edges, n_perm=500)

    assert result.trivial_majority_baseline == pytest.approx(697 / 903, abs=1e-6)
    assert result.rate < result.trivial_majority_baseline
    assert not result.beats_trivial_baseline
    assert not result.is_informative


def test_sign_test_null_absorbs_the_class_imbalance() -> None:
    """The permutation null sits far above 0.5 when the classes are imbalanced.

    That gap is exactly what a binomial test against a coin flip misses.
    """
    edges = build_edges(n_activating=697, n_repressing=206, positive_frac=0.65)
    result = sign_permutation_test(edges, n_perm=500)

    assert result.null_mean_rate > 0.5
    assert result.perm_p > 0.01


def test_sign_test_detects_a_genuine_signal() -> None:
    """Perfect concordance on a balanced edge set is significant and useful."""
    signs = np.array([1] * 100 + [-1] * 100)
    edges = pd.DataFrame({"edge_sign": signs, "slope": signs * 0.01})

    result = sign_permutation_test(edges, n_perm=500)

    assert result.rate == pytest.approx(1.0)
    assert result.beats_trivial_baseline
    assert result.perm_p < 0.01
    assert result.is_informative


def test_sign_test_requires_finite_slopes() -> None:
    """An all-missing edge table is refused rather than silently empty."""
    edges = pd.DataFrame({"edge_sign": [1, -1], "slope": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="finite slope"):
        sign_permutation_test(edges, n_perm=10)


# --------------------------------------------------------------------------
# Trans magnitude and the graph null
# --------------------------------------------------------------------------
def test_magnitude_test_separates_edges_from_controls() -> None:
    """Larger effects on real edges than on matched controls are detected."""
    rng = np.random.default_rng(0)
    sweep = pd.DataFrame(
        {
            "null": "real",
            "mask_mode": "rna_only",
            "is_edge": [1] * 100 + [0] * 100,
            "edge_sign": [1] * 100 + [0] * 100,
            "slope": np.concatenate(
                [rng.normal(0.004, 0.001, 100), rng.normal(0.001, 0.001, 100)]
            ),
            "slope_lo": 0.0005,
            "slope_hi": 0.006,
            "observed_coexpression": rng.normal(size=200),
        }
    )
    results = magnitude_tests(sweep, "rna_only")

    assert results["edge_abs_median"] > results["control_abs_median"]
    assert results["mw_p_edge_stronger"] < 0.01
    assert results["n_edges"] == 100


def test_permuted_graph_retention_reports_both_directions() -> None:
    """Retention and the graph-attributable share are complementary."""
    real = pd.DataFrame({"slope": [0.004] * 50})
    permuted = pd.DataFrame({"slope": [0.001] * 50})

    result = permuted_graph_retention(real, permuted)

    assert result["retained_pct"] == pytest.approx(25.0)
    assert result["graph_attributable_pct"] == pytest.approx(75.0)


def test_high_retention_means_a_small_graph_contribution() -> None:
    """The project's 64% retention leaves roughly a third to the graph."""
    real = pd.DataFrame({"slope": [0.002242913497022542] * 10})
    permuted = pd.DataFrame({"slope": [0.0014332904013906356] * 10})

    result = permuted_graph_retention(real, permuted)

    assert result["retained_pct"] == pytest.approx(63.9, abs=0.1)
    assert result["graph_attributable_pct"] == pytest.approx(36.1, abs=0.1)


# --------------------------------------------------------------------------
# Survival helpers
# --------------------------------------------------------------------------
def test_months_from_days_converts() -> None:
    """Follow-up in days becomes follow-up in months."""
    assert months_from_days(np.array([30.44, 365.28]))[0] == pytest.approx(1.0)
    assert months_from_days(np.array([365.28]))[0] == pytest.approx(12.0)


def test_events_per_variable_warns_below_the_ceiling(caplog) -> None:
    """A model with too few events per covariate is flagged as exploratory."""
    with caplog.at_level("WARNING"):
        check_events_per_variable(n_events=9, n_covariates=3)
    assert "events per variable" in caplog.text


def test_events_per_variable_is_quiet_when_adequate(caplog) -> None:
    """A well-powered model produces no warning."""
    with caplog.at_level("WARNING"):
        check_events_per_variable(n_events=MIN_EVENTS_PER_VARIABLE * 3, n_covariates=3)
    assert caplog.text == ""


def test_events_per_variable_rejects_zero_events() -> None:
    """A model with no events cannot be fitted at all."""
    with pytest.raises(ValueError, match="zero events"):
        check_events_per_variable(n_events=0, n_covariates=1)
