"""Tests for the stratification protocol's building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.analysis.clustering import (
    block_separation,
    bootstrap_agreement,
    fit_partition,
    partition_audit,
    reduce_for_clustering,
    run_consensus_clustering,
)
from mogformer.experiments.patient_stratification import (
    choose_partition,
    differential_features,
)


@pytest.fixture
def two_groups() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated groups of 60 patients in 10 dimensions."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 60)
    values = rng.normal(size=(120, 10)) + 4.0 * labels[:, None] * (np.arange(10) < 3)
    return values, labels


def test_both_algorithms_recover_separated_groups(two_groups) -> None:
    """k-means and the Gaussian mixture find the same two groups."""
    values, labels = two_groups
    for algorithm in ("kmeans", "gmm"):
        found = fit_partition(values, 2, algorithm, seed=0)
        assert abs(np.corrcoef(found, labels)[0, 1]) > 0.95
    with pytest.raises(ValueError, match="unknown algorithm"):
        fit_partition(values, 2, "spectral")


def test_consensus_with_a_mixture_is_stable_on_separated_groups(two_groups) -> None:
    """PAC is near zero and bootstrap refits agree with the full partition."""
    values, _ = two_groups
    result = run_consensus_clustering(values, (2,), n_resample=30, algorithm="gmm")[2]

    assert result.pac < 0.05
    assert bootstrap_agreement(values, result.labels, 2, "gmm", n_boot=10) > 0.9


def test_reduction_keeps_the_components_that_explain_the_variance() -> None:
    """Low-rank data keep few components; one feature stays one column."""
    rng = np.random.default_rng(1)
    low_rank = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 40))
    low_rank += 0.01 * rng.normal(size=low_rank.shape)

    assert reduce_for_clustering(low_rank, 0.9, 32).shape[1] <= 3
    assert reduce_for_clustering(rng.normal(size=(50, 1))).shape == (50, 1)


def test_the_audit_finds_the_axis_a_partition_restates(two_groups) -> None:
    """A partition equal to a known label scores AMI 1; noise scores near 0."""
    _, labels = two_groups
    rng = np.random.default_rng(2)
    axes = pd.DataFrame(
        {
            "subtype": np.where(labels == 1, "A", "B"),
            "noise_label": rng.choice(["x", "y"], len(labels)),
            "score": labels * 3.0 + rng.normal(size=len(labels)),
        }
    )

    audit = partition_audit(labels, axes, {"subtype": "biology"}).set_index("axis")

    assert audit.loc["subtype", "ami"] == pytest.approx(1.0)
    assert audit.loc["noise_label", "ami"] < 0.05
    assert audit.loc["score", "eta_squared"] > 0.6
    assert audit.loc["subtype", "role"] == "biology"
    assert audit.loc["noise_label", "p_holm"] >= audit.loc["noise_label", "p_value"]


def test_block_separation_points_to_the_block_that_splits(two_groups) -> None:
    """Only the block holding the group difference separates the strata."""
    values, labels = two_groups
    embedding = np.hstack(
        [np.random.default_rng(3).normal(size=(120, 4)), values[:, :4]]
    )

    separation = block_separation(embedding, labels, block_width=4)

    assert separation.iloc[0]["block"] == 1
    assert separation.iloc[0]["between_share"] > 0.4


def test_the_reported_partition_follows_the_pre_registered_rule() -> None:
    """Lowest PAC among passing rows wins; with none passing, the best is flagged."""
    table = pd.DataFrame(
        {
            "representation": ["encoder"] * 3 + ["pca_inputs"],
            "algorithm": ["kmeans", "gmm", "kmeans", "kmeans"],
            "k": [2, 2, 3, 2],
            "pac": [0.05, 0.05, 0.02, 0.0],
            "meets_bar": [True, True, False, True],
        }
    )

    row, meets = choose_partition(table)
    assert (row["k"], row["algorithm"], meets) == (2, "kmeans", True)

    table["meets_bar"] = False
    row, meets = choose_partition(table)
    assert (row["k"], meets) == (3, False)


def test_differential_features_find_the_gene_that_differs() -> None:
    """The planted gene ranks first for its stratum and omics."""
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(80, 20, 3))
    labels = np.repeat([0, 1], 40)
    inputs[labels == 1, 7, 0] += 3.0

    table = differential_features(inputs, [f"G{i}" for i in range(20)], labels, top=3)
    top = table[(table["stratum"] == 1) & (table["omics"] == "rna")].iloc[0]

    assert top["gene"] == "G7" and top["difference"] > 2 and top["p_bh"] < 1e-6


def test_uno_concordance_is_one_for_perfect_risk_and_half_for_noise() -> None:
    """Risk that orders events perfectly scores 1; random risk scores near 0.5."""
    from lifelines.utils import concordance_index

    from mogformer.analysis.survival import uno_concordance

    rng = np.random.default_rng(5)
    time = rng.exponential(10, 400)
    event = rng.random(400) < 0.7

    assert uno_concordance(time, event, -time) == pytest.approx(1.0)
    noise = uno_concordance(time, event, rng.normal(size=400))
    assert abs(noise - 0.5) < 0.06
    risk = -time + rng.normal(scale=8, size=400)
    harrell = concordance_index(time, -risk, event)
    assert abs(uno_concordance(time, event, risk) - harrell) < 0.05


def test_the_log_rank_test_handles_more_than_two_groups() -> None:
    """Three groups with different hazards differ; one group alone is refused."""
    from mogformer.analysis.survival import logrank_p_value

    rng = np.random.default_rng(6)
    group = np.repeat([0, 1, 2], 100)
    frame = pd.DataFrame(
        {
            "t": rng.exponential(np.array([5.0, 10.0, 20.0])[group]),
            "e": 1,
            "g": group,
        }
    )

    assert logrank_p_value(frame, "t", "e", "g") < 1e-6
    with pytest.raises(ValueError, match="two or more"):
        logrank_p_value(frame.assign(g=0), "t", "e", "g")
