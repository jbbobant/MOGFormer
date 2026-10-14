"""Tests for the representation evaluation suite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mogformer.analysis.representation_evaluation import (
    ProbeTarget,
    clustering_checkpoint,
    comparator_representations,
    dominant_axis_share,
    linear_probe_scores,
    participation_ratio,
    probe_panel,
    summarise_probe_panel,
)


def test_participation_ratio_counts_evenly_used_dimensions() -> None:
    """Isotropic data uses every dimension; one dominant axis uses about one."""
    rng = np.random.default_rng(0)
    isotropic = rng.normal(size=(4000, 8))
    dominated = isotropic * np.array([30.0, 1, 1, 1, 1, 1, 1, 1])

    assert participation_ratio(isotropic) > 7.5
    assert participation_ratio(dominated) < 1.1


def test_a_probe_reads_a_linear_label_and_not_noise() -> None:
    """A label written into the features is decoded; a random one is not."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(300, 10))
    signal = ProbeTarget("signal", pd.Series(features[:, 0] * 2.0), "regression")
    noise = ProbeTarget("noise", pd.Series(rng.normal(size=300)), "regression")
    classes = ProbeTarget(
        "classes", pd.Series(np.where(features[:, 1] > 0, "a", "b")), "classification"
    )

    assert linear_probe_scores(features, signal).mean() > 0.95
    assert linear_probe_scores(features, noise).mean() < 0.05
    assert linear_probe_scores(features, classes).mean() > 0.9
    assert len(linear_probe_scores(features, signal)) == 25


def test_missing_labels_and_tiny_classes_are_left_out() -> None:
    """Patients without a label and classes too small to split are dropped."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 4))
    labels = pd.Series(np.where(features[:, 0] > 0, "a", "b"), dtype=object)
    labels.iloc[:20] = None
    labels.iloc[20:23] = "rare"

    scores = linear_probe_scores(features, ProbeTarget("t", labels, "classification"))

    assert np.isfinite(scores).all()


def test_the_panel_pairs_every_representation_with_the_reference() -> None:
    """Scores share partitions, so the difference to the reference is paired."""
    rng = np.random.default_rng(3)
    good = rng.normal(size=(250, 6))
    bad = rng.normal(size=(250, 6))
    target = ProbeTarget("t", pd.Series(good[:, 0]), "regression")

    per_fold = probe_panel({"good": good, "bad": bad}, [target])
    summary = summarise_probe_panel(per_fold, reference="good").set_index(
        "representation"
    )

    assert summary.loc["bad", "minus_good"] < -0.5
    assert summary.loc["bad", "corrected_p"] < 0.01
    assert np.isnan(summary.loc["good", "minus_good"])


def test_dominant_axis_share_finds_the_covariate_behind_the_first_component() -> None:
    """A representation built around one covariate is explained by it."""
    rng = np.random.default_rng(4)
    covariate = rng.normal(size=500)
    embedding = np.outer(covariate, rng.normal(size=16)) * 5 + rng.normal(
        size=(500, 16)
    )

    table = dominant_axis_share(
        embedding, {"driver": covariate, "other": rng.normal(size=500)}
    ).set_index("covariate")

    assert table.loc["driver", "pc1_r2"] > 0.9
    assert table.loc["driver", "variance_share"] > 0.8
    assert table.loc["other", "variance_share"] < 0.05


def test_comparators_match_the_requested_dimension() -> None:
    """PCA comparators have the embedding's width; raw inputs keep every value."""
    inputs = np.random.default_rng(5).normal(size=(120, 40, 3))

    representations = comparator_representations(inputs, n_components=30)

    assert representations["pca_concatenated"].shape == (120, 30)
    assert representations["pca_per_omics"].shape == (120, 30)
    assert representations["raw_inputs"].shape == (120, 120)


def test_the_checkpoint_recognises_a_known_split() -> None:
    """When the embedding is two blobs, the stable K=2 split is the reference."""
    rng = np.random.default_rng(6)
    groups = np.repeat([0, 1], 60)
    embedding = rng.normal(size=(120, 5)) + groups[:, None] * 6.0

    table = clustering_checkpoint(
        embedding,
        {"all": np.arange(120)},
        {"groups": groups},
        n_clusters_list=(2, 3),
        n_resample=20,
    ).set_index("k")

    assert table.loc[2, "pac"] < 0.05
    assert table.loc[2, "ari_groups"] > 0.95
