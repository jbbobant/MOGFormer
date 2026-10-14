"""Tests for the locked outcome analysis, on synthetic outcomes only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.experiments.outcome_analysis import (
    CLINICAL,
    continuous_cindex,
    incremental_value,
    landmark_split,
    outcome_frame,
    permute_outcomes,
    strata_ladder,
)


@pytest.fixture
def cohort() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """300 patients; stratum 1 relapses three times faster; a risk feature."""
    rng = np.random.default_rng(0)
    n = 300
    strata = (rng.random(n) < 0.4).astype(int)
    signal = rng.normal(size=n)
    hazard = np.exp(np.log(3.0) * strata + 0.8 * signal)
    time = rng.exponential(60.0 / hazard)
    censor = rng.uniform(20, 200, n)
    frame = pd.DataFrame(
        {
            "rfs_months": np.minimum(time, censor),
            "rfs_event": (time <= censor).astype(int),
            "dss_months": np.minimum(time * 1.3, censor),
            "dss_event": (time * 1.3 <= censor).astype(int),
            **{name: rng.normal(size=n) for name in CLINICAL},
            "cohort": np.arange(n) % 3 + 1.0,
        }
    )
    features = np.column_stack([signal, rng.normal(size=(n, 3))])
    return frame, strata, features


def test_the_outcome_frame_reads_the_metabric_codes() -> None:
    """Relapse, disease-specific death and the clinical codes are decoded."""
    clinical = pd.DataFrame(
        {
            "RFS_MONTHS": [10.0, 20.0],
            "RFS_STATUS": ["1:Recurred", "0:Not Recurred"],
            "OS_MONTHS": [30.0, 40.0],
            "VITAL_STATUS": ["Died of Disease", "Died of Other Causes"],
            "AGE_AT_DIAGNOSIS": [50.0, 60.0],
            "NPI": [3.0, 4.0],
            "HORMONE_THERAPY": ["YES", "NO"],
            "CHEMOTHERAPY": ["NO", "YES"],
            "INTCLUST": ["2", "4ER+"],
            "COHORT": [3.0, 1.0],
        },
        index=["a", "b"],
    )
    frame = outcome_frame(clinical, ["b", "a"], pd.Series({"a": 0.1, "b": 0.2}))

    assert frame["rfs_event"].tolist() == [0, 1]
    assert frame["dss_event"].tolist() == [0, 1]
    assert frame["ic_high"].tolist() == [0, 1]
    assert frame["endocrine"].tolist() == [0, 1]
    assert frame["proliferation"].tolist() == [0.2, 0.1]


def test_permutation_keeps_the_outcome_and_destroys_the_association(cohort) -> None:
    """Event counts survive; the stratum effect disappears."""
    frame, strata, _ = cohort
    shuffled = permute_outcomes(frame, seed=1)

    assert shuffled["rfs_event"].sum() == frame["rfs_event"].sum()
    real = strata_ladder(frame, "rfs", strata).query("rung == 'unadjusted'")
    null = strata_ladder(shuffled, "rfs", strata).query("rung == 'unadjusted'")
    assert real["hazard_ratio"].iloc[0] > 2.0
    assert null["ci_low"].iloc[0] < 1.0 < null["ci_high"].iloc[0]


def test_the_ladder_reports_every_rung_against_the_largest_stratum(cohort) -> None:
    """Seven rungs per endpoint, the majority stratum as reference."""
    frame, strata, _ = cohort
    ladder = strata_ladder(frame, "rfs", strata)

    assert len(ladder) == 7
    assert set(ladder["reference"]) == {0}
    assert (ladder["hazard_ratio"] > 1.5).all()


def test_the_landmark_splits_follow_up_at_five_years(cohort) -> None:
    """Early follow-up stops at 60 months; late starts there."""
    frame, _, _ = cohort
    parts = landmark_split(frame, "rfs")

    assert parts["early"]["rfs_months"].max() <= 60.0
    assert (parts["late"]["rfs_months"] > 0).all()
    assert len(parts["late"]) == int((frame["rfs_months"] > 60).sum())


def test_strata_add_concordance_over_an_uninformative_clinical_model(cohort) -> None:
    """With noise covariates, informative strata raise C and pass the LR test."""
    frame, strata, _ = cohort
    result = incremental_value(frame, "rfs", strata, n_boot=20)

    assert result["delta_c"] > 0.05
    assert result["lr_p_value"] < 1e-4


def test_an_informative_embedding_beats_the_clinical_model_out_of_fold(cohort) -> None:
    """The feature carrying risk scores above 0.6; noise covariates near 0.5."""
    frame, _, features = cohort
    table = continuous_cindex(
        frame, "rfs", {"encoder": features}, n_splits=3, n_repeats=1
    ).set_index("representation")

    assert table.loc["encoder", "c_index_mean"] > 0.6
    assert abs(table.loc["clinical", "c_index_mean"] - 0.5) < 0.08


def test_the_amended_rung_and_batch_models_run(cohort) -> None:
    """Amendment 1: a batch-stratified rung, a stratified ΔC, batch indicators."""
    frame, strata, features = cohort

    ladder = strata_ladder(frame, "rfs", strata)
    stratified = ladder[ladder["rung"] == "+ stratified by batch"]
    value = incremental_value(frame, "rfs", strata, n_boot=10, batch_stratified=True)
    table = continuous_cindex(
        frame,
        "rfs",
        {"encoder": features},
        n_splits=3,
        n_repeats=1,
        batch_indicators=True,
    )

    assert len(ladder) == 7 and stratified["hazard_ratio"].iloc[0] > 1.5
    assert value["batch_stratified"] and value["delta_c"] > 0.03
    assert set(table["representation"]) == {
        "clinical + batch",
        "clinical + batch + encoder",
    }
