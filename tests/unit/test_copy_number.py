"""Unit tests for copy-number scales and ploidy.

Synthetic matrices with known answers: a uniformly doubled genome, homozygous
deletions, high-level amplifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.data.omics import (
    UNLABELLED,
    OmicsData,
    apply_copy_number_representation,
    load_omics,
    ploidy_estimate,
    relative_copy_number,
)


def _absolute(values: dict[str, list[float]]) -> pd.DataFrame:
    """Build a genes-by-patients absolute copy-number frame."""
    return pd.DataFrame(
        values, index=[f"G{i}" for i in range(len(next(iter(values.values()))))]
    )


def test_a_uniformly_doubled_genome_has_the_same_relative_profile() -> None:
    """Doubling every copy number doubles ploidy and leaves the profile alone."""
    base = [2.0, 3.0, 1.0, 2.0, 4.0, 2.0]
    cnv = _absolute({"diploid": base, "doubled": [2 * v for v in base]})
    relative = relative_copy_number(cnv, ploidy_estimate(cnv))

    np.testing.assert_allclose(relative["diploid"], relative["doubled"])


def test_a_homozygous_deletion_maps_to_the_floor_of_the_scale() -> None:
    """Zero copies are floored at 0.5: −2 for any ploidy of two or more."""
    cnv = _absolute({"patient": [0.0, 4.0, 2.0, 2.0], "doubled": [0.0, 8.0, 4, 4]})
    relative = relative_copy_number(cnv, ploidy_estimate(cnv))

    assert relative.loc["G0", "patient"] == pytest.approx(-2.0)
    assert relative.loc["G0", "doubled"] == pytest.approx(-2.0)


def test_high_level_amplifications_are_clipped_at_four_times_ploidy() -> None:
    """A 40-copy amplicon reads +2, not log2(40 / ploidy)."""
    cnv = _absolute({"patient": [40.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]})
    relative = relative_copy_number(cnv, ploidy_estimate(cnv))

    assert relative["patient"].max() == pytest.approx(2.0)
    assert relative["patient"].min() >= -2.0


def test_ploidy_is_the_mean_over_every_gene_not_a_selection() -> None:
    """The estimate uses the whole matrix; a subset would give another value."""
    cnv = _absolute({"patient": [2.0, 2.0, 2.0, 6.0]})

    assert ploidy_estimate(cnv)["patient"] == pytest.approx(3.0)
    assert ploidy_estimate(cnv.iloc[:3])["patient"] == pytest.approx(2.0)


def test_ploidy_rejects_negative_copy_number() -> None:
    """A negative copy number is a coding error, not a value."""
    with pytest.raises(ValueError, match="non-negative"):
        ploidy_estimate(_absolute({"patient": [2.0, -1.0]}))


def _cohort(cnv_block: np.ndarray) -> OmicsData:
    """Wrap a patients-by-genes copy-number block in a three-omics cohort."""
    n_patients, n_genes = cnv_block.shape
    features = np.hstack(
        [np.ones((n_patients, n_genes)), cnv_block, np.zeros((n_patients, n_genes))]
    )
    return OmicsData(
        X=features,
        y=np.zeros(n_patients, dtype=np.int64),
        patient_ids=[f"P{i}" for i in range(n_patients)],
        gene_names=[f"G{i}" for i in range(n_genes)],
        label_map={"a": 0},
        inverse_label_map={0: "a"},
    )


def test_representations_rewrite_only_the_copy_number_block() -> None:
    """Expression and methylation stay as they were on every scale."""
    data = _cohort(np.array([[2.0, 4.0, 1.0], [3.0, 3.0, 6.0]]))
    ploidy = pd.Series({"P0": 2.0, "P1": 4.0})

    for representation in ("ploidy_relative", "log_ratio", "ordinal"):
        rewritten = apply_copy_number_representation(data, representation, ploidy)
        np.testing.assert_array_equal(
            rewritten.X[:, data.block_slice("rna")], data.X[:, data.block_slice("rna")]
        )
        np.testing.assert_array_equal(
            rewritten.X[:, data.block_slice("methy")],
            data.X[:, data.block_slice("methy")],
        )


def test_each_scale_gives_its_defining_value() -> None:
    """Ploidy-relative divides by the patient's ploidy; log ratio by two."""
    data = _cohort(np.array([[4.0, 4.0]]))
    ploidy = pd.Series({"P0": 4.0})
    block = data.block_slice("cnv")

    relative = apply_copy_number_representation(data, "ploidy_relative", ploidy)
    log_ratio = apply_copy_number_representation(data, "log_ratio")
    ordinal = apply_copy_number_representation(data, "ordinal")

    np.testing.assert_allclose(relative.X[:, block], 0.0)
    np.testing.assert_allclose(log_ratio.X[:, block], 1.0)
    np.testing.assert_allclose(ordinal.X[:, block], 2.0)
    assert apply_copy_number_representation(data, "absolute") is data


def test_ploidy_relative_needs_a_ploidy_and_unknown_scales_fail() -> None:
    """Missing inputs and unknown names raise rather than guess."""
    data = _cohort(np.array([[2.0, 2.0]]))
    with pytest.raises(ValueError, match="ploidy"):
        apply_copy_number_representation(data, "ploidy_relative")
    with pytest.raises(ValueError, match="unknown copy-number"):
        apply_copy_number_representation(data, "gistic")


def test_unlabelled_patients_are_kept_when_asked(tmp_path) -> None:
    """A label-free cohort keeps every measured patient under its own class."""
    patients = ["P0", "P1", "P2"]
    genes = ["G0", "G1"]
    for name in ("rna.csv", "cnv.csv", "methy.csv"):
        pd.DataFrame(1.0, index=genes, columns=patients).to_csv(tmp_path / name)
    pd.DataFrame({"SUBTYPE": ["A", "B", None]}, index=patients).to_csv(
        tmp_path / "clin.csv"
    )
    arguments = {
        "raw_dir": tmp_path,
        "rna_file": "rna.csv",
        "cnv_file": "cnv.csv",
        "methy_file": "methy.csv",
        "clin_file": "clin.csv",
        "label_col": "SUBTYPE",
    }

    labelled_only = load_omics(**arguments)
    everyone = load_omics(**arguments, keep_unlabelled=True)

    assert labelled_only.patient_ids == ["P0", "P1"]
    assert everyone.patient_ids == patients
    assert everyone.class_counts() == {"A": 1, "B": 1, UNLABELLED: 1}
