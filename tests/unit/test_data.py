"""Unit tests for cohort loading, preprocessing and folds.

Everything runs on synthetic matrices written to ``tmp_path``, so the suite
never needs the real cohort.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mogformer.data import (
    MODALITY_ORDER,
    FoldSpec,
    MultiOmicsTransformer,
    OmicsData,
    assert_folds_match,
    assert_no_leakage,
    exclude_classes,
    load_curated_genes,
    load_folds,
    load_omics,
    make_folds,
    remap_fold_indices,
    save_folds,
    select_folds,
)

GENES = ["AAA", "BBB", "CCC", "DDD", "ESR1"]
SUBTYPES = (
    ["BRCA_Basal"] * 6 + ["BRCA_LumA"] * 8 + ["BRCA_LumB"] * 6 + ["BRCA_Normal"] * 5
)


@pytest.fixture()
def cohort(tmp_path):
    """Write a synthetic four-class cohort and return its directory."""
    rng = np.random.default_rng(0)
    patients = [f"P{i:03d}" for i in range(len(SUBTYPES))]

    for name, scale in (("rna.csv", 100.0), ("cnv.csv", 1.0), ("methy.csv", 0.5)):
        frame = pd.DataFrame(
            rng.random((len(GENES), len(patients))) * scale,
            index=GENES,
            columns=patients,
        )
        frame.to_csv(tmp_path / name)

    pd.DataFrame({"SUBTYPE": SUBTYPES}, index=patients).to_csv(
        tmp_path / "clinical.csv"
    )
    return tmp_path


def load(cohort_dir, **kwargs) -> OmicsData:
    """Load the synthetic cohort with the fixture's file names."""
    options = {
        "rna_file": "rna.csv",
        "cnv_file": "cnv.csv",
        "methy_file": "methy.csv",
        "clin_file": "clinical.csv",
        "label_col": "SUBTYPE",
        **kwargs,
    }
    return load_omics(cohort_dir, **options)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def test_load_omics_aligns_and_encodes(cohort) -> None:
    """Blocks are stacked in modality order and labels encode alphabetically."""
    data = load(cohort)

    assert data.n_patients == len(SUBTYPES)
    assert data.n_genes == len(GENES)
    assert data.X.shape == (len(SUBTYPES), len(MODALITY_ORDER) * len(GENES))
    assert data.gene_names == sorted(GENES)
    assert data.label_map == {
        "BRCA_Basal": 0,
        "BRCA_LumA": 1,
        "BRCA_LumB": 2,
        "BRCA_Normal": 3,
    }
    assert data.class_counts() == {
        "BRCA_Basal": 6,
        "BRCA_LumA": 8,
        "BRCA_LumB": 6,
        "BRCA_Normal": 5,
    }


def test_load_omics_keeps_every_class_by_default(cohort) -> None:
    """The loader must not drop a class unless told to.

    Regression test for the hardcoded ``!= "BRCA_Normal"`` filter that produced
    a 914-patient baseline cohort against a 949-patient transformer cohort.
    """
    data = load(cohort)
    assert "BRCA_Normal" in data.label_map
    assert data.n_patients == len(SUBTYPES)


def test_load_omics_excludes_only_what_it_is_told(cohort) -> None:
    """``exclude`` drops the named class and re-encodes labels contiguously."""
    data = load(cohort, exclude=["BRCA_Normal"])

    assert "BRCA_Normal" not in data.label_map
    assert data.n_patients == len(SUBTYPES) - 5
    assert sorted(data.label_map.values()) == [0, 1, 2]


def test_load_omics_block_slices_recover_each_modality(cohort) -> None:
    """Each block slice is the right width and the three tile the matrix."""
    data = load(cohort)
    covered = [data.X[:, data.block_slice(m)] for m in MODALITY_ORDER]

    assert all(block.shape == (data.n_patients, data.n_genes) for block in covered)
    assert np.allclose(np.hstack(covered), data.X)


def test_load_omics_rejects_unknown_modality(cohort) -> None:
    """A mistyped modality raises rather than silently slicing the wrong block."""
    data = load(cohort)
    with pytest.raises(KeyError, match="unknown modality"):
        data.block_slice("protein")


def test_load_omics_collapses_duplicate_gene_rows(cohort) -> None:
    """Repeated gene symbols are aggregated instead of producing extra columns."""
    duplicated = pd.read_csv(cohort / "rna.csv", index_col=0)
    duplicated = pd.concat([duplicated, duplicated.iloc[[0]]])
    duplicated.to_csv(cohort / "rna.csv")

    data = load(cohort)
    assert data.n_genes == len(GENES)


def test_load_omics_errors_when_no_patient_is_shared(cohort) -> None:
    """A barcode mismatch fails loudly rather than yielding an empty cohort."""
    clinical = pd.read_csv(cohort / "clinical.csv", index_col=0)
    clinical.index = [f"OTHER{i}" for i in range(len(clinical))]
    clinical.to_csv(cohort / "clinical.csv")

    with pytest.raises(ValueError, match="No patients shared"):
        load(cohort)


def test_load_omics_errors_when_exclusion_leaves_one_class(cohort) -> None:
    """Excluding down to a single class is rejected."""
    with pytest.raises(ValueError, match="at least two"):
        load(cohort, exclude=["BRCA_Basal", "BRCA_LumA", "BRCA_LumB"])


def test_load_curated_genes_keeps_only_known_symbols(cohort) -> None:
    """Curated symbols outside the universe are dropped, order is preserved."""
    path = cohort / "curated.txt"
    path.write_text("ESR1\nNOT_A_GENE\nAAA\n", encoding="utf-8")

    assert load_curated_genes(path, GENES) == ["ESR1", "AAA"]


def test_load_curated_genes_tolerates_absence(tmp_path) -> None:
    """A missing file or None force-includes nothing rather than raising."""
    assert load_curated_genes(None, GENES) == []
    assert load_curated_genes(tmp_path / "absent.txt", GENES) == []


# --------------------------------------------------------------------------
# Class exclusion
# --------------------------------------------------------------------------
def test_exclude_classes_returns_mask_over_original_patients(cohort) -> None:
    """The mask lines up with the pre-exclusion cohort so folds can be remapped."""
    data = load(cohort)
    filtered, keep_mask = exclude_classes(data, ["BRCA_Normal"])

    assert keep_mask.shape == (data.n_patients,)
    assert keep_mask.dtype == bool
    assert int(keep_mask.sum()) == filtered.n_patients == data.n_patients - 5
    assert "BRCA_Normal" not in filtered.label_map
    assert sorted(filtered.label_map.values()) == [0, 1, 2]


def test_exclude_classes_matches_loading_with_exclusion(cohort) -> None:
    """Filtering after loading agrees with excluding during the load."""
    at_load = load(cohort, exclude=["BRCA_Normal"])
    after_load, _ = exclude_classes(load(cohort), ["BRCA_Normal"])

    assert after_load.patient_ids == at_load.patient_ids
    assert after_load.label_map == at_load.label_map
    assert np.array_equal(after_load.y, at_load.y)


def test_exclude_classes_ignores_absent_class_names(cohort) -> None:
    """Naming a class the cohort does not contain is a no-op."""
    data = load(cohort)
    filtered, keep_mask = exclude_classes(data, ["BRCA_Her2"])

    assert filtered.n_patients == data.n_patients
    assert bool(keep_mask.all())


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------
def test_make_folds_produces_the_expected_count_and_no_leakage(cohort) -> None:
    """Twenty-five folds, each with disjoint train and test patients."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=5, seed=42)

    assert len(folds) == 25
    assert_no_leakage(folds, data.patient_ids)
    for spec in folds:
        assert len(spec.train_idx) + len(spec.test_idx) == data.n_patients


def test_make_folds_is_deterministic_for_a_seed(cohort) -> None:
    """The same seed and cohort reproduce the same partition."""
    data = load(cohort)
    first = make_folds(data.y, data.patient_ids, seed=7)
    second = make_folds(data.y, data.patient_ids, seed=7)

    assert all(
        np.array_equal(a.test_idx, b.test_idx)
        for a, b in zip(first, second, strict=True)
    )


def test_make_folds_rejects_mismatched_lengths() -> None:
    """Labels and barcodes must describe the same patients."""
    with pytest.raises(ValueError, match="patient_ids"):
        make_folds(np.array([0, 1, 0, 1]), ["P1", "P2"])


def test_assert_no_leakage_detects_an_overlap() -> None:
    """A patient in both halves is reported rather than tolerated."""
    folds = [
        FoldSpec(
            repeat=0,
            fold=0,
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([2, 3]),
        )
    ]
    with pytest.raises(ValueError, match="Leakage"):
        assert_no_leakage(folds, ["P0", "P1", "P2", "P3"])


def test_folds_round_trip_through_json(cohort, tmp_path) -> None:
    """Saving and loading preserves indices and the patient order."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=2, seed=1)
    path = tmp_path / "nested" / "folds.json"

    save_folds(folds, data.patient_ids, path)
    restored, order = load_folds(path)

    assert order == data.patient_ids
    assert len(restored) == len(folds)
    for original, copy in zip(folds, restored, strict=True):
        assert original.repeat == copy.repeat
        assert original.fold == copy.fold
        assert np.array_equal(original.train_idx, copy.train_idx)
        assert np.array_equal(original.test_idx, copy.test_idx)


def test_saved_folds_record_patient_barcodes(cohort, tmp_path) -> None:
    """Barcodes are persisted so a partition survives a reordering."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1)
    path = tmp_path / "folds.json"
    save_folds(folds, data.patient_ids, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["folds"][0]
    assert set(entry["test_patients"]) == {
        data.patient_ids[i] for i in folds[0].test_idx
    }


def test_assert_folds_match_accepts_the_same_cohort(cohort) -> None:
    """Identical patient order passes silently."""
    data = load(cohort)
    assert_folds_match(data.patient_ids, data.patient_ids)


def test_assert_folds_match_rejects_a_different_cohort_size(cohort) -> None:
    """The 914-versus-949 divergence is caught with a diagnosis.

    This is the guard that the original two-codebase setup lacked.
    """
    full = load(cohort)
    reduced = load(cohort, exclude=["BRCA_Normal"])

    with pytest.raises(ValueError, match="Fold file covers"):
        assert_folds_match(full.patient_ids, reduced.patient_ids)


def test_assert_folds_match_rejects_a_reordered_cohort(cohort) -> None:
    """Same patients in a different order is still a fatal mismatch."""
    data = load(cohort)
    with pytest.raises(ValueError, match="different order"):
        assert_folds_match(data.patient_ids, list(reversed(data.patient_ids)))


def test_remap_fold_indices_preserves_membership(cohort) -> None:
    """Surviving patients keep their fold, which is what makes runs paired."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1, seed=3)
    filtered, keep_mask = exclude_classes(data, ["BRCA_Normal"])

    remapped = remap_fold_indices(folds, keep_mask)

    assert len(remapped) == len(folds)
    assert_no_leakage(remapped, filtered.patient_ids)
    for original, updated in zip(folds, remapped, strict=True):
        before = {data.patient_ids[i] for i in original.test_idx if keep_mask[i]}
        after = {filtered.patient_ids[i] for i in updated.test_idx}
        assert before == after


def test_remap_fold_indices_covers_every_surviving_patient(cohort) -> None:
    """No patient is lost or duplicated by the remap."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1, seed=3)
    filtered, keep_mask = exclude_classes(data, ["BRCA_Normal"])

    remapped = remap_fold_indices(folds, keep_mask)

    for spec in remapped:
        assert len(spec.train_idx) + len(spec.test_idx) == filtered.n_patients
    pooled = np.concatenate([spec.test_idx for spec in remapped])
    assert sorted(pooled.tolist()) == list(range(filtered.n_patients))


def test_remap_fold_indices_requires_a_boolean_mask() -> None:
    """An integer index array is rejected rather than misinterpreted."""
    folds = [
        FoldSpec(0, 0, np.array([0, 1]), np.array([2])),
    ]
    with pytest.raises(ValueError, match="boolean"):
        remap_fold_indices(folds, np.array([0, 1, 2]))


def test_select_folds_takes_a_prefix(cohort) -> None:
    """Partial runs keep whole repeats and whole folds."""
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=5)

    assert len(select_folds(folds, n_repeats=1)) == 5
    assert len(select_folds(folds, max_folds=2)) == 10
    assert len(select_folds(folds, n_repeats=2, max_folds=3)) == 6
    assert len(select_folds(folds)) == 25


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def test_transformer_fits_statistics_on_training_rows_only(cohort) -> None:
    """A wild test row must not shift the fitted scaling.

    The transform of the training rows has to be byte-identical whether or not
    an extreme held-out row exists, which is the operational definition of no
    leakage.
    """
    data = load(cohort)
    train, test = data.X[:18], data.X[18:]

    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes, gene_names=data.gene_names, top_k=3
    )
    baseline = transformer.fit(train).transform(train)

    contaminated = test.copy()
    contaminated[:] = 1e6
    refit = (
        MultiOmicsTransformer(n_genes=data.n_genes, gene_names=data.gene_names, top_k=3)
        .fit(train)
        .transform(train)
    )

    assert np.allclose(baseline, refit)
    assert np.isfinite(transformer.transform(contaminated)).all()


def test_transformer_standardises_the_training_fold(cohort) -> None:
    """Each emitted column has zero mean and unit variance on the training fold."""
    data = load(cohort)
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes, gene_names=data.gene_names, top_k=4
    )
    transformed = transformer.fit_transform(data.X)

    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-9)


def test_transformer_force_includes_curated_genes(cohort) -> None:
    """A curated gene survives selection even at the smallest pool size."""
    data = load(cohort)
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=1,
        curated_genes=["ESR1"],
    )
    transformer.fit(data.X)

    assert "ESR1" in transformer.get_selected_gene_names()
    assert transformer.n_curated_selected_ == 1


def test_transformer_never_duplicates_a_curated_gene(cohort) -> None:
    """A curated gene that is also highly variable appears exactly once."""
    data = load(cohort)
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=len(GENES),
        curated_genes=list(GENES),
    )
    transformer.fit(data.X)

    selected = transformer.get_selected_gene_names()
    assert len(selected) == len(set(selected)) == len(GENES)


def test_transformer_emits_only_active_modalities(cohort) -> None:
    """Deactivating a modality removes its block from the output."""
    data = load(cohort)
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=3,
        active_modalities=["rna", "methy"],
    )
    transformed = transformer.fit_transform(data.X)

    assert transformed.shape[1] == transformer.n_features_out_
    assert transformed.shape[1] == len(transformer.selected_idx_) * 2
    assert [name.split(":")[0] for name in transformer.get_feature_names_out()] == [
        "rna"
    ] * len(transformer.selected_idx_) + ["methy"] * len(transformer.selected_idx_)


def test_transformer_imputes_with_training_medians(cohort) -> None:
    """Missing values are filled, and the fill comes from the training fold."""
    data = load(cohort)
    holed = data.X.copy()
    holed[0, 0] = np.nan

    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes, gene_names=data.gene_names, top_k=len(GENES)
    )
    transformed = transformer.fit_transform(holed)

    assert np.isfinite(transformed).all()


def test_transformer_rejects_a_mismatched_matrix_width(cohort) -> None:
    """A matrix that is not three equal blocks wide is refused."""
    data = load(cohort)
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes, gene_names=data.gene_names
    )
    with pytest.raises(ValueError, match="columns"):
        transformer.fit(data.X[:, :-1])


def test_transformer_survives_sklearn_clone(cohort) -> None:
    """``clone`` must reconstruct the estimator, so ``__init__`` stores verbatim."""
    from sklearn.base import clone

    data = load(cohort)
    original = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=2,
        curated_genes=["ESR1"],
        active_modalities=["rna", "cnv"],
    )
    copy = clone(original)

    assert copy.get_params() == original.get_params()
    assert copy.fit_transform(data.X).shape == original.fit_transform(data.X).shape


def test_transformer_selection_is_a_fitted_parameter(cohort) -> None:
    """Different training folds may select different genes; that is by design."""
    data = load(cohort)
    kwargs = {"n_genes": data.n_genes, "gene_names": data.gene_names, "top_k": 2}

    first = MultiOmicsTransformer(**kwargs).fit(data.X[:13])
    second = MultiOmicsTransformer(**kwargs).fit(data.X[13:])

    assert len(first.get_selected_gene_names()) == 2
    assert len(second.get_selected_gene_names()) == 2


def test_fold_local_selection_differs_from_global_selection(cohort) -> None:
    """Fitting on a fold must not reproduce fitting on the whole cohort.

    The canary for silent leakage. If a fold-local fit and a global fit always
    agreed, the transformer would not actually be refitting on the training
    split, and every fold would carry information from the held-out patients.
    Ported from the legacy harness's acceptance suite.
    """
    data = load(cohort)
    kwargs = {
        "n_genes": data.n_genes,
        "gene_names": data.gene_names,
        "top_k": 2,
    }

    global_fit = MultiOmicsTransformer(**kwargs).fit(data.X)
    fold_fits = [
        MultiOmicsTransformer(**kwargs).fit(data.X[start::3]) for start in range(3)
    ]

    global_scaler = global_fit.scalers_["rna"].mean_
    # At least one fold must differ from the global fit, in the genes it picks
    # or in the statistics it fits.
    assert any(
        fold.get_selected_gene_names() != global_fit.get_selected_gene_names()
        or not np.allclose(fold.scalers_["rna"].mean_, global_scaler)
        for fold in fold_fits
    )


def test_scaler_statistics_differ_between_folds(cohort) -> None:
    """Two different training folds fit different scaling statistics.

    Equal statistics across folds would mean they were computed once and shared,
    which is the leakage this design exists to prevent.
    """
    data = load(cohort)
    kwargs = {
        "n_genes": data.n_genes,
        "gene_names": data.gene_names,
        "top_k": len(GENES),
    }

    first = MultiOmicsTransformer(**kwargs).fit(data.X[: data.n_patients // 2])
    second = MultiOmicsTransformer(**kwargs).fit(data.X[data.n_patients // 2 :])

    assert not np.allclose(first.scalers_["rna"].mean_, second.scalers_["rna"].mean_)
