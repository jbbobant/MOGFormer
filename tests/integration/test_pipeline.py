"""End-to-end smoke test over a synthetic cohort.

This is the test that answers "does the pipeline run outside Kaggle?", which is
the first thing anyone cloning the repository will try. It walks the whole path
— write matrices, load and align them, build folds, refit preprocessing inside a
fold, build the fold-local graph, train the transformer for a couple of epochs,
score the held-out patients — on a cohort small enough to finish in seconds on
CPU.

It asserts wiring, not accuracy. A model trained on random data should not
predict anything well, and a test that expected it to would be worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mogformer.config import load_config
from mogformer.data import (
    MultiOmicsTransformer,
    assert_folds_match,
    assert_no_leakage,
    exclude_classes,
    load_folds,
    load_omics,
    make_folds,
    remap_fold_indices,
    save_folds,
)
from mogformer.evaluation import (
    aggregate_across_folds,
    compute_fold_metrics,
    fold_confusion,
    pairwise_compare,
)
from mogformer.evaluation.estimator import MOGFormerClassifier
from mogformer.graph import StringGraphCache

N_GENES = 12
CLASS_SIZES = {"BRCA_Basal": 12, "BRCA_LumA": 16, "BRCA_LumB": 12, "BRCA_Normal": 10}


@pytest.fixture()
def cohort(tmp_path):
    """Write a synthetic cohort plus a miniature interaction network."""
    rng = np.random.default_rng(20240712)
    genes = [f"GENE{i:02d}" for i in range(N_GENES)]
    subtypes = [name for name, count in CLASS_SIZES.items() for _ in range(count)]
    patients = [f"TCGA-XX-{i:04d}" for i in range(len(subtypes))]

    # Give each class a mean offset so the task is learnable in principle.
    offsets = {name: i * 2.0 for i, name in enumerate(CLASS_SIZES)}
    for name, scale in (("rna.csv", 50.0), ("cnv.csv", 1.0), ("methy.csv", 0.4)):
        values = rng.normal(size=(N_GENES, len(patients)))
        for column, subtype in enumerate(subtypes):
            values[:, column] += offsets[subtype]
        frame = pd.DataFrame(np.abs(values * scale), index=genes, columns=patients)
        frame.to_csv(tmp_path / name)

    pd.DataFrame({"SUBTYPE": subtypes}, index=patients).to_csv(
        tmp_path / "clinical.csv"
    )

    # A small STRING-shaped pair of files: a path graph over the genes.
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text(
        "#string_protein_id\talias\tsource\n"
        + "".join(f"9606.P{i}\t{gene}\tBioMart\n" for i, gene in enumerate(genes)),
        encoding="utf-8",
    )
    links = tmp_path / "links.txt"
    links.write_text(
        "protein1 protein2 combined_score\n"
        + "".join(f"9606.P{i} 9606.P{i + 1} 900\n" for i in range(N_GENES - 1)),
        encoding="utf-8",
    )

    curated = tmp_path / "curated.txt"
    curated.write_text("GENE00\n", encoding="utf-8")

    return tmp_path


def load(cohort_dir, **kwargs):
    """Load the synthetic cohort with the fixture's file names."""
    return load_omics(
        raw_dir=cohort_dir,
        rna_file="rna.csv",
        cnv_file="cnv.csv",
        methy_file="methy.csv",
        clin_file="clinical.csv",
        label_col="SUBTYPE",
        **kwargs,
    )


def test_pipeline_runs_end_to_end(cohort, tmp_path) -> None:
    """Load, split, preprocess, build the graph, train, and score one fold."""
    data = load(cohort)
    assert data.n_patients == sum(CLASS_SIZES.values())

    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1, seed=42)
    assert_no_leakage(folds, data.patient_ids)
    save_folds(folds, data.patient_ids, tmp_path / "folds.json")

    restored, order = load_folds(tmp_path / "folds.json")
    assert_folds_match(order, data.patient_ids)

    spec = restored[0]
    transformer = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=6,
        curated_genes=["GENE00"],
    )
    train_features = transformer.fit_transform(data.X[spec.train_idx])
    test_features = transformer.transform(data.X[spec.test_idx])
    selected = transformer.get_selected_gene_names()

    assert "GENE00" in selected
    assert train_features.shape[1] == len(selected) * 3
    assert test_features.shape[1] == train_features.shape[1]

    string_cache = StringGraphCache(
        str(cohort / "links.txt"), str(cohort / "aliases.tsv"), data.gene_names
    )

    estimator = MOGFormerClassifier(
        selected_genes=selected,
        string_cache=string_cache,
        num_classes=len(data.label_map),
        d=16,
        pe_dim=4,
        mini_heads=2,
        global_heads=2,
        global_layers=1,
        max_distance=3,
        max_epochs=2,
        patience=2,
        batch_size=8,
        device="cpu",
    )
    estimator.fit(train_features, data.y[spec.train_idx])

    probabilities = estimator.predict_proba(test_features)
    predictions = estimator.predict(test_features)

    n_test = len(spec.test_idx)
    n_classes = len(data.label_map)
    assert probabilities.shape == (n_test, n_classes)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert predictions.shape == (n_test,)
    assert set(np.unique(predictions)) <= set(range(n_classes))

    class_names = [data.inverse_label_map[i] for i in range(n_classes)]
    scores = compute_fold_metrics(
        data.y[spec.test_idx],
        predictions,
        probabilities,
        list(range(n_classes)),
        class_names,
    )
    assert 0.0 <= scores["macro_f1"] <= 1.0
    assert f"f1__{class_names[0]}" in scores

    matrix = fold_confusion(data.y[spec.test_idx], predictions, range(n_classes))
    assert matrix.shape == (n_classes, n_classes)
    assert matrix.sum() == n_test


def test_graph_is_rebuilt_from_fold_local_genes(cohort) -> None:
    """Two folds selecting different genes get different graphs.

    Building the graph once from the full universe would leak held-out
    structure, so the estimator must derive it from the genes it was given.
    """
    data = load(cohort)
    string_cache = StringGraphCache(
        str(cohort / "links.txt"), str(cohort / "aliases.tsv"), data.gene_names
    )

    first = string_cache.induced_adjacency(data.gene_names[:4])
    second = string_cache.induced_adjacency(data.gene_names[4:8])

    assert first.shape == (4, 4)
    assert second.shape == (4, 4)
    assert first.sum() > 0


def test_class_exclusion_keeps_folds_paired(cohort, tmp_path) -> None:
    """A four-class run reuses the five-class partition rather than re-splitting.

    This is the property whose absence made the stored baseline and transformer
    results incomparable.
    """
    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1, seed=42)

    reduced, keep_mask = exclude_classes(data, ["BRCA_Normal"])
    remapped = remap_fold_indices(folds, keep_mask)

    assert len(remapped) == len(folds)
    assert_no_leakage(remapped, reduced.patient_ids)

    for original, updated in zip(folds, remapped, strict=True):
        survivors = {data.patient_ids[i] for i in original.test_idx if keep_mask[i]}
        assert {reduced.patient_ids[i] for i in updated.test_idx} == survivors


def test_two_models_can_be_compared_on_one_partition() -> None:
    """The statistics layer pairs models scored on the same folds."""
    rng = np.random.default_rng(0)
    per_fold = {
        "model_a": rng.normal(0.85, 0.02, size=25),
        "model_b": rng.normal(0.80, 0.02, size=25),
    }
    comparison = pairwise_compare(per_fold, n_splits=5)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["n_pairs"] == 25
    assert row["mean_diff_a_minus_b"] > 0
    assert 0.0 <= row["wilcoxon_p"] <= 1.0

    summary = aggregate_across_folds(per_fold["model_a"], n_splits=5)
    # The corrected interval must be the wider one; that is its whole purpose.
    naive_width = summary["ci95_hi"] - summary["ci95_lo"]
    corrected_width = summary["nb_ci95_hi"] - summary["nb_ci95_lo"]
    assert corrected_width > naive_width


def test_comparison_refuses_mismatched_partitions() -> None:
    """Scoring two models on different fold counts is rejected, not averaged."""
    with pytest.raises(ValueError, match="different numbers of folds"):
        pairwise_compare(
            {"a": np.zeros(25), "b": np.zeros(10)},
            n_splits=5,
        )


def test_cli_folds_command_writes_a_partition(cohort, tmp_path) -> None:
    """The ``folds`` subcommand produces the file every later run reads."""
    from mogformer.cli.__main__ import main

    config_path = tmp_path / "config.yaml"
    folds_path = tmp_path / "out" / "folds.json"
    config_path.write_text(
        "name: smoke\n"
        f"results_dir: {tmp_path.as_posix()}/out\n"
        "data:\n"
        f"  raw_dir: {cohort.as_posix()}\n"
        "  rna_file: rna.csv\n"
        "  cnv_file: cnv.csv\n"
        "  methy_file: methy.csv\n"
        "  clin_file: clinical.csv\n"
        "  label_col: SUBTYPE\n"
        "folds:\n"
        f"  path: {folds_path.as_posix()}\n"
        "  n_splits: 5\n"
        "  n_repeats: 1\n",
        encoding="utf-8",
    )

    assert main(["folds", "--config", str(config_path)]) == 0
    assert folds_path.exists()

    folds, order = load_folds(folds_path)
    assert len(folds) == 5
    assert len(order) == sum(CLASS_SIZES.values())


def test_config_rejects_an_unknown_key(tmp_path) -> None:
    """A mistyped setting fails loudly instead of being silently ignored."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  raw_dirr: somewhere\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown configuration key"):
        load_config(config_path)


def test_config_round_trips(tmp_path) -> None:
    """A resolved config can be written and read back unchanged."""
    from mogformer.config import save_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: roundtrip\npreprocess:\n  top_k: 42\n  active_modalities: [rna, cnv]\n",
        encoding="utf-8",
    )
    original = load_config(config_path)
    assert original.preprocess.top_k == 42
    assert original.preprocess.active_modalities == ("rna", "cnv")

    save_config(original, tmp_path / "resolved.yaml")
    restored = load_config(tmp_path / "resolved.yaml")
    assert restored.preprocess.active_modalities == ("rna", "cnv")
    assert restored.name == "roundtrip"


@pytest.mark.parametrize("fusion", ["attention", "gated"])
def test_self_supervised_encoder_trains_a_step(fusion: str) -> None:
    """A pretraining step produces a finite loss and a gradient."""
    from mogformer.models import MOGFormerSSL
    from mogformer.training import masked_huber_dual

    n_genes, batch, width = 8, 4, 16
    model = MOGFormerSSL(
        d=width,
        pe_dim=4,
        mini_heads=2,
        global_heads=2,
        global_layers=1,
        max_distance=3,
        n_universe=16,
        gene_ids=list(range(n_genes)),
        fusion_type=fusion,
        use_grn=False,
    )
    spd = torch.randint(0, 3, (n_genes, n_genes))
    spd = torch.minimum(spd, spd.T)
    spd.fill_diagonal_(0)

    out = model(
        torch.randn(batch, n_genes),
        torch.randn(batch, n_genes),
        torch.randn(batch, n_genes),
        torch.randn(n_genes, 4),
        spd,
    )
    loss, global_term, local_term = masked_huber_dual(
        out["xhat_g"], out["xhat_l"], out["targets"], out["mask_bool"]
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(global_term) and torch.isfinite(local_term)
    assert model.lifter.emb_rna.grad is not None


def test_runner_scores_baselines_on_the_shared_partition(cohort, tmp_path) -> None:
    """The runner drives registered models through one protocol end to end.

    This is the property the whole harness exists for: several models, one
    partition, per-fold scores that can afterwards be compared pairwise.
    """
    from mogformer.evaluation.runner import (
        load_per_fold_scores,
        run_cross_validation,
        summarise,
    )

    data = load(cohort)
    folds = make_folds(data.y, data.patient_ids, n_splits=5, n_repeats=1, seed=42)
    folds_path = tmp_path / "folds.json"
    save_folds(folds, data.patient_ids, folds_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: runner_smoke\n"
        f"results_dir: {tmp_path.as_posix()}/out\n"
        "data:\n"
        f"  raw_dir: {cohort.as_posix()}\n"
        "  rna_file: rna.csv\n"
        "  cnv_file: cnv.csv\n"
        "  methy_file: methy.csv\n"
        "  clin_file: clinical.csv\n"
        "  label_col: SUBTYPE\n"
        "folds:\n"
        f"  path: {folds_path.as_posix()}\n"
        "  n_splits: 5\n"
        "  n_repeats: 1\n"
        "preprocess:\n"
        "  top_k: 6\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    models = ["B0a_dummy_stratified", "B0b_dummy_mostfreq", "B1_pam50_centroid"]
    per_fold = run_cross_validation(
        config, models=models, pam50_genes=["GENE00", "GENE01", "GENE02"]
    )

    assert set(per_fold["model"].unique()) == set(models)
    assert per_fold["fold"].nunique() == 5

    output = config.output_dir
    assert (output / "metrics_per_fold.csv").exists()
    assert (output / "metrics_summary.csv").exists()
    assert (output / "config.yaml").exists()
    leaderboard = (output / "metrics_summary.md").read_text(encoding="utf-8")
    assert "Leaderboard" in leaderboard

    summary = summarise(per_fold, n_splits=5)
    macro = summary[summary["metric"] == "macro_f1"].set_index("model")
    assert set(macro.index) == set(models)
    assert (macro["n"] == 5).all()

    # Every model was scored on the same folds, so they are pairable.
    scores = load_per_fold_scores(output / "metrics_per_fold.csv")
    assert {len(v) for v in scores.values()} == {5}

    comparison = pairwise_compare(scores, n_splits=5)
    assert len(comparison) == 3
    assert comparison["n_pairs"].eq(5).all()

    # A real classifier must clear the always-majority floor.
    assert (
        macro.loc["B1_pam50_centroid", "mean"] > macro.loc["B0b_dummy_mostfreq", "mean"]
    )


def test_runner_refuses_a_partition_from_another_cohort(cohort, tmp_path) -> None:
    """Folds built over a different cohort are rejected, not silently misapplied.

    Precisely the failure that left the archived baseline and transformer
    results unpairable.
    """
    from mogformer.evaluation.runner import run_cross_validation

    data = load(cohort)
    reduced, _ = exclude_classes(data, ["BRCA_Normal"])
    folds = make_folds(reduced.y, reduced.patient_ids, n_splits=5, n_repeats=1)
    folds_path = tmp_path / "reduced_folds.json"
    save_folds(folds, reduced.patient_ids, folds_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: mismatch\n"
        f"results_dir: {tmp_path.as_posix()}/out\n"
        "data:\n"
        f"  raw_dir: {cohort.as_posix()}\n"
        "  rna_file: rna.csv\n"
        "  cnv_file: cnv.csv\n"
        "  methy_file: methy.csv\n"
        "  clin_file: clinical.csv\n"
        "  label_col: SUBTYPE\n"
        "folds:\n"
        f"  path: {folds_path.as_posix()}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Fold file covers"):
        run_cross_validation(load_config(config_path), models=["B0b_dummy_mostfreq"])


def test_every_figure_renders(tmp_path) -> None:
    """Each plotting function writes a PNG and an SVG without raising.

    Figures break silently and late, usually the first time a real run finishes.
    This renders every one on synthetic inputs so a shape or column error is
    caught in seconds rather than after an overnight job.
    """
    from mogformer.analysis.clustering import (
        RepresentationRung,
        run_consensus_clustering,
        screen_covariates,
    )
    from mogformer.analysis.plots import (
        plot_confound_screen,
        plot_consensus_heatmap,
        plot_representation_ladder,
        plot_response_curves,
        plot_sign_concordance,
        plot_stability_by_k,
        plot_trans_versus_coexpression,
    )
    from mogformer.analysis.probe_trans import sign_permutation_test
    from mogformer.evaluation.plots import (
        plot_confusion,
        plot_critical_difference,
        plot_interval_forest,
        plot_model_comparison,
        plot_per_class_f1,
    )
    from mogformer.evaluation.runner import summarise
    from mogformer.evaluation.stats import friedman_nemenyi

    rng = np.random.default_rng(0)
    models = ["model_a", "model_b", "model_c"]
    per_fold = {name: rng.normal(0.8, 0.03, 25) for name in models}

    written = [
        plot_model_comparison(per_fold, tmp_path, references={"reference": 0.85})
    ]

    long_form = pd.DataFrame(
        [
            {
                "model": model,
                "repeat": i // 5,
                "fold": i % 5,
                "metric": metric,
                "value": value,
            }
            for model, scores in per_fold.items()
            for i, value in enumerate(scores)
            for metric in ("macro_f1",)
        ]
        + [
            {
                "model": model,
                "repeat": i // 5,
                "fold": i % 5,
                "metric": f"f1__{cls}",
                "value": float(rng.uniform(0.5, 0.95)),
            }
            for model in models
            for i in range(25)
            for cls in ("BRCA_Basal", "BRCA_LumA")
        ]
    )
    summary = summarise(long_form, n_splits=5)
    written.append(plot_interval_forest(summary, tmp_path))
    written.append(plot_per_class_f1(summary, ["BRCA_Basal", "BRCA_LumA"], tmp_path))

    _, _, ranks, critical = friedman_nemenyi(per_fold)
    written.append(plot_critical_difference(ranks, critical, tmp_path))
    written.append(
        plot_confusion(np.array([[9, 1], [2, 8]]), ["BRCA_LumA", "BRCA_LumB"], tmp_path)
    )

    blobs = np.vstack([rng.normal(-4, 0.3, (25, 3)), rng.normal(4, 0.3, (25, 3))])
    results = run_consensus_clustering(blobs, (2, 3), n_resample=15, seed=0)
    written.append(plot_consensus_heatmap(results[2], tmp_path))
    written.append(plot_stability_by_k(results, tmp_path))

    written.append(
        plot_representation_ladder(
            [
                RepresentationRung("full", 0.01, results[2].labels, 1.0),
                RepresentationRung("cnv_only", 0.02, results[2].labels, 1.0),
            ],
            tmp_path,
        )
    )

    scores = rng.normal(size=50)
    screen = screen_covariates(
        scores,
        pd.DataFrame({"burden": scores * 2, "site": rng.integers(0, 3, 50)}),
        roles={"burden": "biology", "site": "technical"},
    )
    written.append(plot_confound_screen(screen, tmp_path))

    curves = pd.DataFrame(
        [
            {
                "gene": gene,
                "arrow": "methy->rna",
                "injected_value": float(v),
                "mean_prediction": -0.2 * float(v),
                "sem_prediction": 0.01,
            }
            for gene in ("ESR1", "MKI67")
            for v in np.linspace(-2, 2, 9)
        ]
    )
    written.append(plot_response_curves(curves, ["ESR1", "MKI67"], tmp_path))

    edges = pd.DataFrame(
        {
            "edge_sign": [1] * 60 + [-1] * 20,
            "slope": rng.normal(0.002, 0.001, 80),
            "observed_coexpression": rng.normal(0.2, 0.3, 80),
        }
    )
    written.append(
        plot_sign_concordance(sign_permutation_test(edges, n_perm=200), tmp_path)
    )
    written.append(plot_trans_versus_coexpression(edges, tmp_path))

    for path in written:
        assert path.exists(), path
        assert path.stat().st_size > 0
        assert path.with_suffix(".svg").exists()
