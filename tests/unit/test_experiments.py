"""The experiment layer: path rebasing, pairing rules, checkpoints.

These tests guard the places where a run could quietly stop being the run the
repository describes — a configuration rebased onto a bundle in a way that also
changed a setting, two models paired across different partitions, a checkpoint
whose renamed tensors were silently reinitialised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import mogformer.cli.experiments as experiment_driver
from mogformer.config import load_config
from mogformer.experiments.metabric import permute_regulatory_graph
from mogformer.experiments.paths import (
    BundleLayout,
    provenance_record,
    rebase_config,
    save_json,
)
from mogformer.experiments.sign_test import load_edge_table, rerun_sign_test
from mogformer.experiments.statistics import (
    align_scores,
    combine_per_fold,
    run_paired_statistics,
)
from mogformer.models.checkpoints import (
    load_encoder_state,
    read_checkpoint,
    rename_legacy_keys,
)


def write_bundle(root):
    """Write a bundle with every file the layout requires."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "folds.json").write_text("{}", encoding="utf-8")
    (root / "string_edges.tsv").write_text("gene_a\tgene_b\n", encoding="utf-8")
    (root / "curated_genes.txt").write_text("ESR1\n", encoding="utf-8")
    (root / "cohort_beta.npz").write_bytes(b"")
    (root / "pam50_genes.txt").write_text("ESR1\nERBB2\n", encoding="utf-8")
    (root / "collectri_signed_edges.tsv").write_text("tf\ttarget\n", encoding="utf-8")
    return root


def write_config(path, **overrides):
    """Write a minimal configuration naming repository-relative paths."""
    body = {
        "name": "unit",
        "results_dir": "results/runs",
        "data": {
            "raw_dir": "data",
            "rna_file": "raw/rna.csv",
            "cnv_file": "raw/cnv.csv",
            "methy_file": "raw/methy.csv",
            "clin_file": "clean/clinical.csv",
            "label_col": "SUBTYPE",
            "exclude_classes": ["BRCA_Normal"],
            "curated_genes_file": "data/raw/CGenes.txt",
        },
        "folds": {"path": "data/clean/folds.json", "n_splits": 5, "n_repeats": 5},
        "preprocess": {"top_k": 250},
        "graph": {"max_distance": 10, "pe_dim": 32},
        "model": {"d": 128, "global_layers": 1, **overrides.get("model", {})},
        "train": {"max_epochs": 300, "patience": 40, "focal_gamma": 2.0},
    }
    import yaml

    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


# --- bundle layout ------------------------------------------------------------


def test_an_incomplete_bundle_names_everything_it_is_missing(tmp_path) -> None:
    """A partial bundle fails at once, listing all of it, not one file."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "folds.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as error:
        BundleLayout(root).require()

    message = str(error.value)
    assert "cohort" in message
    assert "edges" in message
    assert "curated genes" in message
    assert "folds" not in message.split("missing:")[1]


def test_the_panel_is_optional(tmp_path) -> None:
    """A bundle without the panel still runs; the centroid model just cannot."""
    layout = BundleLayout(write_bundle(tmp_path / "bundle"))
    assert layout.read_pam50() == ["ESR1", "ERBB2"]

    (layout.root / "pam50_genes.txt").unlink()
    assert layout.read_pam50() == []


# --- rebasing -----------------------------------------------------------------


def test_rebasing_moves_paths_and_nothing_else(tmp_path) -> None:
    """The bundle changes where a run reads, never what it computes.

    This is the property that lets a bundled run be compared to a raw one. If
    rebasing could touch the gene budget, the fold count or the cohort
    composition, a bundled run would be a different experiment wearing the same
    configuration's name.
    """
    bundle = BundleLayout(write_bundle(tmp_path / "bundle"))
    original = load_config(write_config(tmp_path / "config.yaml"))

    rebased = rebase_config(original, bundle, tmp_path / "out")

    assert rebased.data.raw_dir == str(bundle.root)
    assert rebased.folds.path == str(bundle.root / "folds.json")
    assert rebased.graph.edges_file == str(bundle.root / "string_edges.tsv")
    assert rebased.results_dir == str(tmp_path / "out")

    # Everything that decides the experiment is untouched.
    assert rebased.preprocess.top_k == original.preprocess.top_k
    assert rebased.folds.n_splits == original.folds.n_splits
    assert rebased.folds.n_repeats == original.folds.n_repeats
    assert rebased.data.exclude_classes == original.data.exclude_classes
    assert rebased.data.label_col == original.data.label_col
    assert rebased.model == original.model
    assert rebased.train == original.train


def test_rebasing_leaves_the_original_alone(tmp_path) -> None:
    """Two runs off one loaded configuration must not contaminate each other."""
    bundle = BundleLayout(write_bundle(tmp_path / "bundle"))
    original = load_config(write_config(tmp_path / "config.yaml"))
    before = original.data.raw_dir

    rebase_config(original, bundle, tmp_path / "out")

    assert original.data.raw_dir == before


def test_rebasing_redirects_pretrained_vectors_to_the_bundle(tmp_path) -> None:
    """A run asking for frozen vectors reads the bundle's compressed copy."""
    bundle = BundleLayout(write_bundle(tmp_path / "bundle"))
    (bundle.root / "gene_embeddings.npz").write_bytes(b"")
    path = write_config(
        tmp_path / "config.yaml",
        model={
            "gene_id_embedding": True,
            "pretrained_emb_file": "data/raw/FROGS-ARCHS4/emb.csv",
            "pretrained_emb_symbol_file": "data/raw/FROGS-ARCHS4/symbols.txt",
            "pretrained_emb_adapter_rank": 32,
        },
    )

    rebased = rebase_config(load_config(path), bundle, tmp_path / "out")

    assert rebased.model.pretrained_emb_file == str(bundle.root / "gene_embeddings.npz")
    # The bundle is a single archive, so the separate symbol list is not needed.
    assert rebased.model.pretrained_emb_symbol_file is None
    assert rebased.model.pretrained_emb_adapter_rank == 32


def test_a_run_without_pretrained_vectors_is_not_given_any(tmp_path) -> None:
    """The bundle carrying an embedding table does not enable it."""
    bundle = BundleLayout(write_bundle(tmp_path / "bundle"))
    (bundle.root / "gene_embeddings.npz").write_bytes(b"")

    rebased = rebase_config(
        load_config(write_config(tmp_path / "config.yaml")), bundle, tmp_path / "out"
    )

    assert rebased.model.pretrained_emb_file is None
    assert rebased.model.gene_id_embedding is False


def test_provenance_records_the_host(tmp_path) -> None:
    """Every run says where it happened, so results can be told apart."""
    record = provenance_record({"experiment": "transformer_variants"})

    assert record["experiment"] == "transformer_variants"
    assert "timestamp_utc" in record
    assert "platform" in record
    assert "cuda_available" in record

    path = save_json(record, tmp_path / "nested" / "provenance.json")
    assert path.exists()


def test_save_json_handles_numpy_scalars(tmp_path) -> None:
    """Statistics come back as numpy types; writing them must not fail."""
    path = save_json(
        {"p": np.float64(0.031), "n": np.int64(25), "vector": np.arange(3)},
        tmp_path / "record.json",
    )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"p": pytest.approx(0.031), "n": 25, "vector": [0, 1, 2]}


# --- pairing rules ------------------------------------------------------------


def test_models_on_different_partitions_are_not_paired() -> None:
    """A short vector is dropped, never truncated to fit.

    Truncating would pair fold 3 of a 25-estimate run against fold 3 of a
    5-estimate one — different splits, so the difference means nothing. This
    is the same class of error as the 914-versus-949 divergence, at fold
    granularity.
    """
    scores = {
        "A": np.arange(25, dtype=float),
        "B": np.arange(25, dtype=float) + 1,
        "C": np.arange(5, dtype=float),
    }

    kept = align_scores(scores)

    assert set(kept) == {"A", "B"}
    assert all(len(vector) == 25 for vector in kept.values())


def test_a_lone_survivor_is_an_error() -> None:
    """One model left is not a comparison."""
    with pytest.raises(ValueError, match="at least two"):
        align_scores({"A": np.arange(25.0), "B": np.arange(5.0)})


def test_paired_statistics_write_every_artifact(tmp_path) -> None:
    """The comparison leaves tables, figures and a machine-readable record."""
    rng = np.random.default_rng(0)
    rows = []
    for model, centre in (("A", 0.80), ("B", 0.86), ("C", 0.84)):
        for repeat in range(5):
            for fold in range(5):
                rows.append(
                    {
                        "model": model,
                        "repeat": repeat,
                        "fold": fold,
                        "metric": "macro_f1",
                        "value": centre + rng.normal(scale=0.02),
                    }
                )
    per_fold = tmp_path / "metrics_per_fold.csv"
    pd.DataFrame(rows).to_csv(per_fold, index=False)

    record = run_paired_statistics(per_fold, tmp_path / "out", n_splits=5)

    assert record["n_models"] == 3
    assert record["n_estimates"] == 25
    assert (tmp_path / "out" / "pairwise_tests.csv").exists()
    assert (tmp_path / "out" / "average_ranks.csv").exists()
    assert (tmp_path / "out" / "paired_tests.json").exists()
    report = (tmp_path / "out" / "paired_tests.md").read_text(encoding="utf-8")
    assert "Friedman" in report
    # B was drawn highest, so it should rank best.
    ranks = record["average_ranks"]
    assert ranks["B"] < ranks["A"]


# --- the corrected sign test --------------------------------------------------


def make_edge_table(rate: float, majority: float, n: int = 400) -> pd.DataFrame:
    """Build a per-edge table with a chosen concordance and sign imbalance."""
    rng = np.random.default_rng(11)
    n_positive = round(majority * n)
    edge_sign = np.array([1] * n_positive + [-1] * (n - n_positive))
    concordant = rng.permutation(n) < round(rate * n)
    slope = np.where(concordant, edge_sign, -edge_sign) * rng.uniform(0.001, 0.01, n)
    return pd.DataFrame(
        {
            "tf": [f"TF{i % 20}" for i in range(n)],
            "target": [f"G{i}" for i in range(n)],
            "edge_sign": edge_sign,
            "is_edge": 1,
            "null": "real",
            "mask_mode": "rna_only",
            "slope": slope,
            "slope_lo": slope - 0.0005,
            "slope_hi": slope + 0.0005,
            "obs_coexpr": rng.uniform(-1, 1, n),
        }
    )


def test_legacy_column_names_are_translated(tmp_path) -> None:
    """Older sweeps wrote ``obs_coexpr``; the analysis expects the long name."""
    path = tmp_path / "edges.csv"
    make_edge_table(0.6, 0.77).to_csv(path, index=False)

    frame = load_edge_table(path)

    assert "observed_coexpression" in frame
    assert "obs_coexpr" not in frame


def test_a_table_missing_slopes_is_rejected(tmp_path) -> None:
    """Co-expression alone cannot be sign-tested, and says so."""
    path = tmp_path / "edges.csv"
    make_edge_table(0.6, 0.77).drop(columns=["slope"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing column"):
        load_edge_table(path)


def test_concordance_below_the_trivial_baseline_is_not_informative(tmp_path) -> None:
    """The headline correction: significance is not usefulness.

    Sixty per cent concordance on an edge set that is 77% activating loses to
    a constant predictor. Whatever the permutation p-value says, the model has
    not recovered direction, and the report must not be allowed to claim it.
    """
    path = tmp_path / "edges.csv"
    make_edge_table(0.60, 0.77).to_csv(path, index=False)

    record = rerun_sign_test(path, tmp_path / "out", "TCGA", n_permutations=500)
    sign = record["modes"]["rna_only"]["sign_concordance"]

    assert sign["trivial_majority_baseline"] == pytest.approx(0.77, abs=0.01)
    assert sign["beats_trivial_baseline"] is False
    assert sign["is_informative"] is False
    assert "**no**" in (tmp_path / "out" / "sign_tests.md").read_text(encoding="utf-8")


def test_a_genuine_signal_clears_both_bars(tmp_path) -> None:
    """A model that does recover direction is reported as informative."""
    path = tmp_path / "edges.csv"
    make_edge_table(0.95, 0.60).to_csv(path, index=False)

    record = rerun_sign_test(path, tmp_path / "out", "synthetic", n_permutations=500)
    sign = record["modes"]["rna_only"]["sign_concordance"]

    assert sign["rate"] > sign["trivial_majority_baseline"]
    assert sign["perm_p"] < 0.05
    assert sign["is_informative"] is True


# --- archived checkpoints -----------------------------------------------------


def test_legacy_parameter_names_are_translated() -> None:
    """Every rename the refactor introduced is written down, not discovered."""
    legacy = {
        "global_transformer.layers.0.attn.b_grn_act": torch.zeros(8),
        "global_transformer.layers.0.attn.b_grn_repr": torch.zeros(8),
        "masker.mod_w": torch.zeros(3),
        "decoder.ffn_g.0.weight": torch.zeros(4, 4),
        "decoder.lin_l.bias": torch.zeros(1),
        "lifter.emb_rna": torch.zeros(2),
    }

    renamed = rename_legacy_keys(legacy)

    assert set(renamed) == {
        "global_transformer.layers.0.attn.b_grn_activation",
        "global_transformer.layers.0.attn.b_grn_repression",
        "masker.modality_weights",
        "decoder.ffn_global.0.weight",
        "decoder.lin_local.bias",
        "lifter.emb_rna",
    }


def test_an_unmappable_checkpoint_refuses_to_load_silently() -> None:
    """A mismatch must raise, not leave half the network freshly initialised.

    PyTorch reports this as a list of missing keys, which is easy to skim past
    and leaves an encoder that is part-trained and part-random. Loading is
    strict so that the failure happens at load time.
    """
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="does not match"):
        load_encoder_state(model, {"nonexistent.weight": torch.zeros(2, 3)})


def test_a_matching_checkpoint_loads(tmp_path) -> None:
    """The happy path still works, and the weights actually arrive."""
    model = nn.Linear(3, 2)
    weights = {"weight": torch.ones(2, 3), "bias": torch.full((2,), 0.5)}

    load_encoder_state(model, weights)

    torch.testing.assert_close(model.weight.data, torch.ones(2, 3))
    torch.testing.assert_close(model.bias.data, torch.full((2,), 0.5))


def test_reading_a_file_that_is_not_a_checkpoint_says_so(tmp_path) -> None:
    """A bare state dict is a common mistake and gets a readable error."""
    path = tmp_path / "encoder.pt"
    torch.save({"weight": torch.zeros(2)}, path)

    with pytest.raises(ValueError, match="state_dict"):
        read_checkpoint(path)


def test_reading_a_checkpoint_returns_weights_and_settings(tmp_path) -> None:
    """The settings travel with the weights, so the model can be rebuilt."""
    path = tmp_path / "encoder.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)}, "config": {"d": 128}}, path)

    state_dict, config = read_checkpoint(path)

    assert list(state_dict) == ["w"]
    assert config == {"d": 128}


# --- the permuted-graph control -----------------------------------------------


def test_permuting_identities_keeps_the_edges_and_moves_them() -> None:
    """The control must destroy correspondence without changing the graph.

    If the permutation dropped or added edges, a weaker effect afterwards
    would be explained by a sparser graph rather than by the identities being
    wrong, and the retained fraction would mean nothing.
    """
    rng = np.random.default_rng(3)
    grn = torch.zeros(40, 40)
    rows = rng.choice(40, 60)
    columns = rng.choice(40, 60)
    for row, column in zip(rows, columns, strict=True):
        if row != column:
            grn[row, column] = rng.choice([-1.0, 1.0])

    permuted = permute_regulatory_graph(grn, seed=1)

    assert int((permuted != 0).sum()) == int((grn != 0).sum())
    assert permuted.sum() == pytest.approx(float(grn.sum()))
    assert not torch.equal(permuted, grn)


def test_permuting_is_deterministic() -> None:
    """The same seed gives the same control, so a run can be reproduced."""
    grn = torch.diag(torch.arange(1.0, 11.0))
    torch.testing.assert_close(
        permute_regulatory_graph(grn, seed=7), permute_regulatory_graph(grn, seed=7)
    )


# --- combining runs -----------------------------------------------------------


def write_per_fold(path, model: str, run: str, n: int = 25, centre: float = 0.8):
    """Write a per-fold file for one model under one run name."""
    rng = np.random.default_rng(abs(hash(run)) % 2**32)
    rows = [
        {
            "model": model,
            "run": run,
            "repeat": index // 5,
            "fold": index % 5,
            "metric": "macro_f1",
            "value": centre + rng.normal(scale=0.01),
        }
        for index in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_variants_sharing_a_registry_key_are_kept_apart(tmp_path) -> None:
    """Three transformer variants must not fuse into one impossible vector.

    All three register as ``M_mogformer`` because they are the same model with
    different settings. Concatenated naively that is a single 75-element
    vector for a 25-fold partition, and every paired test over it is nonsense.
    """
    paths = [
        write_per_fold(tmp_path / run / "metrics_per_fold.csv", "M_mogformer", run)
        for run in ("v1_pretrained", "v2_learned", "v3_gated")
    ]

    combined = combine_per_fold(paths, tmp_path / "combined.csv")

    assert set(combined["model"]) == {
        "M_mogformer@v1_pretrained",
        "M_mogformer@v2_learned",
        "M_mogformer@v3_gated",
    }
    assert (combined.groupby("model").size() == 25).all()


def test_a_model_run_once_keeps_its_plain_name(tmp_path) -> None:
    """Qualification is applied only where it is needed, so names stay legible."""
    paths = [
        write_per_fold(
            tmp_path / "base" / "metrics_per_fold.csv", "B5_xgboost", "base"
        ),
        write_per_fold(tmp_path / "v1" / "metrics_per_fold.csv", "M_mogformer", "v1"),
        write_per_fold(tmp_path / "v2" / "metrics_per_fold.csv", "M_mogformer", "v2"),
    ]

    combined = combine_per_fold(paths, tmp_path / "combined.csv")

    assert "B5_xgboost" in set(combined["model"])
    assert "M_mogformer" not in set(combined["model"])
    assert "M_mogformer@v1" in set(combined["model"])


def test_a_file_without_a_run_column_is_named_after_its_folder(tmp_path) -> None:
    """Per-fold files written before the run column existed still combine."""
    path = write_per_fold(tmp_path / "older_run" / "metrics_per_fold.csv", "M", "x")
    frame = pd.read_csv(path).drop(columns=["run"])
    frame.to_csv(path, index=False)

    combined = combine_per_fold([path], tmp_path / "combined.csv")

    assert set(combined["run"]) == {"older_run"}


def test_combining_nothing_is_an_error(tmp_path) -> None:
    """An empty combine would otherwise surface much later, as a stats error."""
    with pytest.raises(ValueError, match="no per-fold files"):
        combine_per_fold([], tmp_path / "combined.csv")


def test_combined_variants_survive_the_paired_comparison(tmp_path) -> None:
    """End to end: three variants plus a baseline, compared on one partition."""
    paths = [
        write_per_fold(
            tmp_path / "base" / "metrics_per_fold.csv",
            "B5_xgboost",
            "base",
            centre=0.90,
        ),
        *[
            write_per_fold(
                tmp_path / run / "metrics_per_fold.csv",
                "M_mogformer",
                run,
                centre=centre,
            )
            for run, centre in (("v1", 0.84), ("v2", 0.83), ("v3", 0.82))
        ],
    ]
    combined_csv = tmp_path / "combined.csv"
    combine_per_fold(paths, combined_csv)

    record = run_paired_statistics(combined_csv, tmp_path / "out", n_splits=5)

    assert record["n_models"] == 4
    assert record["n_estimates"] == 25
    # The baseline was drawn highest, so it should hold the best average rank.
    ranks = record["average_ranks"]
    assert min(ranks, key=lambda name: ranks[name]) == "B5_xgboost"


# --- default locations --------------------------------------------------------


def test_output_defaults_to_the_experiments_results_directory(
    monkeypatch, tmp_path
) -> None:
    """Without --out, results land beside the committed experiment evidence."""
    monkeypatch.chdir(tmp_path)

    assert experiment_driver.resolve_output(None) == Path("results/experiments")
    assert (tmp_path / "results" / "experiments").is_dir()
    custom = tmp_path / "custom"
    assert experiment_driver.resolve_output(str(custom)) == custom


def test_a_missing_bundle_names_the_command_that_builds_it(
    monkeypatch, tmp_path
) -> None:
    """Without a bundle the error says how to make one, not just that it failed."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"utils\.build_cohort_bundle"):
        experiment_driver.resolve_bundle(None)


def test_an_explicit_bundle_is_used_as_given(tmp_path) -> None:
    """An explicit --bundle is checked and returned as the layout's root."""
    root = write_bundle(tmp_path / "bundle")
    (root / "MANIFEST.json").write_text("{}", encoding="utf-8")

    assert experiment_driver.resolve_bundle(str(root)).root == root


# --- archived encoders in the bundle ------------------------------------------


def test_the_bundle_serves_an_encoder_by_cohort_name(tmp_path) -> None:
    """The external sweep finds its frozen encoder without a filesystem path."""
    layout = BundleLayout(write_bundle(tmp_path / "bundle"))
    (layout.root / "encoders" / "metabric").mkdir(parents=True)

    assert layout.encoder("METABRIC").name == "metabric"
    assert layout.encoder("metabric").is_dir()


def test_a_missing_encoder_names_the_ones_that_are_there(tmp_path) -> None:
    """The error says what the bundle does carry, not just what it lacks."""
    layout = BundleLayout(write_bundle(tmp_path / "bundle"))
    (layout.root / "encoders" / "tcga").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="tcga"):
        layout.encoder("metabric")


# --- baselines per modality subset ----------------------------------------------


def test_the_centroid_reference_is_dropped_without_expression(tmp_path) -> None:
    """A copy-number run must not report an expression score under its name."""
    from mogformer.experiments.benchmark import BASELINE_MODELS, baseline_models_for

    config = load_config(write_config(tmp_path / "config.yaml"))
    assert baseline_models_for(config) == BASELINE_MODELS

    config.preprocess.active_modalities = ("cnv", "methy")
    kept = baseline_models_for(config)
    assert "B1_pam50_centroid" not in kept
    assert set(kept) == set(BASELINE_MODELS) - {"B1_pam50_centroid"}


# --- report figures ---------------------------------------------------------------


def _write_scores(path, models, run, centre):
    """Write a synthetic per-fold file with per-class F1 beside macro-F1."""
    rng = np.random.default_rng(len(run))
    rows = []
    for model in models:
        for repeat in range(5):
            for fold in range(5):
                for metric in ("macro_f1", "f1__BRCA_Basal", "f1__BRCA_LumA"):
                    rows.append(
                        {
                            "model": model,
                            "run": run,
                            "repeat": repeat,
                            "fold": fold,
                            "metric": metric,
                            "value": centre + rng.normal(0, 0.01),
                        }
                    )
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "metrics_per_fold.csv", index=False)
    np.savetxt(path / "confusion_x.csv", np.full((2, 2), 10.0), delimiter=",")


def test_report_figures_are_drawn_from_saved_scores(tmp_path) -> None:
    """Every figure is written when its scores exist, and skipped when not."""
    from mogformer.experiments.report_figures import build_report_figures

    root = tmp_path / "experiments"
    baselines = ["B5_xgboost", "B2_elasticnet_logreg", "B1_pam50_centroid"]
    for run, centre in (
        ("tcga_5class", 0.85),
        ("tcga_4class", 0.9),
        ("tcga_5class_rna_only", 0.86),
        ("tcga_5class_cnv_only", 0.6),
    ):
        _write_scores(root / "baselines" / run, baselines, run, centre)
    for run, centre in (
        ("transformer_pretrained_embedding_5class", 0.84),
        ("transformer_pretrained_embedding_rna_only_5class", 0.85),
        ("transformer_learned_embedding_mask_5class", 0.83),
        ("transformer_attention_no_gene_identity_5class", 0.6),
        ("transformer_gated_fusion_5class", 0.83),
    ):
        _write_scores(root / "transformer_variants" / run, ["M_mogformer"], run, centre)

    written = build_report_figures(
        root, tmp_path / "figures", nearest_distance=np.linspace(0, 0.3, 20)
    )

    assert set(written) == {
        "benchmark_forest",
        "modality_ablation",
        "paired_modality_gain",
        "transformer_ablations",
        "per_class_by_modality",
        "gene_identity",
        "trans_probe_cohorts",
        "trans_sign_sources",
        "simulation_sign_recovery",
        "metabric_target_by_modality",
        "metabric_contrasts",
        "methylation_representation",
        "stopping_rule",
    }
    probe_figures = {
        "trans_probe_cohorts",
        "trans_sign_sources",
        "simulation_sign_recovery",
        "metabric_target_by_modality",
        "metabric_contrasts",
        "methylation_representation",
        "stopping_rule",
    }
    assert all(written[name] is None for name in probe_figures)
    for name, path in written.items():
        if name in probe_figures:
            continue
        assert path is not None, name
        assert Path(path).exists()

    empty = build_report_figures(tmp_path / "nothing", tmp_path / "none")
    assert all(path is None for path in empty.values())


def test_observed_sign_concordance_counts_matching_signs() -> None:
    """Zero or missing co-expression carries no sign and is left out."""
    from mogformer.experiments.report_figures import observed_sign_concordance

    edges = pd.DataFrame(
        {
            "edge_sign": [1, 1, -1, -1, 1],
            "observed_coexpression": [0.3, -0.2, -0.1, 0.0, np.nan],
        }
    )
    assert observed_sign_concordance(edges) == pytest.approx(2 / 3)


def test_trans_probe_figure_reads_each_cohort_it_finds(tmp_path) -> None:
    """A probed cohort is drawn; an unprobed one is skipped, not an error."""
    import json

    from mogformer.experiments.report_figures import plot_trans_probe_cohorts

    run = tmp_path / "sign_concordance" / "TCGA"
    run.mkdir(parents=True)
    entry = {
        "sign_concordance": {
            "rate": 0.6,
            "null_mean_rate": 0.58,
            "null_sd_rate": 0.01,
            "trivial_majority_baseline": 0.77,
            "perm_p": 0.15,
            "n_edges": 4,
        },
        "magnitude": {
            "edge_abs_median": 0.003,
            "control_abs_median": 0.002,
            "mw_p_edge_stronger": 1e-5,
        },
        "graph_nulls": {
            "permute_grn_only": {"graph_attributable_pct": -8.0},
            "permute_graph": {"graph_attributable_pct": 35.0},
        },
    }
    (run / "sign_tests.json").write_text(
        json.dumps({"modes": {"whole_gene": entry}}), encoding="utf-8"
    )
    pd.DataFrame(
        {"edge_sign": [1, -1, 1, 1], "observed_coexpression": [0.2, 0.1, -0.3, 0.4]}
    ).to_csv(run / "edges_whole_gene.csv", index=False)

    assert plot_trans_probe_cohorts(tmp_path, tmp_path / "figures").exists()
    assert plot_trans_probe_cohorts(tmp_path / "empty", tmp_path / "none") is None


def test_sign_agreements_compare_three_sources() -> None:
    """Model, data and curation are compared pairwise on the same edges."""
    from mogformer.experiments.report_figures import sign_agreements

    edges = pd.DataFrame(
        {
            "slope": [0.1, 0.2, -0.1, 0.3],
            "observed_coexpression": [0.5, 0.4, -0.2, -0.1],
            "edge_sign": [1, -1, -1, 1],
        }
    )
    rates = sign_agreements(edges)
    assert rates["model_vs_data"] == pytest.approx(0.75)
    assert rates["model_vs_curated"] == pytest.approx(0.75)
    assert rates["data_vs_curated"] == pytest.approx(0.5)


def test_simulation_figure_reads_simulated_and_observed_rows(tmp_path: Path) -> None:
    """Both kinds of row are drawn, and a missing benchmark draws nothing."""
    from mogformer.experiments.report_figures import (
        SIMULATION_ESTIMATORS,
        plot_simulation_benchmark,
    )

    rows = []
    for condition, confounding, use_grn, retention in (
        ("n949_confounding0.8_with_grn", 0.8, True, 30.0),
        ("n949_confounding0.8_without_grn", 0.8, False, None),
        ("TCGA-BRCA (observed)", None, True, -8.0),
    ):
        for column, label, _, _ in SIMULATION_ESTIMATORS:
            rows.append(
                {
                    "condition": condition,
                    "n_patients": 949,
                    "confounding": confounding,
                    "use_grn": use_grn,
                    "estimator": label,
                    "column": column,
                    "curated_sign_rate": 0.7,
                    "trivial_majority_baseline": 0.77,
                    "grn_attributable_pct": retention if column == "slope" else None,
                }
            )
    run = tmp_path / "simulation_benchmark"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "summary.csv", index=False)

    assert plot_simulation_benchmark(tmp_path, tmp_path / "figures").exists()
    assert plot_simulation_benchmark(tmp_path / "empty", tmp_path / "none") is None


def test_target_by_modality_figure_reads_every_run_it_finds(tmp_path: Path) -> None:
    """Classical and transformer runs are both drawn; a missing cohort draws nothing."""
    from mogformer.experiments.report_figures import plot_target_by_modality

    for subset, centre in (("all_omics", 0.7), ("cnv_only", 0.8), ("rna_only", 0.6)):
        _write_scores(
            tmp_path / "baselines" / f"metabric_her2_snp6_{subset}",
            ["B3_linear_svm", "B4_random_forest", "B0a_dummy_stratified"],
            subset,
            centre,
        )
        _write_scores(
            tmp_path
            / "transformer_variants"
            / f"metabric_her2_snp6_{subset}_gated_fusion_smoothed_stopping",
            ["M_mogformer"],
            subset,
            centre,
        )

    assert plot_target_by_modality(tmp_path, tmp_path / "figures").exists()
    assert plot_target_by_modality(tmp_path / "empty", tmp_path / "none") is None


def test_a_run_reads_the_cohort_archive_its_configuration_names(tmp_path) -> None:
    """A configuration asking for M-values is scored on the M-value archive.

    ``prepare`` chooses the archive; the layout it returns is what the run must
    read from, not the bundle's default.
    """
    import argparse

    root = write_bundle(tmp_path / "bundle")
    (root / "cohort_m.npz").write_bytes(b"")
    config_path = write_config(tmp_path / "m_values.yaml")
    text = config_path.read_text(encoding="utf-8").replace(
        "raw/methy.csv", "raw/data_methylation_M.csv"
    )
    config_path.write_text(text, encoding="utf-8")

    cohort_file = experiment_driver._cohort_for(load_config(config_path))
    _, layout = experiment_driver.prepare(
        config_path,
        BundleLayout(root),
        tmp_path / "out",
        argparse.Namespace(),
        cohort_file,
    )

    assert cohort_file == "cohort_m.npz"
    assert layout.cohort == root / "cohort_m.npz"


def test_contrast_figure_pairs_subsets_within_a_model(tmp_path: Path) -> None:
    """A contrast is drawn whenever both of its subsets were run for a model."""
    from mogformer.experiments.report_figures import plot_metabric_contrasts

    for subset, centre in (("cnv_only", 0.7), ("rna_only", 0.6), ("all_omics", 0.65)):
        _write_scores(
            tmp_path / "baselines" / f"metabric_her2_snp6_{subset}",
            ["B3_linear_svm", "B4_random_forest"],
            subset,
            centre,
        )

    assert plot_metabric_contrasts(tmp_path, tmp_path / "figures").exists()
    assert plot_metabric_contrasts(tmp_path / "empty", tmp_path / "none") is None


def test_methylation_figure_pairs_runs_with_the_archived_beta_scores(
    tmp_path: Path,
) -> None:
    """M-value runs are paired fold by fold with the archived beta scores."""
    from mogformer.experiments.report_figures import plot_methylation_representation

    run = "transformer_gated_fusion_5class"
    _write_scores(tmp_path / "transformer_variants" / run, ["M_mogformer"], run, 0.83)
    archive = tmp_path / "methylation_representation"
    archive.mkdir()
    pd.DataFrame(
        {
            "model": "M_mogformer",
            "run": run,
            "repeat": np.repeat(np.arange(5), 5),
            "fold": np.tile(np.arange(5), 5),
            "metric": "macro_f1",
            "value": 0.82,
        }
    ).to_csv(archive / "tcga_transformer_scores_beta.csv", index=False)

    assert plot_methylation_representation(tmp_path, tmp_path / "figures").exists()
    assert plot_methylation_representation(tmp_path / "none", tmp_path / "x") is None


def test_stopping_rule_figure_pairs_the_two_rules_fold_by_fold(tmp_path: Path) -> None:
    """A target is drawn only when both stopping rules were run on it."""
    from mogformer.experiments.report_figures import plot_stopping_rule

    for suffix, centre in (
        ("_gated_fusion", 0.6),
        ("_gated_fusion_smoothed_stopping", 0.65),
    ):
        _write_scores(
            tmp_path / "transformer_variants" / f"metabric_her2_snp6_cnv_only{suffix}",
            ["M_mogformer"],
            "cnv_only",
            centre,
        )

    assert plot_stopping_rule(tmp_path, tmp_path / "figures").exists()
    assert plot_stopping_rule(tmp_path / "empty", tmp_path / "none") is None


def test_paper_figures_are_drawn_at_print_size_from_saved_scores(tmp_path) -> None:
    """Both composite figures are written at page width; nothing without scores."""
    from PIL import Image

    from mogformer.experiments.paper_figures import PAGE_WIDTH, build_paper_figures

    root = tmp_path / "experiments"
    for run, centre in (("tcga_5class", 0.85), ("tcga_5class_rna_only", 0.86)):
        _write_scores(
            root / "baselines" / run,
            ["B5_xgboost", "B2_elasticnet_logreg"],
            run,
            centre,
        )
    for run in (
        "transformer_pretrained_embedding_5class",
        "transformer_learned_embedding_mask_5class",
        "transformer_gated_fusion_5class",
    ):
        _write_scores(root / "transformer_variants" / run, ["M_mogformer"], run, 0.83)
    for subset, centre in (("all_omics", 0.6), ("cnv_only", 0.7), ("rna_only", 0.55)):
        run = f"metabric_her2_snp6_{subset}"
        _write_scores(
            root / "baselines" / run,
            ["B3_linear_svm", "B4_random_forest", "B0a_dummy_stratified"],
            run,
            centre,
        )
        _write_scores(
            root / "transformer_variants" / f"{run}_gated_fusion_smoothed_stopping",
            ["M_mogformer"],
            run,
            centre,
        )

    written = build_paper_figures(root, tmp_path / "paper")

    assert all(path is not None for path in written.values())
    for path in written.values():
        width_inches = Image.open(path).size[0] / 300
        assert width_inches <= PAGE_WIDTH + 0.05
    empty = build_paper_figures(tmp_path / "nothing", tmp_path / "none")
    assert all(path is None for path in empty.values())
