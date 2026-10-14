"""Cross-validation orchestration: one protocol, every model.

The runner is the single place a model is scored. It loads the cohort, reads the
persisted partition, and for each fold refits preprocessing on the training
patients, optionally tunes on an inner split, fits, predicts, and records the
metrics. Baselines and the transformer take the same path, which is what makes
the resulting per-fold scores pairable.

Everything it writes — per-fold scores, the aggregated summary, the resolved
configuration — goes into one directory, so a number in a table can always be
traced to the run that produced it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from mogformer.config import ExperimentConfig, save_config
from mogformer.data.folds import (
    FoldSpec,
    assert_folds_match,
    load_folds,
    remap_fold_indices,
    select_folds,
)
from mogformer.data.gene_embeddings import (
    load_gene_embedding_bundle,
    load_gene_embedding_table,
)
from mogformer.data.omics import (
    OmicsData,
    apply_copy_number_representation,
    exclude_classes,
    load_curated_genes,
    load_omics,
    load_omics_bundle,
    load_ploidy,
)
from mogformer.evaluation.metrics import (
    aggregate_across_folds,
    compute_fold_metrics,
    fold_confusion,
)
from mogformer.evaluation.registry import HarnessContext, ModelSpec, build_registry
from mogformer.graph.grn import GRNSignedCache
from mogformer.graph.string_graph import StringGraphCache, load_string_edges
from mogformer.training.seed import seed_everything

logger = logging.getLogger(__name__)

#: Metric decided in advance as primary, so the choice is not made after seeing
#: results.
PRIMARY_METRIC = "macro_f1"


def load_cohort(
    config: ExperimentConfig, bundle: str | Path | None = None
) -> OmicsData:
    """Load the cohort described by a configuration.

    A bundle is the same cohort, aligned once and cached in compressed form.
    Class exclusion is applied here either way, so a run's composition always
    comes from its configuration and never from where the numbers were read.

    Args:
        config: Resolved experiment configuration.
        bundle: A cohort archive written by
            :func:`~mogformer.data.omics.save_omics_bundle`, or None to read
            the raw matrices.

    Returns:
        Aligned raw data with encoded labels, copy number on the scale the
        configuration names (``preprocess.cnv_representation``).
    """
    if bundle is None and config.data.cohort_file is not None:
        bundle = Path(config.data.raw_dir) / config.data.cohort_file
    if bundle is not None:
        data = load_omics_bundle(bundle)
        if config.data.exclude_classes:
            data, _ = exclude_classes(data, config.data.exclude_classes)
    else:
        data = load_omics(
            raw_dir=config.data.raw_dir,
            rna_file=config.data.rna_file,
            cnv_file=config.data.cnv_file,
            methy_file=config.data.methy_file,
            clin_file=config.data.clin_file,
            label_col=config.data.label_col,
            exclude=config.data.exclude_classes,
        )
    ploidy = (
        load_ploidy(config.data.ploidy_file)
        if config.data.ploidy_file is not None
        else None
    )
    return apply_copy_number_representation(
        data, config.preprocess.cnv_representation, ploidy
    )


def project_folds(
    folds: Sequence[FoldSpec], patient_order: Sequence[str], kept: Sequence[str]
) -> tuple[list[FoldSpec], list[str]]:
    """Restrict a partition to the patients a run actually keeps.

    The four-class run is the three-class question asked of the same people, so
    it must reuse the five-class partition rather than draw a new one — a fresh
    draw would put the two runs on different splits and quietly destroy the
    pairing that every comparison in this project depends on.

    Args:
        folds: Partition built over the full cohort.
        patient_order: Barcodes in the order the partition indexes them.
        kept: Barcodes surviving class exclusion.

    Returns:
        Tuple of the projected partition and the barcodes it now indexes.
    """
    keep_mask = np.isin(np.asarray(patient_order), np.asarray(kept))
    projected = remap_fold_indices(list(folds), keep_mask)
    remaining = [
        pid for pid, keep in zip(patient_order, keep_mask, strict=True) if keep
    ]
    logger.info(
        "projected the partition from %d onto %d patients",
        len(patient_order),
        len(remaining),
    )
    return projected, remaining


def build_graph_caches(
    config: ExperimentConfig, gene_universe: Sequence[str]
) -> tuple[StringGraphCache, GRNSignedCache | None]:
    """Parse the biological priors once for a whole run.

    Parsing STRING dominates runtime and the gene universe is fixed across
    folds, so this happens here rather than inside the fold loop. Only the
    *induced* subgraph is fold-local, and that is rebuilt per fold from the
    genes that fold selected.

    Args:
        config: Resolved experiment configuration.
        gene_universe: Every gene in the cohort.

    Returns:
        Tuple of the interaction cache and the regulatory cache, the latter
        None unless the run enables it.

    Raises:
        ValueError: If the regulatory bias is enabled without a source file.
    """
    edges_file = config.graph.edges_file
    if edges_file:
        string_cache = StringGraphCache.from_edges(
            load_string_edges(edges_file), config.graph.string_threshold
        )
        logger.info(
            "STRING: %d edges read from %s",
            string_cache.n_universe_edges,
            edges_file,
        )
    else:
        string_cache = StringGraphCache(
            config.graph.ppi_file,
            config.graph.alias_file,
            gene_universe,
            config.graph.string_threshold,
        )
    grn_cache = None
    if config.graph.use_grn:
        if not config.graph.grn_file:
            raise ValueError("use_grn is set but no grn_file was configured")
        grn_cache = GRNSignedCache(config.graph.grn_file, gene_universe)
    return string_cache, grn_cache


def build_context(
    config: ExperimentConfig,
    data: OmicsData,
    pam50_genes: Sequence[str] = (),
    with_graph: bool = False,
    device: str | None = None,
) -> HarnessContext:
    """Assemble the context every model factory needs.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort.
        pam50_genes: Panel genes for the label-source reference model.
        with_graph: Parse the biological priors and the pretrained gene
            vectors, which the transformer needs and the baselines do not.
        device: Torch device for the transformer, or None to detect.

    Returns:
        A context describing this run.
    """
    curated = load_curated_genes(config.data.curated_genes_file, data.gene_names)
    string_cache: StringGraphCache | None = None
    grn_cache: GRNSignedCache | None = None
    embeddings = None
    if with_graph:
        string_cache, grn_cache = build_graph_caches(config, data.gene_names)
        embeddings = load_pretrained_embeddings(config)
    return HarnessContext(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        curated_genes=curated,
        pam50_genes=list(pam50_genes),
        n_classes=len(data.label_map),
        top_k=config.preprocess.top_k,
        active_modalities=config.preprocess.active_modalities,
        seed=config.train.seed,
        mad_on_log_rna=config.preprocess.mad_on_log_rna,
        rna_scale=config.preprocess.rna_scale,
        string_cache=string_cache,
        grn_cache=grn_cache,
        graph=config.graph,
        model=config.model,
        train=config.train,
        pretrained_embeddings=embeddings,
        device=device,
    )


def load_pretrained_embeddings(config: ExperimentConfig) -> pd.DataFrame | None:
    """Read the frozen gene vectors a run asks for, if any.

    Args:
        config: Resolved experiment configuration.

    Returns:
        Embeddings indexed by gene symbol, or None when the run uses a learned
        identity table instead.
    """
    source = config.model.pretrained_emb_file
    if not source:
        return None
    path = Path(source)
    if path.suffix == ".npz":
        return load_gene_embedding_bundle(path)
    symbols = config.model.pretrained_emb_symbol_file
    if not symbols:
        raise ValueError(
            "pretrained_emb_file in text form needs pretrained_emb_symbol_file"
        )
    return load_gene_embedding_table(path, symbols)


def score_fold(
    spec: ModelSpec,
    context: HarnessContext,
    data: OmicsData,
    fold: FoldSpec,
    n_inner_splits: int = 3,
    n_search_iter: int = 20,
    n_search_jobs: int = 1,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit and score one model on one fold.

    Hyperparameter search, when the specification asks for it, runs on an inner
    split of the training fold only. The held-out fold is never seen during
    model selection.

    Args:
        spec: Model to fit.
        context: Run-level settings.
        data: Loaded cohort.
        fold: The partition to use.
        n_inner_splits: Folds in the inner search.
        n_search_iter: Parameter settings sampled by the inner search.
        n_search_jobs: Processes for the inner search; -1 uses every core.
            Every estimator is seeded, so the chosen parameters and the
            scores do not depend on it.

    Returns:
        Tuple of the fold's metrics and its confusion matrix.
    """
    train_x, train_y = data.X[fold.train_idx], data.y[fold.train_idx]
    test_x, test_y = data.X[fold.test_idx], data.y[fold.test_idx]

    estimator = spec.build(context)
    if spec.tune and spec.search_space:
        search = RandomizedSearchCV(
            estimator,
            spec.search_space,
            n_iter=n_search_iter,
            scoring="f1_macro",
            cv=StratifiedKFold(
                n_splits=n_inner_splits, shuffle=True, random_state=context.seed
            ),
            random_state=context.seed,
            n_jobs=n_search_jobs,
            refit=True,
        )
        search.fit(train_x, train_y)
        estimator = search.best_estimator_
        logger.debug(
            "%s fold %d best params: %s", spec.name, fold.fold, search.best_params_
        )
    else:
        estimator = clone(estimator).fit(train_x, train_y)

    predictions = estimator.predict(test_x)
    probabilities = (
        estimator.predict_proba(test_x) if hasattr(estimator, "predict_proba") else None
    )

    class_labels = list(range(len(data.label_map)))
    class_names = [data.inverse_label_map[i] for i in class_labels]
    metrics = compute_fold_metrics(
        test_y, predictions, probabilities, class_labels, class_names
    )
    return metrics, fold_confusion(test_y, predictions, class_labels)


def run_cross_validation(
    config: ExperimentConfig,
    models: Sequence[str] | None = None,
    pam50_genes: Sequence[str] = (),
    data: OmicsData | None = None,
    bundle: str | Path | None = None,
    with_graph: bool | None = None,
    device: str | None = None,
    n_search_jobs: int = 1,
) -> pd.DataFrame:
    """Score every requested model across the shared partition.

    Args:
        config: Resolved experiment configuration.
        models: Registry keys to run, or None for every registered model.
        pam50_genes: Panel genes for the label-source reference model.
        data: A cohort loaded elsewhere, or None to load from ``config``.
        bundle: A cohort archive to read instead of the raw matrices, through
            the same protocol rather than a second loading path.
        with_graph: Parse the biological priors, which only the transformer
            needs. None means "whenever the transformer was requested".
        device: Torch device for the transformer, or None to detect.
        n_search_jobs: Processes for each inner hyperparameter search.

    Returns:
        Long-format frame with one row per model, fold and metric.

    Raises:
        KeyError: If a requested model is not registered.
        ValueError: If the persisted partition does not match the cohort.
    """
    seed_everything(config.train.seed)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")

    cohort = load_cohort(config, bundle) if data is None else data
    if with_graph is None:
        with_graph = models is None or any(name.startswith("M_") for name in models)
    context = build_context(config, cohort, pam50_genes, with_graph, device)

    folds, patient_order = load_folds(config.folds.path)
    if len(patient_order) != cohort.n_patients:
        folds, patient_order = project_folds(folds, patient_order, cohort.patient_ids)
    # The guard that would have caught the 914-versus-949 divergence.
    assert_folds_match(patient_order, cohort.patient_ids)
    folds = select_folds(
        folds,
        n_repeats=config.folds.n_repeats_used,
        max_folds=config.folds.max_folds,
    )
    logger.info("scoring across %d folds", len(folds))

    registry = build_registry(context)
    requested = list(registry) if models is None else list(models)
    missing = [name for name in requested if name not in registry]
    if missing:
        raise KeyError(
            f"unregistered model(s): {missing}; available: {sorted(registry)}"
        )

    records: list[dict[str, object]] = []
    confusions: dict[str, np.ndarray] = {}

    for name in requested:
        spec = registry[name]
        logger.info("=== %s ===", name)
        for fold in folds:
            metrics, matrix = score_fold(
                spec, context, cohort, fold, n_search_jobs=n_search_jobs
            )
            running = confusions.get(name)
            confusions[name] = matrix if running is None else running + matrix
            records.extend(
                {
                    "model": name,
                    # Three transformer variants all register as M_mogformer,
                    # so without the run name their per-fold files cannot be
                    # told apart once they are concatenated.
                    "run": config.name,
                    "repeat": fold.repeat,
                    "fold": fold.fold,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
            logger.info(
                "  repeat %d fold %d | %s %.4f",
                fold.repeat,
                fold.fold,
                PRIMARY_METRIC,
                metrics[PRIMARY_METRIC],
            )

    per_fold = pd.DataFrame(records)
    per_fold.to_csv(output / "metrics_per_fold.csv", index=False)

    summary = summarise(per_fold, n_splits=config.folds.n_splits)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    (output / "metrics_summary.md").write_text(
        render_leaderboard(summary), encoding="utf-8"
    )

    for name, matrix in confusions.items():
        np.savetxt(output / f"confusion_{name}.csv", matrix, fmt="%d", delimiter=",")

    logger.info("wrote results to %s", output)
    return per_fold


def summarise(per_fold: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Aggregate a long-format per-fold frame into one row per model and metric.

    Args:
        per_fold: Frame with ``model``, ``repeat``, ``fold``, ``metric`` and
            ``value`` columns.
        n_splits: Folds per repeat, for the corrected interval.

    Returns:
        Frame carrying the mean, both intervals and the estimate count.
    """
    rows = []
    for (model, metric), group in per_fold.groupby(["model", "metric"]):
        summary = aggregate_across_folds(group["value"].to_numpy(), n_splits=n_splits)
        rows.append({"model": model, "metric": metric, **summary})
    return pd.DataFrame(rows)


def render_leaderboard(summary: pd.DataFrame, metric: str = PRIMARY_METRIC) -> str:
    """Render the leaderboard for one metric as Markdown.

    Both intervals are shown side by side so the optimism of the naive one
    remains visible.

    Args:
        summary: Aggregated frame from :func:`summarise`.
        metric: Metric to rank on.

    Returns:
        A Markdown table, best model first.
    """
    rows = summary[summary["metric"] == metric].sort_values("mean", ascending=False)
    lines = [
        f"# Leaderboard — {metric}",
        "",
        f"Mean over {int(rows['n'].max()) if len(rows) else 0} estimates, with the "
        "naive and Nadeau–Bengio corrected 95% intervals.",
        "",
        "| model | mean | 95% CI | NB 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['model']} | {row['mean']:.3f} | "
            f"[{row['ci95_lo']:.3f}, {row['ci95_hi']:.3f}] | "
            f"[{row['nb_ci95_lo']:.3f}, {row['nb_ci95_hi']:.3f}] |"
        )
    return "\n".join(lines) + "\n"


def load_per_fold_scores(
    path: str | Path, metric: str = PRIMARY_METRIC
) -> dict[str, np.ndarray]:
    """Read per-fold scores for one metric, ready for paired comparison.

    Args:
        path: A ``metrics_per_fold.csv`` written by
            :func:`run_cross_validation`.
        metric: Metric to extract.

    Returns:
        Mapping of model name to its per-fold scores, ordered by repeat then
        fold so every model's vector is aligned to the same partition.
    """
    frame = pd.read_csv(path)
    frame = frame[frame["metric"] == metric].sort_values(["repeat", "fold"])
    return {
        str(model): group["value"].to_numpy() for model, group in frame.groupby("model")
    }
