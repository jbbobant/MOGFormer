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
    select_folds,
)
from mogformer.data.omics import OmicsData, load_curated_genes, load_omics
from mogformer.evaluation.metrics import (
    aggregate_across_folds,
    compute_fold_metrics,
    fold_confusion,
)
from mogformer.evaluation.registry import HarnessContext, ModelSpec, build_registry
from mogformer.training.seed import seed_everything

logger = logging.getLogger(__name__)

#: Metric decided in advance as primary, so the choice is not made after seeing
#: results.
PRIMARY_METRIC = "macro_f1"


def load_cohort(config: ExperimentConfig) -> OmicsData:
    """Load the cohort described by a configuration.

    Args:
        config: Resolved experiment configuration.

    Returns:
        Aligned raw data with encoded labels.
    """
    return load_omics(
        raw_dir=config.data.raw_dir,
        rna_file=config.data.rna_file,
        cnv_file=config.data.cnv_file,
        methy_file=config.data.methy_file,
        clin_file=config.data.clin_file,
        label_col=config.data.label_col,
        exclude=config.data.exclude_classes,
    )


def build_context(
    config: ExperimentConfig, data: OmicsData, pam50_genes: Sequence[str] = ()
) -> HarnessContext:
    """Assemble the context every model factory needs.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort.
        pam50_genes: Panel genes for the label-source reference model.

    Returns:
        A context describing this run.
    """
    curated = load_curated_genes(config.data.curated_genes_file, data.gene_names)
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
    )


def score_fold(
    spec: ModelSpec,
    context: HarnessContext,
    data: OmicsData,
    fold: FoldSpec,
    n_inner_splits: int = 3,
    n_search_iter: int = 20,
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
            n_jobs=1,
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
) -> pd.DataFrame:
    """Score every requested model across the shared partition.

    Args:
        config: Resolved experiment configuration.
        models: Registry keys to run, or None for every registered model.
        pam50_genes: Panel genes for the label-source reference model.

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

    data = load_cohort(config)
    context = build_context(config, data, pam50_genes)

    folds, patient_order = load_folds(config.folds.path)
    # The guard that would have caught the 914-versus-949 divergence.
    assert_folds_match(patient_order, data.patient_ids)
    folds = select_folds(
        folds,
        n_repeats=config.folds.n_repeats_used,
        max_folds=config.folds.max_folds,
    )
    logger.info("scoring across %d folds", len(folds))

    registry = build_registry()
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
            metrics, matrix = score_fold(spec, context, data, fold)
            running = confusions.get(name)
            confusions[name] = matrix if running is None else running + matrix
            records.extend(
                {
                    "model": name,
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
