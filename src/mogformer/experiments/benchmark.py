"""Score models on the shared partition — the runs behind the leaderboard.

Classical baselines and the transformer deliberately share one code path.
They are not two pipelines but registry entries: they take the same folds, the
same fold-local preprocessing and the same metric functions, and what comes out
is two vectors indexed by the same 25 estimates. Only then does a paired test
over them mean anything.

The four-class run reuses the five-class partition rather than drawing its own,
so the two cohorts stay paired. That is handled inside the runner; here it is
just a second configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mogformer.config import ExperimentConfig
from mogformer.evaluation.runner import PRIMARY_METRIC, run_cross_validation, summarise
from mogformer.experiments.paths import BundleLayout, provenance_record, save_json

logger = logging.getLogger(__name__)

#: Registry keys of the classical baselines, floor first.
BASELINE_MODELS: tuple[str, ...] = (
    "B0a_dummy_stratified",
    "B0b_dummy_mostfreq",
    "B1_pam50_centroid",
    "B2_elasticnet_logreg",
    "B3_linear_svm",
    "B4_random_forest",
    "B5_xgboost",
)


def baseline_models_for(config: ExperimentConfig) -> tuple[str, ...]:
    """Return the baselines that are meaningful for a configuration's modalities.

    The PAM50 centroid reference always reads expression of the panel genes,
    whatever the configuration selects. In a run without expression it would
    report an expression score under a copy-number or methylation label, so it
    is left out there.

    Args:
        config: The run's configuration.

    Returns:
        Registry keys, floor first.
    """
    if "rna" in config.preprocess.active_modalities:
        return BASELINE_MODELS
    return tuple(name for name in BASELINE_MODELS if name != "B1_pam50_centroid")


#: Registry key of the transformer.
TRANSFORMER_MODELS: tuple[str, ...] = ("M_mogformer",)


@dataclass
class BenchmarkResult:
    """What one benchmark run produced.

    Attributes:
        name: Run identifier, which is also the results subdirectory.
        per_fold: Long-format frame, one row per model, fold and metric.
        summary: Aggregated frame, one row per model and metric.
        output_dir: Directory everything was written to.
    """

    name: str
    per_fold: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path

    def leaderboard(self, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
        """Return the run's models ranked on one metric.

        Args:
            metric: Metric to rank on.

        Returns:
            Frame sorted best first.
        """
        rows = self.summary[self.summary["metric"] == metric]
        return rows.sort_values("mean", ascending=False).reset_index(drop=True)


def run_benchmark(
    config: ExperimentConfig,
    layout: BundleLayout,
    models: Sequence[str],
    results_dir: Path | str,
    device: str | None = None,
    n_search_jobs: int = 1,
) -> BenchmarkResult:
    """Run one configuration over one set of models and save everything.

    Args:
        config: Configuration already rebased onto ``layout``.
        layout: Where the bundle's files sit.
        models: Registry keys to score.
        results_dir: Root the run writes under.
        device: Torch device for the transformer, or None to detect.
        n_search_jobs: Processes for each inner hyperparameter search.

    Returns:
        The run's scores and where they were written.
    """
    layout.require()
    logger.info("=== %s: %d model(s) ===", config.name, len(models))

    per_fold = run_cross_validation(
        config,
        models=list(models),
        pam50_genes=layout.read_pam50(),
        bundle=layout.cohort,
        device=device,
        n_search_jobs=n_search_jobs,
    )
    summary = summarise(per_fold, n_splits=config.folds.n_splits)
    output = config.output_dir

    save_json(
        provenance_record(
            {
                "run": config.name,
                "models": list(models),
                "cohort_file": layout.cohort_file,
                "exclude_classes": list(config.data.exclude_classes),
                "n_estimates": int(per_fold["fold"].count() / max(len(models), 1)),
            }
        ),
        output / "provenance.json",
    )

    result = BenchmarkResult(
        name=config.name, per_fold=per_fold, summary=summary, output_dir=output
    )
    board = result.leaderboard()
    for _, row in board.iterrows():
        logger.info(
            "  %-24s %.4f  NB [%.3f, %.3f]  n=%d",
            row["model"],
            row["mean"],
            row["nb_ci95_lo"],
            row["nb_ci95_hi"],
            int(row["n"]),
        )
    return result


def run_benchmark_pair(
    five_class: ExperimentConfig,
    four_class: ExperimentConfig,
    layout: BundleLayout,
    models: Sequence[str],
    results_dir: Path | str,
    device: str | None = None,
) -> dict[str, BenchmarkResult]:
    """Run both class configurations of one benchmark.

    Reporting only the four-class numbers would flatter every model, because
    the Normal-like class is the one they all fail on. Reporting only the
    five-class ones would drop the comparison the earlier baselines were
    published under. Both are run, from one partition, so neither is a
    retrospective choice.

    Args:
        five_class: Configuration keeping every labelled patient.
        four_class: Configuration excluding the Normal-like class.
        layout: Where the bundle's files sit.
        models: Registry keys to score.
        results_dir: Root both runs write under.
        device: Torch device for the transformer, or None to detect.

    Returns:
        Mapping of run name to its result.
    """
    results: dict[str, BenchmarkResult] = {}
    for config in (five_class, four_class):
        results[config.name] = run_benchmark(
            config, layout, models, results_dir, device
        )
    return results
