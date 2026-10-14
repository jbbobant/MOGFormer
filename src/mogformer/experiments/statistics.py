"""Paired comparison of models over the shared partition.

Repeated cross-validation reuses the same patients across folds, so the 25
estimates are not independent and a plain t-test over them is anticonservative.
Two corrections are applied together: the Nadeau-Bengio variance correction on
each pairwise difference, and a rank-based test that assumes nothing about the
shape of the differences. Where more than two models are in play, a Friedman
test with Nemenyi's critical difference says which gaps survive at all.

This module reads a ``metrics_per_fold.csv`` written by the runner. It never
computes a score itself, so it cannot silently disagree with the leaderboard it
is testing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mogformer.evaluation.plots import (
    apply_house_style,
    plot_critical_difference,
    plot_interval_forest,
    plot_model_comparison,
)
from mogformer.evaluation.runner import PRIMARY_METRIC, load_per_fold_scores, summarise
from mogformer.evaluation.stats import friedman_nemenyi, pairwise_compare
from mogformer.experiments.paths import save_json

logger = logging.getLogger(__name__)

#: Models must share this many estimates before a paired test is meaningful.
MIN_PAIRED_ESTIMATES = 5

#: Separator between a model's registry key and the run it came from, used
#: only when the same key appears under more than one run.
RUN_SEPARATOR = "@"


def combine_per_fold(
    paths: Sequence[Path | str],
    output_csv: Path | str,
    exclude_models: Sequence[str] = (),
) -> pd.DataFrame:
    """Concatenate per-fold files from separate runs into one frame.

    The runs are launched one variant at a time, and all three transformer
    variants register under the same key — ``M_mogformer`` — because they are
    the same model with different settings. Concatenated naively, the three
    become one seventy-five-row vector that pairs fold 3 of one variant against
    fold 3 of another. A key that appears under more than one run is therefore
    qualified by its run name; a key that appears under only one is left alone,
    so a single-run comparison reads exactly as it did before.

    Args:
        paths: ``metrics_per_fold.csv`` files to combine.
        output_csv: Where to write the combined frame.
        exclude_models: Registry keys to leave out, such as the dummy floors,
            which a rank test would only use as filler.

    Returns:
        The combined frame.

    Raises:
        ValueError: If no file could be read, or none carries any rows.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "run" not in frame:
            # Written before the run column existed; name it after its folder.
            frame["run"] = Path(path).parent.name
        frames.append(frame)

    if not frames:
        raise ValueError("no per-fold files to combine")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["model"].isin(list(exclude_models))]
    if combined.empty:
        raise ValueError("the per-fold files carry no rows")

    runs_per_model = combined.groupby("model")["run"].nunique()
    ambiguous = set(runs_per_model[runs_per_model > 1].index)
    if ambiguous:
        logger.info(
            "qualifying %d model key(s) by run: %s",
            len(ambiguous),
            ", ".join(sorted(ambiguous)),
        )
        combined["model"] = [
            f"{model}{RUN_SEPARATOR}{run}" if model in ambiguous else model
            for model, run in zip(combined["model"], combined["run"], strict=True)
        ]

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    logger.info(
        "combined %d rows over %d model(s) into %s",
        len(combined),
        combined["model"].nunique(),
        destination,
    )
    return combined


def align_scores(
    per_fold: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Keep only models scored on the same number of estimates.

    A model that ran on ten folds and one that ran on twenty-five cannot be
    paired, and quietly truncating the longer vector would pair fold 3 of one
    run against fold 3 of a different partition. Dropping the odd ones out and
    saying so is the only honest option.

    Args:
        per_fold: Mapping of model name to its per-fold scores.

    Returns:
        The subset sharing the most common estimate count.

    Raises:
        ValueError: If fewer than two models remain.
    """
    lengths = {name: len(scores) for name, scores in per_fold.items()}
    if not lengths:
        raise ValueError("no per-fold scores to compare")

    counts = pd.Series(list(lengths.values())).value_counts()
    target = int(counts.idxmax())
    kept = {name: per_fold[name] for name, n in lengths.items() if n == target}
    dropped = sorted(set(per_fold) - set(kept))
    if dropped:
        logger.warning(
            "dropped %s: %d estimates each, against the shared %d",
            ", ".join(dropped),
            lengths[dropped[0]],
            target,
        )
    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} model(s) share {target} estimates; a paired "
            "comparison needs at least two"
        )
    return kept


def run_paired_statistics(
    per_fold_csv: Path | str,
    output_dir: Path | str,
    metric: str = PRIMARY_METRIC,
    n_splits: int = 5,
) -> dict[str, object]:
    """Compare every model against every other, and save the evidence.

    Args:
        per_fold_csv: A ``metrics_per_fold.csv`` written by the runner.
        output_dir: Directory to write tables and figures into.
        metric: Metric to compare on.
        n_splits: Folds per repeat, for the corrected variance.

    Returns:
        A record of what was computed, also written as ``paired_tests.json``.
    """
    apply_house_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    scores = align_scores(load_per_fold_scores(per_fold_csv, metric))
    n_estimates = len(next(iter(scores.values())))
    if n_estimates < MIN_PAIRED_ESTIMATES:
        logger.warning(
            "only %d estimates per model; intervals will be very wide",
            n_estimates,
        )

    comparison = pairwise_compare(scores, n_splits=n_splits)
    comparison.to_csv(output / "pairwise_tests.csv", index=False)

    statistic, p_value, average_ranks, critical_difference = friedman_nemenyi(scores)
    average_ranks.rename("average_rank").to_frame().to_csv(output / "average_ranks.csv")

    frame = pd.read_csv(per_fold_csv)
    frame = frame[frame["model"].isin(scores)]
    context = (
        "runs: " + ", ".join(sorted(map(str, frame["run"].unique())))
        if "run" in frame
        else ""
    )

    figures = {
        "model_comparison": plot_model_comparison(
            scores, output, metric=metric, context=context
        ),
        "critical_difference": plot_critical_difference(
            average_ranks, critical_difference, output, context=context
        ),
    }

    summary = summarise(frame, n_splits=n_splits)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    figures["interval_forest"] = plot_interval_forest(
        summary, output, metric=metric, context=context
    )

    record: dict[str, object] = {
        "source": str(per_fold_csv),
        "metric": metric,
        "n_models": len(scores),
        "n_estimates": n_estimates,
        "n_splits": n_splits,
        "friedman": {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "critical_difference": float(critical_difference),
        },
        "average_ranks": {
            str(name): float(rank) for name, rank in average_ranks.items()
        },
        "figures": {name: str(path) for name, path in figures.items()},
    }
    save_json(record, output / "paired_tests.json")

    (output / "paired_tests.md").write_text(
        render_comparison(
            comparison, average_ranks, p_value, critical_difference, metric, n_estimates
        ),
        encoding="utf-8",
    )
    logger.info("paired statistics written to %s", output)
    return record


def render_comparison(
    comparison: pd.DataFrame,
    average_ranks: pd.Series,
    friedman_p: float,
    critical_difference: float,
    metric: str,
    n_estimates: int,
) -> str:
    """Render the paired comparison as Markdown.

    Args:
        comparison: Frame from :func:`pairwise_compare`.
        average_ranks: Mean rank per model, lower is better.
        friedman_p: The omnibus p-value.
        critical_difference: Nemenyi's critical difference at the same level.
        metric: Metric being compared.
        n_estimates: Estimates behind every vector.

    Returns:
        A Markdown document.
    """
    lines = [
        f"# Paired comparison — {metric}",
        "",
        f"{len(average_ranks)} models, {n_estimates} estimates each, all on the",
        "same partition. Differences are tested with the Nadeau-Bengio",
        "corrected variance and, independently, with a rank-based test that",
        "assumes nothing about their shape.",
        "",
        "## Omnibus",
        "",
        f"- Friedman p = {friedman_p:.3g}",
        f"- Nemenyi critical difference = {critical_difference:.3f} ranks",
        "",
        "| model | average rank |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {rank:.2f} |" for name, rank in average_ranks.sort_values().items()
    ]
    lines += ["", "## Pairwise", ""]

    columns = [column for column in comparison.columns if column != "n_pairs"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in comparison.iterrows():
        cells = [
            f"{row[column]:.4g}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"
