"""Figures for model comparison.

Every plot here draws from the same per-fold scores the statistics use, so a
figure and a table can never disagree. Uncertainty is always drawn: a bare mean
across cross-validation folds hides exactly the variation that decides whether
two models actually differ.

The palette is Okabe–Ito, which stays distinguishable under the common forms of
colour vision deficiency.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#: Okabe-Ito qualitative palette, safe under colour vision deficiency.
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}


def apply_house_style() -> None:
    """Set the matplotlib defaults used by every figure in the project."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save_figure(figure: plt.Figure, directory: str | Path, name: str) -> Path:
    """Write a figure as both PNG and SVG.

    Both formats are written because a manuscript needs the vector version and a
    README needs the raster one.

    Args:
        figure: Figure to write.
        directory: Destination directory, created if absent.
        name: File stem, without an extension.

    Returns:
        The path of the written PNG.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    png = target / f"{name}.png"
    figure.savefig(png)
    figure.savefig(target / f"{name}.svg")
    plt.close(figure)
    logger.info("wrote %s", png)
    return png


def plot_model_comparison(
    per_fold: Mapping[str, np.ndarray],
    directory: str | Path,
    name: str = "model_comparison",
    metric: str = "macro-F1",
    references: Mapping[str, float] | None = None,
) -> Path:
    """Draw per-fold scores per model as a box plot with the individual folds.

    Points are drawn over the boxes because at twenty-five estimates the reader
    should see the actual spread rather than a five-number summary of it.

    Args:
        per_fold: Mapping of model name to its per-fold scores.
        directory: Destination directory.
        name: File stem.
        metric: Axis label.
        references: Optional horizontal reference lines, by label.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    ordered = sorted(per_fold, key=lambda k: float(np.nanmean(per_fold[k])))
    values = [np.asarray(per_fold[model], dtype=float) for model in ordered]

    figure, axes = plt.subplots(figsize=(7.5, 0.42 * len(ordered) + 2.2))
    axes.boxplot(values, orientation="horizontal", widths=0.6, showfliers=False)

    rng = np.random.default_rng(0)
    for position, scores in enumerate(values, start=1):
        jitter = rng.uniform(-0.14, 0.14, len(scores))
        axes.scatter(
            scores,
            np.full(len(scores), position) + jitter,
            s=14,
            alpha=0.55,
            color=OKABE_ITO["blue"],
            zorder=3,
        )

    for label, value in (references or {}).items():
        axes.axvline(value, color=OKABE_ITO["vermilion"], ls="--", lw=1.2, zorder=1)
        axes.text(
            value,
            len(ordered) + 0.6,
            f" {label} {value:.3f}",
            color=OKABE_ITO["vermilion"],
            fontsize=8,
            va="bottom",
        )

    axes.set_yticks(range(1, len(ordered) + 1))
    axes.set_yticklabels(ordered)
    axes.set_xlabel(f"{metric} ({len(values[0])} folds)")
    axes.set_title("Model comparison on the shared partition")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_interval_forest(
    summary: pd.DataFrame,
    directory: str | Path,
    name: str = "interval_forest",
    metric: str = "macro_f1",
) -> Path:
    """Draw each model's mean with both confidence intervals.

    The naive and corrected intervals are drawn as nested bars so the cost of
    the fold overlap is visible at a glance rather than buried in a table.

    Args:
        summary: Aggregated frame with ``model``, ``metric``, ``mean`` and the
            two interval pairs.
        directory: Destination directory.
        name: File stem.
        metric: Metric to draw.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    rows = summary[summary["metric"] == metric].sort_values("mean")

    figure, axes = plt.subplots(figsize=(7.5, 0.4 * len(rows) + 2.0))
    positions = np.arange(len(rows))

    axes.hlines(
        positions,
        rows["nb_ci95_lo"],
        rows["nb_ci95_hi"],
        color=OKABE_ITO["grey"],
        lw=2.2,
        label="Nadeau-Bengio 95%",
    )
    axes.hlines(
        positions,
        rows["ci95_lo"],
        rows["ci95_hi"],
        color=OKABE_ITO["blue"],
        lw=4.0,
        label="naive 95%",
    )
    axes.scatter(rows["mean"], positions, color=OKABE_ITO["black"], zorder=4, s=26)

    axes.set_yticks(positions)
    axes.set_yticklabels(rows["model"])
    axes.set_xlabel(metric)
    axes.set_title("Mean and interval per model")
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_per_class_f1(
    summary: pd.DataFrame,
    class_names: Sequence[str],
    directory: str | Path,
    name: str = "per_class_f1",
) -> Path:
    """Draw per-class F1 for every model.

    Always drawn beside the macro average, because a macro score can look
    healthy while the rarest class has collapsed to zero — which is the dominant
    failure mode in this cohort.

    Args:
        summary: Aggregated frame carrying ``f1__<class>`` metric rows.
        class_names: Classes to draw, in order.
        directory: Destination directory.
        name: File stem.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    models = sorted(summary["model"].unique())
    width = 0.8 / max(len(models), 1)
    colours = list(OKABE_ITO.values())

    figure, axes = plt.subplots(figsize=(1.6 * len(class_names) + 3.0, 4.2))
    for index, model in enumerate(models):
        means = [
            summary[
                (summary["model"] == model) & (summary["metric"] == f"f1__{name_}")
            ]["mean"].squeeze()
            for name_ in class_names
        ]
        axes.bar(
            np.arange(len(class_names)) + index * width,
            means,
            width,
            label=model,
            color=colours[index % len(colours)],
        )

    axes.set_xticks(np.arange(len(class_names)) + 0.4 - width / 2)
    axes.set_xticklabels(class_names, rotation=20, ha="right")
    axes.set_ylabel("F1")
    axes.set_ylim(0, 1)
    axes.set_title("Per-class F1 — where the macro average comes from")
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_critical_difference(
    average_ranks: pd.Series,
    critical_difference: float,
    directory: str | Path,
    name: str = "critical_difference",
) -> Path:
    """Draw a critical-difference diagram over average ranks.

    Models whose average ranks differ by less than the critical difference are
    statistically indistinguishable; the bar makes that tie explicit instead of
    inviting a reader to rank them anyway.

    Args:
        average_ranks: Average rank per model, best first.
        critical_difference: Nemenyi critical difference.
        directory: Destination directory.
        name: File stem.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    ranks = average_ranks.sort_values()

    figure, axes = plt.subplots(figsize=(7.5, 0.34 * len(ranks) + 2.2))
    positions = np.arange(len(ranks))
    axes.scatter(ranks.to_numpy(), positions, color=OKABE_ITO["blue"], s=40, zorder=3)

    for position, (model, rank) in enumerate(ranks.items()):
        axes.text(rank, position + 0.22, f" {model}", fontsize=9, va="bottom")

    best = float(ranks.iloc[0])
    axes.hlines(
        -0.9,
        best,
        best + critical_difference,
        color=OKABE_ITO["vermilion"],
        lw=3.0,
    )
    axes.text(
        best + critical_difference / 2,
        -1.25,
        f"critical difference = {critical_difference:.2f}",
        color=OKABE_ITO["vermilion"],
        fontsize=8,
        ha="center",
    )

    axes.set_yticks([])
    axes.set_xlabel("average rank (lower is better)")
    axes.set_title("Critical-difference diagram")
    axes.set_ylim(-1.8, len(ranks))
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_confusion(
    matrix: np.ndarray,
    class_names: Sequence[str],
    directory: str | Path,
    name: str = "confusion",
    normalise: bool = True,
) -> Path:
    """Draw a confusion matrix, row-normalised by default.

    Row normalisation shows recall per true class, which is what matters under
    heavy imbalance; raw counts make the majority class dominate the picture.

    Args:
        matrix: Confusion counts, shape ``(n_classes, n_classes)``.
        class_names: Class labels in matrix order.
        directory: Destination directory.
        name: File stem.
        normalise: Divide each row by its total.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    values = matrix.astype(float)
    if normalise:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    figure, axes = plt.subplots(figsize=(1.0 * len(class_names) + 3.0,) * 2)
    image = axes.imshow(values, cmap="Blues", vmin=0, vmax=values.max() or 1)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            axes.text(
                j,
                i,
                f"{values[i, j]:.2f}" if normalise else f"{int(matrix[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[i, j] > values.max() * 0.6 else "black",
            )

    axes.set_xticks(range(len(class_names)))
    axes.set_xticklabels(class_names, rotation=45, ha="right")
    axes.set_yticks(range(len(class_names)))
    axes.set_yticklabels(class_names)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title("Confusion" + (" (row-normalised)" if normalise else ""))
    axes.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return save_figure(figure, directory, name)
