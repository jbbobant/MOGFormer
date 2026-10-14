"""Figures for the subtyping benchmark, drawn from the experiments' saved scores.

Every figure reads the ``metrics_per_fold.csv`` files the benchmark wrote and
summarises them with the same functions the tables use, so a figure and a
table can never disagree. Intervals are Nadeau–Bengio corrected throughout,
because the 25 estimates of a repeated five-fold partition share training
patients and a naive interval would be too narrow.

Identity is never carried by colour alone: model families differ in marker
shape as well, and every mark that matters is labelled directly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from mogformer.config import ExperimentConfig
from mogformer.data.folds import load_folds
from mogformer.evaluation.metrics import aggregate_across_folds
from mogformer.evaluation.plots import (
    OKABE_ITO,
    apply_house_style,
    save_figure,
    set_figure_title,
)
from mogformer.evaluation.registry import build_preprocessor
from mogformer.evaluation.runner import build_context, load_cohort, load_per_fold_scores
from mogformer.evaluation.stats import corrected_resampled_ttest
from mogformer.experiments.paths import BundleLayout
from mogformer.graph.positional_encoding import (
    GraphPositionalEncoding,
    nearest_encoding_distance,
)

logger = logging.getLogger(__name__)

#: Primary metric of every figure.
METRIC = "macro_f1"

#: Folds per repeat of the shared partition.
N_SPLITS = 5

#: Repeats of the shared partition, used to turn pooled confusion counts back
#: into a patient count.
N_REPEATS = 5

#: Colour and marker per model family. Shape repeats the colour's information.
FAMILY_STYLE: dict[str, tuple[str, str]] = {
    "mogformer": (OKABE_ITO["blue"], "o"),
    "xgboost": (OKABE_ITO["vermilion"], "s"),
    "linear": (OKABE_ITO["green"], "^"),
    "classical": (OKABE_ITO["vermilion"], "v"),
    "reference": (OKABE_ITO["grey"], "D"),
}

#: Ink for text, never a data colour.
INK = "#222222"
MUTED_INK = "#666666"

#: Readable names for the classical registry keys.
BASELINE_LABELS: dict[str, tuple[str, str]] = {
    "B5_xgboost": ("XGBoost", "xgboost"),
    "B4_random_forest": ("Random forest", "classical"),
    "B2_elasticnet_logreg": ("Elastic-net logistic", "linear"),
    "B3_linear_svm": ("Linear SVM", "classical"),
    "B1_pam50_centroid": ("PAM50 centroid (expression)", "reference"),
}

#: Readable names for the transformer variants, by run-name stem.
TRANSFORMER_LABELS: dict[str, str] = {
    "pretrained_embedding": "MOGFormer, pretrained gene vectors",
    "learned_embedding_mask": "MOGFormer, learned gene identity",
    "gated_fusion": "MOGFormer, gated fusion",
    "attention_no_gene_identity": "MOGFormer, no gene identity",
}

#: Display names of the PAM50 subtypes, keyed by their label suffix.
SUBTYPE_NAMES: dict[str, str] = {
    "Basal": "Basal-like",
    "Her2": "HER2-enriched",
    "LumA": "Luminal A",
    "LumB": "Luminal B",
    "Normal": "Normal-like",
}

#: Modality subsets, in display order, with their baseline run names.
MODALITY_SUBSETS: tuple[tuple[str, str], ...] = (
    ("Expression + copy number + methylation", "tcga_5class"),
    ("Expression", "tcga_5class_rna_only"),
    ("Methylation", "tcga_5class_methy_only"),
    ("Copy number + methylation", "tcga_5class_cnv_methy"),
    ("Copy number", "tcga_5class_cnv_only"),
)


@dataclass(frozen=True)
class ScoredModel:
    """One model's per-fold scores and how to draw them.

    Attributes:
        label: Name shown on the figure.
        family: Key into :data:`FAMILY_STYLE`.
        scores: Per-fold macro-F1, ordered by repeat then fold.
    """

    label: str
    family: str
    scores: np.ndarray

    def interval(self) -> dict[str, float]:
        """Return the mean and its Nadeau–Bengio corrected 95% interval."""
        return aggregate_across_folds(self.scores, n_splits=N_SPLITS)


def read_scores(per_fold_csv: Path, model: str) -> np.ndarray | None:
    """Read one model's per-fold macro-F1 from a run's saved scores.

    Args:
        per_fold_csv: A ``metrics_per_fold.csv`` written by the runner.
        model: Registry key of the model.

    Returns:
        Scores ordered by repeat then fold, or None when the file or the
        model is absent.
    """
    if not per_fold_csv.exists():
        logger.warning("missing %s; its models are left out", per_fold_csv)
        return None
    return load_per_fold_scores(per_fold_csv, METRIC).get(model)


def count_patients(run_dir: Path) -> int | None:
    """Recover a run's cohort size from its pooled confusion counts.

    Args:
        run_dir: A run's output directory.

    Returns:
        Patients per repeat, or None when no confusion file exists.
    """
    matrices = sorted(run_dir.glob("confusion_*.csv"))
    if not matrices:
        return None
    total = np.loadtxt(matrices[0], delimiter=",").sum()
    return round(total / N_REPEATS)


def collect_benchmark(
    results_root: Path, class_config: str
) -> tuple[list[ScoredModel], int | None]:
    """Gather every model scored on one class configuration.

    Args:
        results_root: Directory holding ``baselines/`` and
            ``transformer_variants/``.
        class_config: ``"5class"`` or ``"4class"``.

    Returns:
        The scored models, and the cohort size of the configuration.
    """
    baseline_dir = results_root / "baselines" / f"tcga_{class_config}"
    models: list[ScoredModel] = []
    for key, (label, family) in BASELINE_LABELS.items():
        scores = read_scores(baseline_dir / "metrics_per_fold.csv", key)
        if scores is not None:
            models.append(ScoredModel(label, family, scores))

    for stem, label in TRANSFORMER_LABELS.items():
        run_dir = (
            results_root / "transformer_variants" / f"transformer_{stem}_{class_config}"
        )
        scores = read_scores(run_dir / "metrics_per_fold.csv", "M_mogformer")
        if scores is not None:
            models.append(ScoredModel(label, "mogformer", scores))
    return models, count_patients(baseline_dir)


def _draw_interval_rows(
    axes: plt.Axes, models: Sequence[ScoredModel], label_values: bool = True
) -> None:
    """Draw one row per model: mean marker, corrected interval, value label.

    Args:
        axes: Axes to draw on.
        models: Models in top-to-bottom order.
        label_values: Print each mean beside its interval.
    """
    positions = np.arange(len(models))[::-1]
    for position, model in zip(positions, models, strict=True):
        colour, marker = FAMILY_STYLE[model.family]
        summary = model.interval()
        axes.hlines(
            position,
            summary["nb_ci95_lo"],
            summary["nb_ci95_hi"],
            color=colour,
            lw=2,
            capstyle="round",
        )
        axes.scatter(
            summary["mean"],
            position,
            s=64,
            color=colour,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )
        if label_values:
            axes.text(
                summary["nb_ci95_hi"] + 0.006,
                position,
                f"{summary['mean']:.3f}",
                va="center",
                fontsize=8,
                color=INK,
            )
    axes.set_yticks(positions)
    axes.set_yticklabels([model.label for model in models], color=INK)
    axes.grid(axis="y", visible=False)


def plot_benchmark_forest(results_root: Path, directory: Path) -> Path | None:
    """Draw every model's macro-F1 on both class configurations.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no scores exist.
    """
    panels = []
    for class_config, title in (
        ("5class", "Five subtypes"),
        ("4class", "Four subtypes, Normal-like excluded"),
    ):
        models, n_patients = collect_benchmark(results_root, class_config)
        if models:
            models.sort(key=lambda m: -m.interval()["mean"])
            panels.append((title, models, n_patients))
    if not panels:
        return None

    apply_house_style()
    height = max(len(models) for _, models, _ in panels) * 0.42 + 1.6
    figure, axes_row = plt.subplots(
        1, len(panels), figsize=(12.5, height), sharex=True, squeeze=False
    )
    for axes, (title, models, n_patients) in zip(axes_row[0], panels, strict=True):
        _draw_interval_rows(axes, models)
        patients = f", {n_patients} patients" if n_patients else ""
        axes.set_title(f"{title}{patients}", fontsize=10, color=INK)
        axes.set_xlabel("macro-F1 (mean, Nadeau–Bengio 95% CI)", color=INK)
    axes_row[0][0].set_xlim(0.5, 0.99)
    figure.suptitle(
        "PAM50 subtyping on TCGA-BRCA: every model on one partition, "
        "25 estimates (5 repeats of 5-fold CV)",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.legend(
        handles=_family_handles(
            ["mogformer", "xgboost", "linear", "classical", "reference"]
        ),
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    figure.tight_layout()
    return save_figure(figure, directory, "benchmark_forest")


def _family_handles(families: Sequence[str]) -> list[plt.Line2D]:
    """Build legend handles naming each model family.

    Args:
        families: Keys into :data:`FAMILY_STYLE`.

    Returns:
        One handle per family.
    """
    names = {
        "mogformer": "MOGFormer",
        "xgboost": "Gradient-boosted trees",
        "linear": "Elastic-net logistic",
        "classical": "Other classical models",
        "reference": "Label-source reference",
    }
    return [
        plt.Line2D(
            [],
            [],
            color=FAMILY_STYLE[family][0],
            marker=FAMILY_STYLE[family][1],
            lw=2,
            markersize=8,
            markeredgecolor="white",
            label=names[family],
        )
        for family in families
    ]


def collect_modality_models(
    results_root: Path,
) -> list[tuple[str, list[ScoredModel]]]:
    """Gather the modality-subset scores, one group per subset.

    Args:
        results_root: Directory holding the benchmark outputs.

    Returns:
        ``(subset label, models)`` pairs in display order.
    """
    transformer_runs = {
        "tcga_5class": "transformer_pretrained_embedding_5class",
        "tcga_5class_rna_only": "transformer_pretrained_embedding_rna_only_5class",
    }
    groups = []
    for label, run in MODALITY_SUBSETS:
        per_fold = results_root / "baselines" / run / "metrics_per_fold.csv"
        models = []
        for key in ("B5_xgboost", "B2_elasticnet_logreg"):
            scores = read_scores(per_fold, key)
            if scores is not None:
                name, family = BASELINE_LABELS[key]
                models.append(ScoredModel(name, family, scores))
        if run in transformer_runs:
            scores = read_scores(
                results_root
                / "transformer_variants"
                / transformer_runs[run]
                / "metrics_per_fold.csv",
                "M_mogformer",
            )
            if scores is not None:
                models.append(
                    ScoredModel(
                        "MOGFormer, pretrained gene vectors", "mogformer", scores
                    )
                )
        if models:
            groups.append((label, models))
    return groups


def plot_modality_ablation(results_root: Path, directory: Path) -> Path | None:
    """Draw macro-F1 for every subset of the three omics used as input.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no scores exist.
    """
    groups = collect_modality_models(results_root)
    if not groups:
        return None

    apply_house_style()
    figure, axes = plt.subplots(figsize=(9.0, 0.9 * len(groups) + 1.6))
    offsets = {"xgboost": 0.22, "linear": 0.0, "mogformer": -0.22}
    for row, (_, models) in enumerate(groups):
        centre = len(groups) - 1 - row
        for model in models:
            colour, marker = FAMILY_STYLE[model.family]
            summary = model.interval()
            y = centre + offsets[model.family]
            axes.hlines(
                y, summary["nb_ci95_lo"], summary["nb_ci95_hi"], color=colour, lw=2
            )
            axes.scatter(
                summary["mean"],
                y,
                s=56,
                color=colour,
                marker=marker,
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
            )
            axes.text(
                summary["nb_ci95_hi"] + 0.005,
                y,
                f"{summary['mean']:.3f}",
                va="center",
                fontsize=7.5,
                color=INK,
            )
    axes.set_yticks(np.arange(len(groups))[::-1])
    axes.set_yticklabels([label for label, _ in groups], color=INK)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("macro-F1 (mean, Nadeau–Bengio 95% CI)", color=INK)
    axes.set_xlim(0.5, 0.95)
    axes.legend(
        handles=_family_handles(["xgboost", "linear", "mogformer"]),
        fontsize=8,
        loc="lower right",
    )
    set_figure_title(
        axes,
        "Five-subtype PAM50 macro-F1 by the omics given to the model",
        "TCGA-BRCA, 949 patients, 25 paired estimates; genes re-selected "
        "within each subset",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "modality_ablation")


def _draw_paired_differences(
    axes: plt.Axes, rows: Sequence[tuple[str, str, np.ndarray]]
) -> None:
    """Draw per-fold paired differences with their corrected mean interval.

    Args:
        axes: Axes to draw on.
        rows: ``(label, family, differences)`` per row, top to bottom.
    """
    rng = np.random.default_rng(0)
    label_x = max(float(np.nanmax(d)) for _, _, d in rows) + 0.012
    for row, (_label, family, differences) in enumerate(rows):
        y = len(rows) - 1 - row
        colour, marker = FAMILY_STYLE[family]
        summary = aggregate_across_folds(differences, n_splits=N_SPLITS)
        test = corrected_resampled_ttest(differences, n_splits=N_SPLITS)
        axes.scatter(
            differences,
            y + rng.uniform(-0.18, 0.18, len(differences)),
            s=14,
            color=colour,
            alpha=0.35,
            edgecolors="none",
        )
        axes.hlines(
            y, summary["nb_ci95_lo"], summary["nb_ci95_hi"], color=colour, lw=2.5
        )
        axes.scatter(
            summary["mean"],
            y,
            s=70,
            color=colour,
            marker=marker,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )
        axes.text(
            label_x,
            y,
            f"{summary['mean']:+.3f}  (corrected p = {test['p_value']:.2g})",
            va="center",
            fontsize=8,
            color=INK,
        )
    axes.axvline(0.0, color=MUTED_INK, lw=1)
    axes.set_yticks(np.arange(len(rows))[::-1])
    axes.set_yticklabels([label for label, _, _ in rows], color=INK)
    axes.grid(axis="y", visible=False)
    lo, _ = axes.get_xlim()
    axes.set_xlim(lo, label_x + 0.1)


def plot_paired_modality_gain(results_root: Path, directory: Path) -> Path | None:
    """Draw the per-fold gain from adding copy number and methylation.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no pair of runs exists.
    """
    base = results_root / "baselines"
    pairs = [
        (
            "XGBoost",
            "xgboost",
            read_scores(base / "tcga_5class" / "metrics_per_fold.csv", "B5_xgboost"),
            read_scores(
                base / "tcga_5class_rna_only" / "metrics_per_fold.csv", "B5_xgboost"
            ),
        ),
        (
            "Elastic-net logistic",
            "linear",
            read_scores(
                base / "tcga_5class" / "metrics_per_fold.csv", "B2_elasticnet_logreg"
            ),
            read_scores(
                base / "tcga_5class_rna_only" / "metrics_per_fold.csv",
                "B2_elasticnet_logreg",
            ),
        ),
        (
            "MOGFormer, pretrained gene vectors",
            "mogformer",
            read_scores(
                results_root
                / "transformer_variants"
                / "transformer_pretrained_embedding_5class"
                / "metrics_per_fold.csv",
                "M_mogformer",
            ),
            read_scores(
                results_root
                / "transformer_variants"
                / "transformer_pretrained_embedding_rna_only_5class"
                / "metrics_per_fold.csv",
                "M_mogformer",
            ),
        ),
    ]
    pairs = [p for p in pairs if p[2] is not None and p[3] is not None]
    if not pairs:
        return None

    apply_house_style()
    figure, axes = plt.subplots(figsize=(8.6, 0.85 * len(pairs) + 1.7))
    _draw_paired_differences(
        axes, [(p[0], p[1], np.asarray(p[2]) - np.asarray(p[3])) for p in pairs]
    )
    axes.set_xlabel(
        "macro-F1 with all three omics minus macro-F1 with expression only", color=INK
    )
    set_figure_title(
        axes,
        "Adding copy number and methylation to expression does not improve "
        "PAM50 subtyping",
        "TCGA-BRCA, five subtypes; dots are the 25 paired estimates, bars the "
        "Nadeau–Bengio 95% CI of the mean difference",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "paired_modality_gain")


#: Single-change transformer ablations: label, run changed, reference run.
TRANSFORMER_ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("gated fusion instead of attention", "gated_fusion", "learned_embedding_mask"),
    (
        "no gene identity (and zero dropout fill)",
        "attention_no_gene_identity",
        "learned_embedding_mask",
    ),
)


def plot_transformer_ablations(results_root: Path, directory: Path) -> Path | None:
    """Draw the paired effect of each single-setting change to the transformer.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no pair of runs exists.
    """
    runs = results_root / "transformer_variants"
    rows = []
    for class_config, subtypes in (("5class", "five"), ("4class", "four")):
        for label, changed, reference in TRANSFORMER_ABLATIONS:
            after = read_scores(
                runs / f"transformer_{changed}_{class_config}" / "metrics_per_fold.csv",
                "M_mogformer",
            )
            before = read_scores(
                runs
                / f"transformer_{reference}_{class_config}"
                / "metrics_per_fold.csv",
                "M_mogformer",
            )
            if after is not None and before is not None:
                rows.append(
                    (f"{subtypes} subtypes: {label}", "mogformer", after - before)
                )
    if not rows:
        return None

    apply_house_style()
    figure, axes = plt.subplots(figsize=(9.2, 0.75 * len(rows) + 1.8))
    _draw_paired_differences(axes, rows)
    axes.set_xlabel(
        "macro-F1 after the change minus macro-F1 of MOGFormer with learned gene "
        "identity",
        color=INK,
    )
    set_figure_title(
        axes,
        "Gated fusion matches attention; removing gene identity costs a quarter "
        "of macro-F1",
        "TCGA-BRCA; dots are the 25 paired estimates, bars the Nadeau–Bengio 95% "
        "CI of the mean difference",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "transformer_ablations")


def per_class_f1_by_subset(results_root: Path, model: str) -> pd.DataFrame:
    """Tabulate one model's mean per-class F1 for every modality subset.

    Args:
        results_root: Directory holding the benchmark outputs.
        model: Registry key of the model.

    Returns:
        Frame indexed by subset label, one column per subtype.
    """
    rows = {}
    for label, run in MODALITY_SUBSETS:
        path = results_root / "baselines" / run / "metrics_per_fold.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[
            (frame["model"] == model) & frame["metric"].str.startswith("f1__")
        ]
        means = frame.groupby("metric")["value"].mean()
        means.index = [
            name.removeprefix("f1__").removeprefix("BRCA_") for name in means.index
        ]
        rows[label] = means
    return pd.DataFrame(rows).T


def plot_per_class_heatmap(results_root: Path, directory: Path) -> Path | None:
    """Draw per-subtype F1 for each modality subset as a heatmap.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no scores exist.
    """
    table = per_class_f1_by_subset(results_root, "B5_xgboost")
    if table.empty:
        return None
    order = [name for name in SUBTYPE_NAMES if name in table]
    table = table[order]

    apply_house_style()
    figure, axes = plt.subplots(figsize=(8.8, 0.6 * len(table) + 2.3))
    values = table.to_numpy(dtype=float)
    image = axes.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            axes.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value > 0.6 else INK,
            )
    axes.set_xticks(range(len(order)))
    axes.set_xticklabels(
        [SUBTYPE_NAMES[name] for name in order], color=INK, rotation=25, ha="right"
    )
    axes.set_yticks(range(len(table)))
    axes.set_yticklabels(table.index, color=INK)
    axes.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02, label="F1")
    set_figure_title(
        axes,
        "Per-subtype F1 of gradient-boosted trees by the omics given to the model",
        "TCGA-BRCA, 949 patients, mean over 25 estimates",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "per_class_by_modality")


def plot_gene_identity(
    results_root: Path,
    directory: Path,
    nearest_distance: np.ndarray | None = None,
) -> Path | None:
    """Draw what removing the gene-identity embedding costs, and why.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.
        nearest_distance: Per-gene distance to the nearest other gene's
            positional encoding, relative to the encodings' spread, for one
            fold. The right-hand panel is drawn only when given.

    Returns:
        The written PNG, or None when the paired runs are absent.
    """
    runs = results_root / "transformer_variants"
    rows = []
    for class_config, subtypes in (("5class", "five"), ("4class", "four")):
        for stem, label in (
            ("learned_embedding_mask", "learned gene identity"),
            ("attention_no_gene_identity", "no gene identity"),
        ):
            scores = read_scores(
                runs / f"transformer_{stem}_{class_config}" / "metrics_per_fold.csv",
                "M_mogformer",
            )
            if scores is not None:
                rows.append(
                    ScoredModel(f"{subtypes} subtypes, {label}", "mogformer", scores)
                )
    if not rows:
        return None

    apply_house_style()
    width = 12.5 if nearest_distance is not None else 7.0
    columns = 2 if nearest_distance is not None else 1
    figure, axes_row = plt.subplots(1, columns, figsize=(width, 3.6), squeeze=False)
    left = axes_row[0][0]
    _draw_interval_rows(left, rows)
    left.set_xlabel("macro-F1 (mean, Nadeau–Bengio 95% CI)", color=INK)
    left.set_title(
        "MOGFormer with and without a gene-identity embedding", fontsize=10, color=INK
    )

    if nearest_distance is not None:
        right = axes_row[0][1]
        values = np.sort(np.asarray(nearest_distance, dtype=float))
        fraction = np.arange(1, len(values) + 1) / len(values)
        right.step(values, fraction, where="post", color=OKABE_ITO["blue"], lw=2)
        right.axvline(0.05, color=MUTED_INK, lw=1)
        within = int((values < 0.05).sum())
        right.text(
            0.055,
            0.08,
            f"{within} of {len(values)} genes have another gene\n"
            "within 5% of the encoding spread",
            fontsize=8,
            color=INK,
        )
        right.set_xlabel(
            "distance to the nearest other gene's positional encoding "
            "(fraction of spread)",
            color=INK,
        )
        right.set_ylabel("cumulative fraction of genes", color=INK)
        right.set_title(
            "Without identity, genes differ only by their graph encoding",
            fontsize=10,
            color=INK,
        )
    figure.suptitle(
        "Gene identity is necessary: the interaction graph alone cannot tell "
        "the selected genes apart",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout()
    return save_figure(figure, directory, "gene_identity")


#: Trans-probe outputs per cohort, relative to the experiments directory.
TRANS_PROBE_RUNS: tuple[tuple[str, str], ...] = (
    ("TCGA-BRCA", "sign_concordance/TCGA"),
    ("METABRIC", "external_trans_sweep/metabric"),
)

#: Display names of the two masking modes of the trans probe.
MASK_MODE_NAMES: dict[str, str] = {
    "rna_only": "targets' expression hidden",
    "whole_gene": "targets fully hidden",
}


def observed_sign_concordance(edges: pd.DataFrame) -> float:
    """Measure how often observed co-expression carries a curated edge's sign.

    The data-side counterpart of the model's sign concordance: if the cohort's
    own expression does not show an edge's direction, a model trained on it
    has nothing to recover.

    Args:
        edges: Edge rows carrying ``edge_sign`` and ``observed_coexpression``.

    Returns:
        Fraction of edges with a non-zero co-expression whose sign matches.
    """
    observed = np.sign(edges["observed_coexpression"].to_numpy(dtype=float))
    curated = np.sign(edges["edge_sign"].to_numpy(dtype=float))
    usable = np.isfinite(observed) & (observed != 0)
    return float((observed[usable] == curated[usable]).mean())


def plot_trans_probe_cohorts(results_root: Path, directory: Path) -> Path | None:
    """Draw what the in-silico trans probe recovers, cohort by cohort.

    Args:
        results_root: Directory holding the trans-probe outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no cohort has been probed.
    """
    rows = []
    for cohort, relative in TRANS_PROBE_RUNS:
        run = results_root / relative
        record_path = run / "sign_tests.json"
        if not record_path.exists():
            logger.warning("missing %s; that cohort is left out", record_path)
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for mode, entry in record["modes"].items():
            edges = pd.read_csv(run / f"edges_{mode}.csv")
            rows.append(
                {
                    "label": f"{cohort}, {MASK_MODE_NAMES.get(mode, mode)}",
                    "cohort": cohort,
                    "mode": mode,
                    "sign": entry["sign_concordance"],
                    "magnitude": entry["magnitude"],
                    "graph_nulls": entry.get("graph_nulls", {}),
                    "observed": observed_sign_concordance(edges),
                }
            )
    if not rows:
        return None

    apply_house_style()
    figure, (signs, magnitude, graph) = plt.subplots(
        1,
        3,
        figsize=(15.5, 0.75 * len(rows) + 3.0),
        gridspec_kw={"width_ratios": [2.4, 1.2, 1.1]},
    )
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows, strict=True):
        sign = row["sign"]
        low = sign["null_mean_rate"] - 2 * sign["null_sd_rate"]
        high = sign["null_mean_rate"] + 2 * sign["null_sd_rate"]
        signs.hlines(position, low, high, color=OKABE_ITO["grey"], lw=7, alpha=0.45)
        signs.scatter(
            sign["trivial_majority_baseline"],
            position,
            marker="|",
            s=260,
            color=INK,
            linewidths=2.2,
            zorder=3,
        )
        signs.scatter(
            row["observed"],
            position,
            marker="D",
            s=52,
            color=OKABE_ITO["grey"],
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )
        signs.scatter(
            sign["rate"],
            position,
            marker="o",
            s=70,
            color=OKABE_ITO["blue"],
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        signs.text(
            0.84,
            position,
            f"{sign['rate']:.3f}  (perm. p = {sign['perm_p']:.2f}, "
            f"{sign['n_edges']} edges)",
            va="center",
            fontsize=8,
            color=INK,
        )
    signs.set_xlim(0.45, 1.14)
    signs.set_xticks(np.arange(0.5, 1.01, 0.1))
    signs.set_yticks(positions)
    signs.set_yticklabels([row["label"] for row in rows], color=INK)
    signs.grid(axis="y", visible=False)
    signs.set_xlabel("fraction of curated edge signs matched", color=INK)
    signs.set_title("Sign of the effect", fontsize=10, color=INK)
    ratios = [
        row["magnitude"]["edge_abs_median"] / row["magnitude"]["control_abs_median"]
        for row in rows
    ]
    magnitude.scatter(
        ratios,
        positions,
        marker="o",
        s=70,
        color=OKABE_ITO["blue"],
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    for position, ratio, row in zip(positions, ratios, rows, strict=True):
        magnitude.text(
            ratio + 0.04,
            position,
            f"{ratio:.2f}x  (p = {row['magnitude']['mw_p_edge_stronger']:.0e})",
            va="center",
            fontsize=8,
            color=INK,
        )
    magnitude.axvline(1.0, color=MUTED_INK, lw=1)
    magnitude.set_xlim(0.8, max(ratios) + 0.9)
    magnitude.set_yticks(positions)
    magnitude.set_yticklabels([])
    magnitude.grid(axis="y", visible=False)
    magnitude.set_xlabel("|slope| ratio, edges / non-edges", color=INK)
    magnitude.set_title("Size of the effect", fontsize=10, color=INK)

    controls = (
        ("permute_grn_only", "regulatory graph shuffled", OKABE_ITO["blue"], "o"),
        (
            "permute_graph",
            "interaction + regulatory graphs shuffled",
            OKABE_ITO["orange"],
            "s",
        ),
    )
    for position, row in zip(positions, rows, strict=True):
        measured = [c for c in controls if c[0] in row["graph_nulls"]]
        if not measured:
            graph.text(
                2, position, "not measured", va="center", fontsize=8, color=MUTED_INK
            )
            continue
        for offset, (label, _, colour, marker) in zip(
            (0.14, -0.14), controls, strict=True
        ):
            if label not in row["graph_nulls"]:
                continue
            value = row["graph_nulls"][label]["graph_attributable_pct"]
            y = position + offset
            graph.hlines(y, 0, value, color=colour, lw=2)
            graph.scatter(
                value,
                y,
                marker=marker,
                s=56,
                color=colour,
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
            )
            graph.text(
                max(value, 0) + 3,
                y,
                f"{value:.0f}%",
                va="center",
                fontsize=8,
                color=INK,
            )
    graph.axvline(0, color=MUTED_INK, lw=1)
    graph.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=colour,
                marker=marker,
                lw=2,
                markersize=7,
                markeredgecolor="white",
                label=name,
            )
            for _, name, colour, marker in controls
        ],
        fontsize=7.5,
        loc="center right",
        title="control",
        title_fontsize=7.5,
    )
    graph.set_xlim(-15, 100)
    graph.set_yticks(positions)
    graph.set_yticklabels([])
    graph.grid(axis="y", visible=False)
    graph.set_xlabel("% of effect lost when the graph is shuffled", color=INK)
    graph.set_title("Attributable to the graph", fontsize=10, color=INK)

    figure.suptitle(
        "In-silico perturbation of a transcription factor: the curated sign is "
        "recovered in neither cohort, and the regulatory graph carries almost "
        "none of the effect",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=OKABE_ITO["blue"],
                markersize=8,
                label="model's trans slope",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                ls="",
                color=OKABE_ITO["grey"],
                markersize=7,
                label="observed co-expression",
            ),
            plt.Line2D(
                [],
                [],
                lw=7,
                alpha=0.45,
                color=OKABE_ITO["grey"],
                label="label-permutation null, ±2 s.d.",
            ),
            plt.Line2D(
                [],
                [],
                marker="|",
                ls="",
                color=INK,
                markersize=14,
                markeredgewidth=2.2,
                label="always 'activating'",
            ),
        ],
        fontsize=8,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return save_figure(figure, directory, "trans_probe_cohorts")


def sign_agreements(edges: pd.DataFrame) -> dict[str, float]:
    """Compare three sign sources on the same edges: model, data and curation.

    Args:
        edges: Edge rows carrying ``slope``, ``observed_coexpression`` and
            ``edge_sign``.

    Returns:
        Agreement rates between each pair of sign sources, computed over the
        edges whose co-expression is non-zero.
    """
    model = np.sign(edges["slope"].to_numpy(dtype=float))
    data = np.sign(edges["observed_coexpression"].to_numpy(dtype=float))
    curated = np.sign(edges["edge_sign"].to_numpy(dtype=float))
    usable = np.isfinite(data) & (data != 0)
    model, data, curated = model[usable], data[usable], curated[usable]
    return {
        "model_vs_data": float((model == data).mean()),
        "model_vs_curated": float((model == curated).mean()),
        "data_vs_curated": float((data == curated).mean()),
    }


def plot_trans_sign_sources(results_root: Path, directory: Path) -> Path | None:
    """Draw whether the probe's signs follow the data or the curated network.

    Args:
        results_root: Directory holding the trans-probe outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no cohort has been probed.
    """
    rows = []
    for cohort, relative in TRANS_PROBE_RUNS:
        edges_path = results_root / relative / "edges_whole_gene.csv"
        if edges_path.exists():
            edges = pd.read_csv(edges_path)
            rows.append((cohort, len(edges), sign_agreements(edges)))
    if not rows:
        return None

    pairs = (
        ("model_vs_data", "model vs observed co-expression", OKABE_ITO["blue"], "o"),
        (
            "data_vs_curated",
            "observed co-expression vs curated",
            OKABE_ITO["grey"],
            "D",
        ),
        ("model_vs_curated", "model vs curated", OKABE_ITO["vermilion"], "s"),
    )
    apply_house_style()
    figure, axes = plt.subplots(figsize=(9.4, 1.3 * len(rows) + 1.9))
    offsets = (0.22, 0.0, -0.22)
    for row, (_cohort, _n, rates) in enumerate(rows):
        centre = len(rows) - 1 - row
        for offset, (key, _label, colour, marker) in zip(offsets, pairs, strict=True):
            y = centre + offset
            axes.hlines(y, 0.5, rates[key], color=colour, lw=2)
            axes.scatter(
                rates[key],
                y,
                s=64,
                color=colour,
                marker=marker,
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
            )
            axes.text(
                rates[key] + 0.006,
                y,
                f"{rates[key]:.2f}",
                va="center",
                fontsize=8,
                color=INK,
            )
    axes.axvline(0.5, color=MUTED_INK, lw=1)
    axes.set_xlim(0.45, 0.92)
    axes.set_yticks(np.arange(len(rows))[::-1])
    axes.set_yticklabels([f"{cohort}\n{n} edges" for cohort, n, _ in rows], color=INK)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("fraction of edges on which the two signs agree", color=INK)
    axes.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=colour,
                marker=marker,
                lw=2,
                markersize=8,
                markeredgecolor="white",
                label=label,
            )
            for _, label, colour, marker in pairs
        ],
        fontsize=8,
        loc="lower right",
    )
    set_figure_title(
        axes,
        "The probe's effect signs follow observed co-expression, not the curated "
        "regulatory sign",
        "Transcription factor perturbed in silico, targets fully hidden; "
        "CollecTRI signed edges",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "trans_sign_sources")


#: Estimators drawn in the simulation benchmark: summary column, colour, marker.
SIMULATION_ESTIMATORS: tuple[tuple[str, str, str, str], ...] = (
    ("slope", "MOGFormer probe", OKABE_ITO["blue"], "o"),
    ("observed_coexpression", "co-expression", OKABE_ITO["grey"], "D"),
    ("ridge_effect", "ridge on all visible inputs", OKABE_ITO["green"], "s"),
    ("instrumental_effect", "copy-number instrument", OKABE_ITO["orange"], "^"),
)


def _simulation_row_label(name: str, row: pd.Series) -> str:
    """Describe one benchmark condition in words."""
    if pd.isna(row["confounding"]):
        cohort = name.replace(" (observed)", "")
        return f"{cohort}, observed\n{int(row['n_patients']):,} patients"
    graph = "graph given" if row["use_grn"] else "graph withheld"
    return (
        f"simulated, {int(row['n_patients']):,} patients\n"
        f"confounding {row['confounding']:.1f}, {graph}"
    )


def plot_simulation_benchmark(results_root: Path, directory: Path) -> Path | None:
    """Draw sign recovery on simulated cohorts beside the observed ones.

    Args:
        results_root: Directory holding the benchmark's ``summary.csv``.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when the benchmark has not been run.
    """
    summary_path = results_root / "simulation_benchmark" / "summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    observed = summary["confounding"].isna()
    # Without hidden programs the reconstruction objective stalls near the
    # per-gene mean and early stopping ends training there; those encoders
    # never learned the regulation the probe reads, so they are not shown.
    simulated = summary[~observed & (summary["confounding"] > 0)].sort_values(
        ["confounding", "n_patients", "use_grn"], ascending=[False, True, False]
    )
    ordered = pd.concat([simulated, summary[observed]])
    conditions = list(dict.fromkeys(ordered["condition"]))
    first = ordered.drop_duplicates("condition").set_index("condition")

    apply_house_style()
    figure, (signs, graph) = plt.subplots(
        1,
        2,
        figsize=(13.5, 0.62 * len(conditions) + 2.8),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    positions = {name: len(conditions) - 1 - i for i, name in enumerate(conditions)}
    offsets = np.linspace(0.24, -0.24, len(SIMULATION_ESTIMATORS))
    for offset, (column, _label, colour, marker) in zip(
        offsets, SIMULATION_ESTIMATORS, strict=True
    ):
        rows = ordered[ordered["column"] == column]
        y = [positions[name] + offset for name in rows["condition"]]
        signs.scatter(
            rows["curated_sign_rate"],
            y,
            s=46,
            color=colour,
            marker=marker,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
    for name in conditions:
        signs.scatter(
            first.loc[name, "trivial_majority_baseline"],
            positions[name],
            marker="|",
            s=420,
            color=INK,
            linewidths=2.0,
            zorder=2,
        )
    n_simulated = len(dict.fromkeys(simulated["condition"]))
    if 0 < n_simulated < len(conditions):
        for axes in (signs, graph):
            axes.axhline(len(conditions) - n_simulated - 0.5, color=MUTED_INK, lw=0.8)
    signs.axvline(0.5, color=MUTED_INK, lw=0.8, ls=":")
    signs.set_xlim(0.4, 1.0)
    signs.set_yticks([positions[name] for name in conditions])
    signs.set_yticklabels(
        [_simulation_row_label(name, first.loc[name]) for name in conditions],
        color=INK,
        fontsize=8,
    )
    signs.grid(axis="y", visible=False)
    signs.set_xlabel("fraction of curated edges whose sign is recovered", color=INK)
    signs.set_title("Sign of the effect", fontsize=10, color=INK)
    signs.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker=marker,
                ls="",
                color=colour,
                markersize=7,
                markeredgecolor="white",
                label=label,
            )
            for _, label, colour, marker in SIMULATION_ESTIMATORS
        ]
        + [
            plt.Line2D(
                [],
                [],
                marker="|",
                ls="",
                color=INK,
                markersize=14,
                markeredgewidth=2.0,
                label="always 'activating'",
            )
        ],
        fontsize=7.5,
        loc="lower right",
    )

    probe = ordered[ordered["column"] == "slope"]
    for _, row in probe.iterrows():
        centre = positions[row["condition"]]
        value = row["grn_attributable_pct"]
        if pd.isna(value):
            label = "graph withheld" if not row["use_grn"] else "not measured"
            graph.text(2, centre, label, va="center", fontsize=8, color=MUTED_INK)
            continue
        graph.hlines(centre, 0, value, color=OKABE_ITO["blue"], lw=2)
        graph.scatter(
            value,
            centre,
            marker="o",
            s=56,
            color=OKABE_ITO["blue"],
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        graph.text(
            max(value, 0) + 3,
            centre,
            f"{value:.0f}%",
            va="center",
            fontsize=8,
            color=INK,
        )
    graph.axvline(0, color=MUTED_INK, lw=0.8)
    graph.set_xlim(-15, 100)
    graph.set_yticks([positions[name] for name in conditions])
    graph.set_yticklabels([])
    graph.grid(axis="y", visible=False)
    graph.set_xlabel(
        "% of the probe's effect lost when the\nregulatory graph is shuffled",
        color=INK,
    )
    graph.set_title("Carried by the regulatory graph", fontsize=10, color=INK)
    graph.set_ylim(signs.get_ylim())

    figure.suptitle(
        "Transcription-factor signs on the CollecTRI graph: cohorts simulated with "
        "every sign true and TCGA-like confounding, beside the observed cohorts",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout()
    return save_figure(figure, directory, "simulation_sign_recovery")


#: METABRIC targets, by bundle directory: panel title stating what the label
#: is derived from.
METABRIC_TARGETS: tuple[tuple[str, str], ...] = (
    ("pam50", "PAM50 subtype\n(derived from expression)"),
    ("er_ihc", "ER status\n(protein, immunohistochemistry)"),
    ("intclust", "Integrative cluster\n(copy number with expression)"),
    ("her2_snp6", "HER2 status\n(ERBB2 copy number, SNP6)"),
)

#: Input subsets of the METABRIC runs: row label and run-name suffix.
METABRIC_SUBSETS: tuple[tuple[str, str], ...] = (
    ("All three omics", "all_omics"),
    ("Expression", "rna_only"),
    ("Copy number", "cnv_only"),
    ("Methylation", "methy_only"),
)

#: Models drawn per subset: label, registry key, directory, run-name suffix,
#: colour and marker.
METABRIC_MODELS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "MOGFormer, gated fusion",
        "M_mogformer",
        "transformer_variants",
        "_gated_fusion_smoothed_stopping",
        OKABE_ITO["blue"],
        "o",
    ),
    ("Linear SVM", "B3_linear_svm", "baselines", "", OKABE_ITO["green"], "^"),
    ("Random forest", "B4_random_forest", "baselines", "", OKABE_ITO["vermilion"], "v"),
)


def plot_target_by_modality(results_root: Path, directory: Path) -> Path | None:
    """Draw macro-F1 by input omics for METABRIC labels of different origin.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no METABRIC run exists.
    """
    panels = []
    for target, title in METABRIC_TARGETS:
        rows = []
        for subset_label, subset in METABRIC_SUBSETS:
            for label, key, folder, suffix, colour, marker in METABRIC_MODELS:
                run = results_root / folder / f"metabric_{target}_{subset}{suffix}"
                # MOGFormer was run on a subset of the grid; absent cells are
                # expected, not missing.
                if not (run / "metrics_per_fold.csv").exists():
                    continue
                scores = read_scores(run / "metrics_per_fold.csv", key)
                if scores is not None:
                    rows.append((subset_label, label, colour, marker, scores))
        chance = read_scores(
            results_root
            / "baselines"
            / f"metabric_{target}_all_omics"
            / "metrics_per_fold.csv",
            "B0a_dummy_stratified",
        )
        n_patients = count_patients(
            results_root / "baselines" / f"metabric_{target}_all_omics"
        )
        if n_patients is not None:
            title = f"{title}, n = {n_patients:,}"
        if rows:
            panels.append((title, rows, chance))
    if not panels:
        return None

    apply_house_style()
    figure, grid = plt.subplots(
        1, len(panels), figsize=(4.3 * len(panels) + 1.6, 4.6), sharey=True
    )
    axes_list = np.atleast_1d(grid)
    subset_rows = {
        name: len(METABRIC_SUBSETS) - 1 - i
        for i, (name, _) in enumerate(METABRIC_SUBSETS)
    }
    offsets = dict(
        zip(
            [m[0] for m in METABRIC_MODELS],
            np.linspace(0.22, -0.22, len(METABRIC_MODELS)),
            strict=True,
        )
    )
    for axes, (title, rows, chance) in zip(axes_list, panels, strict=True):
        for subset_label, label, colour, marker, scores in rows:
            summary = aggregate_across_folds(scores, n_splits=N_SPLITS)
            y = subset_rows[subset_label] + offsets[label]
            axes.hlines(
                y, summary["nb_ci95_lo"], summary["nb_ci95_hi"], color=colour, lw=2
            )
            axes.scatter(
                summary["mean"],
                y,
                s=46,
                color=colour,
                marker=marker,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
        if chance is not None:
            axes.axvline(float(np.mean(chance)), color=MUTED_INK, lw=1, ls=":")
        axes.set_title(title, fontsize=9.5, color=INK)
        axes.set_xlim(0.0, 1.0)
        axes.grid(axis="y", visible=False)
        axes.set_xlabel("macro-F1", color=INK)
    axes_list[0].set_yticks(list(subset_rows.values()))
    axes_list[0].set_yticklabels(list(subset_rows), color=INK)
    handles = [
        plt.Line2D(
            [],
            [],
            color=colour,
            marker=marker,
            lw=2,
            markersize=7,
            markeredgecolor="white",
            label=label,
        )
        for label, _, _, _, colour, marker in METABRIC_MODELS
    ] + [plt.Line2D([], [], color=MUTED_INK, lw=1, ls=":", label="stratified chance")]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(
        "METABRIC: which omics predicts a label depends on what the label is "
        "derived from (mean and Nadeau–Bengio 95% CI over 25 estimates)",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(figure, directory, "metabric_target_by_modality")


#: Pre-registered METABRIC contrasts: label, target, minuend subset, subtrahend
#: subset. Each is drawn as a paired difference, positive meaning the first
#: subset scores higher.
METABRIC_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("PAM50: all omics − expression", "pam50", "all_omics", "rna_only"),
    ("ER status: all omics − expression", "er_ihc", "all_omics", "rna_only"),
    ("ER status: copy number − methylation", "er_ihc", "cnv_only", "methy_only"),
    (
        "Integrative cluster: all omics − expression",
        "intclust",
        "all_omics",
        "rna_only",
    ),
    (
        "Integrative cluster: copy number − expression",
        "intclust",
        "cnv_only",
        "rna_only",
    ),
    ("HER2 status: copy number − expression", "her2_snp6", "cnv_only", "rna_only"),
    ("HER2 status: all omics − copy number", "her2_snp6", "all_omics", "cnv_only"),
)

#: Model families drawn in the contrast figure: name, registry key, results
#: folder, run-name suffix and style family.
CONTRAST_MODELS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "MOGFormer, gated fusion",
        "M_mogformer",
        "transformer_variants",
        "_gated_fusion_smoothed_stopping",
        "mogformer",
    ),
    ("Linear SVM", "B3_linear_svm", "baselines", "", "linear"),
    ("Random forest", "B4_random_forest", "baselines", "", "classical"),
)


def _metabric_scores(
    results_root: Path, folder: str, target: str, subset: str, suffix: str, key: str
) -> np.ndarray | None:
    """Read one METABRIC cell's per-fold macro-F1, or None when it was not run."""
    path = (
        results_root
        / folder
        / f"metabric_{target}_{subset}{suffix}"
        / "metrics_per_fold.csv"
    )
    return read_scores(path, key) if path.exists() else None


def plot_metabric_contrasts(results_root: Path, directory: Path) -> Path | None:
    """Draw each pre-registered METABRIC contrast as a paired difference.

    Args:
        results_root: Directory holding the benchmark outputs.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no contrast can be computed.
    """
    rows = []
    for label, target, first, second in METABRIC_CONTRASTS:
        for name, key, folder, suffix, family in CONTRAST_MODELS:
            a = _metabric_scores(results_root, folder, target, first, suffix, key)
            b = _metabric_scores(results_root, folder, target, second, suffix, key)
            if a is not None and b is not None:
                rows.append((f"{label}  ·  {name}", family, a - b))
    if not rows:
        return None

    apply_house_style()
    figure, axes = plt.subplots(figsize=(11.5, 0.42 * len(rows) + 2.2))
    _draw_paired_differences(axes, rows)
    for tick in axes.get_yticklabels():
        tick.set_fontsize(8)
    axes.set_xlabel("difference in macro-F1 between the two input subsets", color=INK)
    axes.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=FAMILY_STYLE[family][0],
                marker=FAMILY_STYLE[family][1],
                lw=2.5,
                markersize=7,
                markeredgecolor="white",
                label=name,
            )
            for name, _, _, _, family in CONTRAST_MODELS
        ],
        fontsize=8,
        loc="upper left",
    )
    set_figure_title(
        axes,
        "Pre-registered METABRIC contrasts: copy number wins only where the "
        "label is defined by copy number",
        "Paired over the target's 25 estimates; bars are Nadeau–Bengio 95% CIs "
        "of the mean difference",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "metabric_contrasts")


#: Transformer variants rerun with methylation as M-values, by run-name stem.
METHYLATION_VARIANTS: tuple[tuple[str, str], ...] = (
    ("MOGFormer, learned gene identity", "learned_embedding_mask"),
    ("MOGFormer, gated fusion", "gated_fusion"),
    ("MOGFormer, no gene identity", "attention_no_gene_identity"),
)


def plot_methylation_representation(results_root: Path, directory: Path) -> Path | None:
    """Draw how much reading methylation as M-values instead of beta changes.

    Args:
        results_root: Directory holding the transformer runs and the archived
            beta-value scores.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when either representation is missing.
    """
    archive = (
        results_root / "methylation_representation" / "tcga_transformer_scores_beta.csv"
    )
    if not archive.exists():
        return None
    beta = pd.read_csv(archive)
    beta = beta[beta["metric"] == METRIC]
    rows = []
    for class_config, subtypes in (("5class", "five"), ("4class", "four")):
        for label, stem in METHYLATION_VARIANTS:
            run = f"transformer_{stem}_{class_config}"
            m_values = read_scores(
                results_root / "transformer_variants" / run / "metrics_per_fold.csv",
                "M_mogformer",
            )
            reference = (
                beta[beta["run"] == run]
                .sort_values(["repeat", "fold"])["value"]
                .to_numpy(dtype=float)
            )
            if m_values is not None and len(reference) == len(m_values):
                rows.append(
                    (f"{subtypes} subtypes: {label}", "mogformer", m_values - reference)
                )
    if not rows:
        return None

    apply_house_style()
    figure, axes = plt.subplots(figsize=(9.4, 0.7 * len(rows) + 1.9))
    _draw_paired_differences(axes, rows)
    axes.set_xlabel(
        "macro-F1 with methylation as M-values minus as beta values", color=INK
    )
    set_figure_title(
        axes,
        "Reading methylation as M-values instead of beta values changes nothing "
        "measurable",
        "TCGA-BRCA; same folds and settings; dots are the 25 paired estimates, "
        "bars the Nadeau–Bengio 95% CI",
    )
    figure.tight_layout()
    return save_figure(figure, directory, "methylation_representation")


#: Runs compared in the stopping-rule figure: the original rule's run-name
#: suffix and the final one's.
STOPPING_RULES: tuple[str, str] = ("_gated_fusion", "_gated_fusion_smoothed_stopping")

#: Input subsets drawn in the stopping-rule figure: label, run-name part,
#: colour and marker. Copy number only: it is the input the rule was changed
#: for, and the only one whose original-rule runs did not read expression,
#: whose preprocessing was corrected afterwards.
STOPPING_SUBSETS: tuple[tuple[str, str, str, str], ...] = (
    ("Copy number", "cnv_only", OKABE_ITO["vermilion"], "s"),
)


def plot_stopping_rule(results_root: Path, directory: Path) -> Path | None:
    """Draw each fold's macro-F1 under the original and the final stopping rule.

    A fold where training collapsed onto the majority class sits far to the
    left of the diagonal under the original rule; the figure shows whether the
    final rule removes those folds without moving the others.

    Args:
        results_root: Directory holding ``transformer_variants/``.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no target has both runs.
    """
    original, final = STOPPING_RULES
    panels = []
    for target, title in METABRIC_TARGETS:
        points = []
        for label, subset, colour, marker in STOPPING_SUBSETS:
            stem = results_root / "transformer_variants" / f"metabric_{target}_{subset}"
            before = read_scores(
                Path(f"{stem}{original}") / "metrics_per_fold.csv", "M_mogformer"
            )
            after = read_scores(
                Path(f"{stem}{final}") / "metrics_per_fold.csv", "M_mogformer"
            )
            if before is not None and after is not None and len(before) == len(after):
                points.append((label, colour, marker, before, after))
        if points:
            panels.append((title, points))
    if not panels:
        return None

    apply_house_style()
    figure, axes_row = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 5.4), squeeze=False
    )
    for axes, (title, points) in zip(axes_row[0], panels, strict=True):
        values = np.concatenate([np.r_[p[3], p[4]] for p in points])
        low = max(0.0, float(values.min()) - 0.03)
        high = min(1.0, float(values.max()) + 0.03)
        axes.plot([low, high], [low, high], color=MUTED_INK, lw=1, ls=":", zorder=1)
        for label, colour, marker, before, after in points:
            axes.scatter(
                before,
                after,
                s=38,
                color=colour,
                marker=marker,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
                label=f"{label} (mean {before.mean():.3f} → {after.mean():.3f})",
            )
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.set_aspect("equal")
        axes.set_title(title, color=INK, fontsize=10, fontweight="bold")
        axes.set_xlabel("macro-F1, original rule (best single epoch)", color=INK)
        axes.set_ylabel(
            "macro-F1, final rule (≥100 epochs, 5-epoch average)", color=INK
        )
        axes.legend(loc="lower right", fontsize=8, frameon=False)
    figure.suptitle(
        "The final stopping rule removes collapsed folds and leaves the others "
        "in place\nMOGFormer (gated fusion) on METABRIC from copy number; one "
        "point per estimate (25 per target); dotted line: no change",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return save_figure(figure, directory, "stopping_rule")


def build_report_figures(
    results_root: Path | str,
    output_dir: Path | str,
    nearest_distance: np.ndarray | None = None,
) -> Mapping[str, str | None]:
    """Draw every benchmark figure the report uses.

    Args:
        results_root: Directory holding ``baselines/`` and
            ``transformer_variants/``.
        output_dir: Where the figures are written.
        nearest_distance: Optional per-gene positional-encoding separation for
            the gene-identity figure.

    Returns:
        Mapping of figure name to the written PNG, or None where the inputs
        were missing.
    """
    root, output = Path(results_root), Path(output_dir)
    written = {
        "benchmark_forest": plot_benchmark_forest(root, output),
        "modality_ablation": plot_modality_ablation(root, output),
        "paired_modality_gain": plot_paired_modality_gain(root, output),
        "transformer_ablations": plot_transformer_ablations(root, output),
        "per_class_by_modality": plot_per_class_heatmap(root, output),
        "gene_identity": plot_gene_identity(root, output, nearest_distance),
        "trans_probe_cohorts": plot_trans_probe_cohorts(root, output),
        "trans_sign_sources": plot_trans_sign_sources(root, output),
        "simulation_sign_recovery": plot_simulation_benchmark(root, output),
        "metabric_target_by_modality": plot_target_by_modality(root, output),
        "metabric_contrasts": plot_metabric_contrasts(root, output),
        "methylation_representation": plot_methylation_representation(root, output),
        "stopping_rule": plot_stopping_rule(root, output),
    }
    return {name: None if path is None else str(path) for name, path in written.items()}


def fold_encoding_separation(
    config: ExperimentConfig, layout: BundleLayout, fold_index: int = 0
) -> np.ndarray:
    """Measure positional-encoding separation of one fold's selected genes.

    Reproduces exactly what the transformer sees on that fold: the genes the
    fold-local preprocessing selects, their induced interaction subgraph, and
    the configured positional encoding.

    Args:
        config: Configuration rebased onto ``layout``.
        layout: The cohort bundle, pointing at the configuration's cohort file.
        fold_index: Which estimate of the shared partition to use.

    Returns:
        Relative nearest-neighbour encoding distance per selected gene.
    """
    data = load_cohort(config, layout.cohort)
    context = build_context(config, data, layout.read_pam50(), with_graph=True)
    if context.string_cache is None:
        raise ValueError("the configuration carries no interaction network")
    fold = load_folds(config.folds.path)[0][fold_index]

    preprocessor = build_preprocessor(context, context.active_modalities)
    preprocessor.fit(data.X[fold.train_idx], data.y[fold.train_idx])
    genes = list(preprocessor.selected_genes_)

    adjacency = torch.as_tensor(
        context.string_cache.induced_adjacency(genes), dtype=torch.float32
    )
    encodings = GraphPositionalEncoding(config.graph.pe_dim, config.graph.pe_method)(
        adjacency
    )
    return nearest_encoding_distance(encodings.numpy())
