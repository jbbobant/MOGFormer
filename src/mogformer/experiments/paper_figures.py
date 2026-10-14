"""Composite figures sized for a two-column manuscript.

The report figures are drawn one result per file, at a size meant for a screen
or a README. Scaled into a 7-inch page they shrink their text below legibility.
These figures regroup the same per-fold scores into multi-panel blocks drawn at
their printed size, with 7-point text, so that nothing is scaled when the
manuscript includes them. Every number is read from the saved per-fold scores,
through the same readers as the report figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib import pyplot as plt

from mogformer.evaluation.metrics import aggregate_across_folds
from mogformer.evaluation.plots import OKABE_ITO, apply_house_style, save_figure
from mogformer.experiments.report_figures import (
    INK,
    MUTED_INK,
    N_SPLITS,
    STOPPING_RULES,
    ScoredModel,
    _metabric_scores,
    collect_benchmark,
    collect_modality_models,
    read_scores,
)

#: One colour and marker per model, fixed across every panel of the
#: manuscript, so a symbol never names two models.
PAPER_MODEL_STYLE: dict[str, tuple[str, str]] = {
    "mogformer": (OKABE_ITO["blue"], "o"),
    "svm": (OKABE_ITO["green"], "^"),
    "forest": (OKABE_ITO["vermilion"], "v"),
    "xgboost": (OKABE_ITO["orange"], "s"),
    "elasticnet": (OKABE_ITO["purple"], "D"),
    "reference": (OKABE_ITO["grey"], "X"),
}

#: Paper style key for each classical model label used by the report readers.
PAPER_STYLE_OF_LABEL: dict[str, str] = {
    "XGBoost": "xgboost",
    "Random forest": "forest",
    "Elastic-net logistic": "elasticnet",
    "Linear SVM": "svm",
    "PAM50 centroid (expression)": "reference",
}

#: Width of a two-column page's text block, in inches.
PAGE_WIDTH = 7.0

#: Type sizes at print scale.
PAPER_STYLE: dict[str, Any] = {
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.5,
    # Saved at exactly the drawn size, so the page includes it unscaled.
    "savefig.bbox": "standard",
}

#: Models drawn on METABRIC: label, registry key, results folder, run-name
#: suffix and style family.
PAPER_METABRIC_MODELS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Linear SVM", "B3_linear_svm", "baselines", "", "svm"),
    ("Random forest", "B4_random_forest", "baselines", "", "forest"),
    (
        "MOGFormer (gated fusion)",
        "M_mogformer",
        "transformer_variants",
        "_gated_fusion_smoothed_stopping",
        "mogformer",
    ),
)

#: METABRIC targets: bundle name and a short title naming the label's origin.
PAPER_METABRIC_TARGETS: tuple[tuple[str, str], ...] = (
    ("pam50", "PAM50\n(from expression)"),
    ("er_ihc", "ER status\n(from protein)"),
    ("intclust", "IntClust\n(from CN + expression)"),
    ("her2_snp6", "HER2 status\n(from ERBB2 CN)"),
)

#: METABRIC input subsets: short label and run-name part.
PAPER_METABRIC_SUBSETS: tuple[tuple[str, str], ...] = (
    ("All three", "all_omics"),
    ("Expression", "rna_only"),
    ("Copy number", "cnv_only"),
    ("Methylation", "methy_only"),
)

#: Short labels for the TCGA modality subsets, in the report's order.
PAPER_TCGA_SUBSETS: dict[str, str] = {
    "Expression + copy number + methylation": "All three",
    "Expression": "Expression",
    "Methylation": "Methylation",
    "Copy number + methylation": "CN + methylation",
    "Copy number": "Copy number",
}

#: Pre-registered METABRIC contrasts: label, target, minuend and subtrahend.
PAPER_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("PAM50: all − expression", "pam50", "all_omics", "rna_only"),
    ("ER: all − expression", "er_ihc", "all_omics", "rna_only"),
    ("IntClust: all − expression", "intclust", "all_omics", "rna_only"),
    ("HER2: CN − expression", "her2_snp6", "cnv_only", "rna_only"),
    ("HER2: all − CN", "her2_snp6", "all_omics", "cnv_only"),
)

#: Single-setting changes to MOGFormer on five-subtype TCGA: label, changed
#: run stem, reference run stem.
PAPER_ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("Gated fusion", "gated_fusion", "learned_embedding_mask"),
    (
        "Pretrained identity",
        "pretrained_embedding",
        "learned_embedding_mask",
    ),
    (
        "No identity*",
        "attention_no_gene_identity",
        "learned_embedding_mask",
    ),
)

#: Display names of the TCGA benchmark models, shortened for a narrow panel.
SHORT_MODEL_NAMES: dict[str, str] = {
    "MOGFormer, pretrained gene vectors": "MOGFormer, pretrained identity",
    "MOGFormer, learned gene identity": "MOGFormer, learned identity",
    "MOGFormer, gated fusion": "MOGFormer, gated fusion",
    "MOGFormer, no gene identity": "MOGFormer, no identity",
    "PAM50 centroid (expression)": "PAM50 centroid",
}


def _style_key(model: ScoredModel) -> str:
    """Return the paper style key of a model read by the report readers."""
    if model.family == "mogformer":
        return "mogformer"
    return PAPER_STYLE_OF_LABEL[model.label]


def _panel_letter(axes: plt.Axes, letter: str, x: float = -0.02) -> None:
    """Write a bold panel letter at the top left of an axes."""
    axes.text(
        x,
        1.02,
        letter,
        transform=axes.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=INK,
    )


def _interval(
    axes: plt.Axes, y: float, scores: np.ndarray, family: str, size: float = 14
) -> dict[str, float]:
    """Draw one mean with its corrected 95% interval as a horizontal bar."""
    colour, marker = PAPER_MODEL_STYLE[family]
    summary = aggregate_across_folds(scores, n_splits=N_SPLITS)
    axes.hlines(y, summary["nb_ci95_lo"], summary["nb_ci95_hi"], color=colour, lw=1.3)
    axes.scatter(
        summary["mean"],
        y,
        s=size,
        color=colour,
        marker=marker,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    return summary


def _legend_handles(entries: Sequence[tuple[str, str]]) -> list[plt.Line2D]:
    """Build legend handles from ``(label, family)`` pairs."""
    return [
        plt.Line2D(
            [],
            [],
            color=PAPER_MODEL_STYLE[family][0],
            marker=PAPER_MODEL_STYLE[family][1],
            lw=1.3,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
        for label, family in entries
    ]


def _save(figure: plt.Figure, directory: Path, name: str) -> Path:
    """Write PNG and SVG through the house writer, plus a PDF for LaTeX."""
    png = save_figure(figure, directory, name)
    figure.savefig(Path(directory) / f"{name}.pdf")
    plt.close(figure)
    return png


def _metabric_panels(results_root: Path, axes_row: Sequence[plt.Axes]) -> bool:
    """Draw macro-F1 by input omics for each METABRIC target; True if any cell."""
    drawn = False
    offsets = np.linspace(0.22, -0.22, len(PAPER_METABRIC_MODELS))
    for column, (axes, (target, title)) in enumerate(
        zip(axes_row, PAPER_METABRIC_TARGETS, strict=True)
    ):
        for row, (_, subset) in enumerate(PAPER_METABRIC_SUBSETS):
            y = len(PAPER_METABRIC_SUBSETS) - 1 - row
            for offset, (_, key, folder, suffix, family) in zip(
                offsets, PAPER_METABRIC_MODELS, strict=True
            ):
                scores = _metabric_scores(
                    results_root, folder, target, subset, suffix, key
                )
                if scores is not None:
                    _interval(axes, y + offset, scores, family)
                    drawn = True
        chance = _metabric_scores(
            results_root, "baselines", target, "all_omics", "", "B0a_dummy_stratified"
        )
        if chance is not None:
            axes.axvline(float(np.mean(chance)), color=MUTED_INK, lw=0.7, ls=":")
        axes.set_title(title, color=INK)
        axes.set_xlim(0.0, 1.0)
        axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axes.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        axes.set_ylim(-0.6, len(PAPER_METABRIC_SUBSETS) - 0.4)
        axes.set_yticks(range(len(PAPER_METABRIC_SUBSETS))[::-1])
        axes.set_yticklabels([label for label, _ in PAPER_METABRIC_SUBSETS])
        axes.tick_params(axis="y", labelleft=column == 0)
        axes.grid(axis="y", visible=False)
        axes.set_xlabel("macro-F1")
    return drawn


def _tcga_subset_panel(results_root: Path, axes: plt.Axes) -> bool:
    """Draw TCGA five-subtype macro-F1 by input omics; True if anything drawn."""
    groups = collect_modality_models(results_root)
    if not groups:
        return False
    families = ("xgboost", "elasticnet", "mogformer")
    offsets = dict(zip(families, np.linspace(0.22, -0.22, 3), strict=True))
    for row, (_, models) in enumerate(groups):
        y = len(groups) - 1 - row
        for model in models:
            key = _style_key(model)
            _interval(axes, y + offsets[key], model.scores, key)
    axes.set_yticks(range(len(groups))[::-1])
    axes.set_yticklabels([PAPER_TCGA_SUBSETS.get(label, label) for label, _ in groups])
    axes.set_ylim(-0.6, len(groups) - 0.4)
    axes.set_xlim(0.5, 0.95)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("macro-F1")
    axes.set_title("TCGA PAM50, five subtypes", color=INK)
    return True


def _contrast_panel(results_root: Path, axes: plt.Axes) -> bool:
    """Draw the pre-registered METABRIC contrasts; True if anything drawn."""
    offsets = np.linspace(0.22, -0.22, len(PAPER_METABRIC_MODELS))
    drawn = False
    for row, (_, target, minuend, subtrahend) in enumerate(PAPER_CONTRASTS):
        y = len(PAPER_CONTRASTS) - 1 - row
        for offset, (_, key, folder, suffix, family) in zip(
            offsets, PAPER_METABRIC_MODELS, strict=True
        ):
            first = _metabric_scores(results_root, folder, target, minuend, suffix, key)
            second = _metabric_scores(
                results_root, folder, target, subtrahend, suffix, key
            )
            if first is not None and second is not None:
                _interval(axes, y + offset, first - second, family)
                drawn = True
    axes.axvline(0.0, color=MUTED_INK, lw=0.7)
    axes.set_yticks(range(len(PAPER_CONTRASTS))[::-1])
    axes.set_yticklabels([label for label, _, _, _ in PAPER_CONTRASTS])
    axes.set_ylim(-0.6, len(PAPER_CONTRASTS) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("paired difference in macro-F1")
    axes.set_title("METABRIC, pre-registered contrasts", color=INK)
    return drawn


def plot_label_provenance(results_root: Path, directory: Path) -> Path | None:
    """Draw which omics predicts each label, in METABRIC and TCGA.

    Panel a: METABRIC macro-F1 by input omics for four labels of different
    origin. Panel b: TCGA five-subtype PAM50 by input omics. Panel c: the
    pre-registered METABRIC contrasts as paired differences.

    Args:
        results_root: Directory holding ``baselines/`` and
            ``transformer_variants/``.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no METABRIC score exists.
    """
    apply_house_style()
    with plt.rc_context(cast(Any, PAPER_STYLE)):
        figure = plt.figure(figsize=(PAGE_WIDTH, 4.3))
        grid = figure.add_gridspec(
            2, 4, height_ratios=(1.0, 1.05), hspace=0.62, wspace=0.12
        )
        top = [figure.add_subplot(grid[0, i]) for i in range(4)]
        if not _metabric_panels(results_root, top):
            plt.close(figure)
            return None
        bottom = figure.add_gridspec(
            2,
            2,
            height_ratios=(1.0, 1.05),
            hspace=0.62,
            wspace=0.75,
            left=0.165,
            right=0.985,
        )
        tcga = figure.add_subplot(bottom[1, 0])
        contrasts = figure.add_subplot(bottom[1, 1])
        _tcga_subset_panel(results_root, tcga)
        _contrast_panel(results_root, contrasts)
        _panel_letter(top[0], "a", x=-0.38)
        _panel_letter(tcga, "b", x=-0.36)
        _panel_letter(contrasts, "c", x=-0.52)
        figure.legend(
            handles=[
                *_legend_handles(
                    [
                        ("MOGFormer", "mogformer"),
                        ("Linear SVM", "svm"),
                        ("Random forest", "forest"),
                        ("XGBoost", "xgboost"),
                        ("Elastic-net logistic", "elasticnet"),
                    ]
                ),
                plt.Line2D([], [], color=MUTED_INK, lw=0.7, ls=":", label="chance"),
            ],
            loc="lower center",
            ncol=6,
            bbox_to_anchor=(0.55, -0.01),
            handlelength=1.6,
            columnspacing=1.4,
        )
        figure.subplots_adjust(left=0.165, right=0.985, top=0.93, bottom=0.14)
        return _save(figure, directory, "paper_label_provenance")


def _benchmark_panel(results_root: Path, axes: plt.Axes) -> bool:
    """Draw every model on five-subtype TCGA PAM50; True if anything drawn."""
    models, _ = collect_benchmark(results_root, "5class")
    if not models:
        return False
    models.sort(key=lambda m: -m.interval()["mean"])
    for row, model in enumerate(models):
        y = len(models) - 1 - row
        summary = _interval(axes, y, model.scores, _style_key(model))
        axes.text(
            summary["nb_ci95_hi"] + 0.008,
            y,
            f"{summary['mean']:.3f}",
            va="center",
            fontsize=6,
            color=INK,
        )
    axes.set_yticks(range(len(models))[::-1])
    axes.set_yticklabels([SHORT_MODEL_NAMES.get(m.label, m.label) for m in models])
    axes.set_xlim(0.5, 1.0)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("macro-F1")
    axes.set_title("TCGA PAM50 (five subtypes)", color=INK)
    return True


def _ablation_panel(results_root: Path, axes: plt.Axes) -> bool:
    """Draw paired effects of single changes to MOGFormer; True if drawn."""
    runs = results_root / "transformer_variants"
    rows: list[tuple[str, np.ndarray]] = []
    for label, changed, reference in PAPER_ABLATIONS:
        after = read_scores(
            runs / f"transformer_{changed}_5class" / "metrics_per_fold.csv",
            "M_mogformer",
        )
        before = read_scores(
            runs / f"transformer_{reference}_5class" / "metrics_per_fold.csv",
            "M_mogformer",
        )
        if after is not None and before is not None:
            rows.append((label, after - before))
    if not rows:
        return False
    rng = np.random.default_rng(0)
    colour = PAPER_MODEL_STYLE["mogformer"][0]
    for row, (_, differences) in enumerate(rows):
        y = len(rows) - 1 - row
        axes.scatter(
            differences,
            y + rng.uniform(-0.16, 0.16, len(differences)),
            s=4,
            color=colour,
            alpha=0.35,
            edgecolors="none",
        )
        _interval(axes, y, differences, "mogformer", size=18)
    axes.axvline(0.0, color=MUTED_INK, lw=0.7)
    axes.set_yticks(range(len(rows))[::-1])
    axes.set_yticklabels([label for label, _ in rows])
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("paired difference in macro-F1")
    axes.set_title("Change to MOGFormer (TCGA)", color=INK)
    return True


def _stopping_panel(results_root: Path, axes: plt.Axes) -> bool:
    """Draw per-fold macro-F1 under both stopping rules; True if drawn."""
    original, final = STOPPING_RULES
    styles = {
        "her2_snp6": ("HER2 status", OKABE_ITO["vermilion"], "s"),
        "intclust": ("IntClust", OKABE_ITO["blue"], "o"),
    }
    values = []
    for target, (label, colour, marker) in styles.items():
        stem = results_root / "transformer_variants" / f"metabric_{target}_cnv_only"
        before = read_scores(
            Path(f"{stem}{original}") / "metrics_per_fold.csv", "M_mogformer"
        )
        after = read_scores(
            Path(f"{stem}{final}") / "metrics_per_fold.csv", "M_mogformer"
        )
        if before is None or after is None or len(before) != len(after):
            continue
        axes.scatter(
            before,
            after,
            s=10,
            color=colour,
            marker=marker,
            edgecolors="white",
            linewidths=0.4,
            label=label,
            zorder=3,
        )
        values.extend([before, after])
    if not values:
        return False
    pooled = np.concatenate(values)
    low, high = float(pooled.min()) - 0.02, float(pooled.max()) + 0.02
    axes.plot([low, high], [low, high], color=MUTED_INK, lw=0.7, ls=":")
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    axes.set_aspect("equal")
    axes.set_xlabel("macro-F1, original rule")
    axes.set_ylabel("macro-F1, final rule")
    axes.set_title("Stopping rule", color=INK)
    axes.legend(loc="lower right", handletextpad=0.1, borderaxespad=0.1)
    return True


def plot_benchmark_ablation_stopping_panels(
    results_root: Path, directory: Path
) -> Path | None:
    """Draw the model benchmark, the transformer ablations and training stability.

    Panel a: every model on five-subtype TCGA PAM50. Panel b: paired effects
    of single changes to MOGFormer. Panel c: per-fold macro-F1 of MOGFormer
    from copy number under the original and final stopping rules.

    Args:
        results_root: Directory holding ``baselines/`` and
            ``transformer_variants/``.
        directory: Where the figure is written.

    Returns:
        The written PNG, or None when no TCGA benchmark score exists.
    """
    apply_house_style()
    with plt.rc_context(cast(Any, PAPER_STYLE)):
        figure = plt.figure(figsize=(PAGE_WIDTH, 2.4))
        grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 0.72, 0.72), wspace=0.72)
        benchmark = figure.add_subplot(grid[0, 0])
        ablation = figure.add_subplot(grid[0, 1])
        stopping = figure.add_subplot(grid[0, 2])
        if not _benchmark_panel(results_root, benchmark):
            plt.close(figure)
            return None
        _ablation_panel(results_root, ablation)
        _stopping_panel(results_root, stopping)
        figure.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.18)
        for axes, letter in ((benchmark, "a"), (ablation, "b"), (stopping, "c")):
            left = axes.get_position().x0
            offset = 0.2 if letter == "a" else 0.12 if letter == "b" else 0.07
            figure.text(
                left - offset,
                0.95,
                letter,
                fontsize=9,
                fontweight="bold",
                color=INK,
                va="bottom",
            )
        return _save(figure, directory, "paper_model_comparison")


def build_paper_figures(
    results_root: Path | str, output_dir: Path | str
) -> dict[str, str | None]:
    """Draw every composite figure the manuscript includes.

    Args:
        results_root: Directory holding the benchmark outputs.
        output_dir: Where the figures are written.

    Returns:
        Mapping of figure name to the written PNG, or None where inputs were
        missing.
    """
    root, output = Path(results_root), Path(output_dir)
    written = {
        "paper_label_provenance": plot_label_provenance(root, output),
        "paper_model_comparison": plot_benchmark_ablation_stopping_panels(root, output),
    }
    return {name: None if path is None else str(path) for name, path in written.items()}


__all__ = [
    "ScoredModel",
    "build_paper_figures",
    "plot_benchmark_ablation_stopping_panels",
    "plot_label_provenance",
]
