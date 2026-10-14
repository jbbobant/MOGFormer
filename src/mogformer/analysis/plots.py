"""Figures for the downstream analyses.

Each plot corresponds to a claim made in the writeup, and several are designed
specifically to keep a weak result visible rather than flattering:

* the representation ladder draws the simple baselines beside the model, so a
  redundant embedding cannot hide behind a good-looking heatmap;
* the sign-concordance panel draws the trivial majority baseline alongside the
  permutation null, because a result can clear the null and still lose to a
  constant predictor;
* the confound screen colours bars by role, so a technical driver is obvious.

Shares :func:`~mogformer.evaluation.plots.save_figure` and the palette with the
evaluation figures, so the whole project's output looks like one system.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mogformer.analysis.clustering import ConsensusResult, order_by_linkage
from mogformer.analysis.probe_trans import SignConcordanceResult
from mogformer.data.omics import MODALITY_ORDER
from mogformer.evaluation.plots import (
    OKABE_ITO,
    SOFT_PALETTE,
    apply_house_style,
    light,
    save_figure,
    set_figure_title,
)

logger = logging.getLogger(__name__)

#: Colour of each cluster or stratum, in label order, in every figure.
CLUSTER_COLOURS: tuple[str, ...] = (
    SOFT_PALETTE["blue"],
    SOFT_PALETTE["coral"],
    SOFT_PALETTE["teal"],
    SOFT_PALETTE["mauve"],
    SOFT_PALETTE["amber"],
    SOFT_PALETTE["grey"],
)

#: Bar colour per covariate role in the confound screen.
ROLE_COLOURS: dict[str, str] = {
    "biology": SOFT_PALETTE["blue"],
    "clinical": SOFT_PALETTE["amber"],
    "technical": SOFT_PALETTE["coral"],
    "unspecified": SOFT_PALETTE["grey"],
}

#: Colour of each representation wherever representations are compared; the
#: encoder leads in coral.
REPRESENTATION_COLOURS: dict[str, str] = {
    "encoder": SOFT_PALETTE["coral"],
    "pca_inputs": SOFT_PALETTE["blue"],
    "mofa": SOFT_PALETTE["teal"],
    "pca_rna": SOFT_PALETTE["amber"],
    "pca_cnv": SOFT_PALETTE["sky"],
    "pca_methy": SOFT_PALETTE["mauve"],
    "burden": SOFT_PALETTE["grey"],
    "ploidy": SOFT_PALETTE["ink"],
}

#: Colour of each PAM50 subtype, the same in every figure.
SUBTYPE_COLOURS: dict[str, str] = {
    "LumA": SOFT_PALETTE["blue"],
    "LumB": SOFT_PALETTE["coral"],
    "Her2": SOFT_PALETTE["mauve"],
    "Basal": "#9E2A2B",
    "Normal": SOFT_PALETTE["teal"],
    "claudin-low": SOFT_PALETTE["amber"],
    "NC": SOFT_PALETTE["grey"],
    "not labelled": "#C8C8C8",
}

#: Edge colour of framed legends.
LEGEND_EDGE = "#D0D0D0"


def significance_stars(p_value: float) -> str:
    """Return the conventional marks for a (corrected) p-value.

    Args:
        p_value: The p-value.

    Returns:
        ``"ns"`` at 0.05 or above, else one star per threshold passed among
        0.05, 0.01, 0.001 and 0.0001.
    """
    if not np.isfinite(p_value) or p_value >= 0.05:
        return "ns"
    return "*" * int(sum(p_value < t for t in (0.05, 0.01, 0.001, 0.0001)))


def plot_consensus_heatmap(
    result: ConsensusResult,
    directory: str | Path,
    name: str = "consensus_heatmap",
    context: str = "",
) -> Path:
    """Draw a consensus matrix with patients ordered by linkage.

    Two clean blocks with near-zero off-diagonal mass is what a reproducible
    partition looks like; mid-range grey between the blocks is the ambiguity the
    accompanying statistic counts.

    Args:
        result: Consensus clustering result to draw.
        directory: Destination directory.
        name: File stem.
        context: Cohort, sample size or run the figure describes, drawn as a
            final title line so the figure reads on its own.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    order = order_by_linkage(result.consensus)

    figure, axes = plt.subplots(figsize=(5.6, 5.0))
    image = axes.imshow(
        result.consensus[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1
    )
    axes.set_xticks([])
    axes.set_yticks([])
    set_figure_title(
        axes,
        f"Consensus co-clustering at K={result.n_clusters}\n"
        f"PAC={result.pac:.3f}, n={len(result.labels)}",
        context,
    )
    axes.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="co-cluster rate")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_stability_by_k(
    table: pd.DataFrame,
    directory: str | Path,
    name: str = "stability_by_k",
    bar: Mapping[str, float] | None = None,
    highlight: str = "encoder",
) -> Path:
    """Draw partition stability against K for every representation and algorithm.

    One column of panels per population: PAC on top and, when present,
    bootstrap ARI beneath. The highlighted representation is drawn bold, the
    others thin; k-means solid, a Gaussian mixture dashed with open markers.
    The region that meets the bar is shaded.

    Args:
        table: One row per (population, representation, algorithm, K) with
            ``pac`` and optionally ``bootstrap_ari``; ``population``,
            ``representation`` and ``algorithm`` default to one value each.
        directory: Destination directory.
        name: File stem.
        bar: Thresholds ``pac`` and ``bootstrap_ari`` to shade, or None.
        highlight: Representation drawn bold.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    table = table.copy()
    for field_name, default in (
        ("population", ""),
        ("representation", highlight),
        ("algorithm", "kmeans"),
    ):
        if field_name not in table:
            table[field_name] = default
    populations = list(dict.fromkeys(table["population"]))
    names = list(dict.fromkeys(table["representation"]))
    metrics = ["pac"] + (["bootstrap_ari"] if "bootstrap_ari" in table else [])
    figure, axes = plt.subplots(
        len(metrics),
        len(populations),
        figsize=(4.0 * len(populations), 3.3 * len(metrics)),
        squeeze=False,
    )
    band = light(SOFT_PALETTE["teal"], 0.88)
    for column, population in enumerate(populations):
        subset = table[table["population"] == population]
        if bar is not None:
            axes[0, column].axhspan(0, bar["pac"], color=band, lw=0)
            if len(metrics) > 1:
                axes[1, column].axhspan(bar["bootstrap_ari"], 1.02, color=band, lw=0)
        for index, representation in enumerate(names):
            colour = REPRESENTATION_COLOURS.get(
                representation, CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]
            )
            bold = representation == highlight
            for algorithm, style in (("kmeans", "-"), ("gmm", "--")):
                rows = subset[
                    (subset["representation"] == representation)
                    & (subset["algorithm"] == algorithm)
                ].sort_values("k")
                if rows.empty:
                    continue
                kwargs = {
                    "color": colour,
                    "ls": style,
                    "lw": 2.6 if bold else 1.1,
                    "alpha": 1.0 if bold else 0.75,
                    "marker": "o",
                    "ms": 5.5 if bold else 3.5,
                    "mfc": "white" if algorithm == "gmm" else colour,
                    "mec": colour,
                    "zorder": 4 if bold else 2,
                }
                label = f"{representation} ({algorithm})" if column == 0 else None
                for row, metric in enumerate(metrics):
                    axes[row, column].plot(
                        rows["k"],
                        rows[metric],
                        label=label if row == 0 else None,
                        **kwargs,
                    )
        axes[0, column].set_title(population or "Stability by K")
        axes[0, column].set_ylim(-0.02, max(0.9, float(subset["pac"].max()) + 0.05))
        if len(metrics) > 1:
            axes[1, column].set_ylim(0, 1.02)
        axes[-1, column].set_xlabel("number of clusters K")
        for row in range(len(metrics)):
            axes[row, column].set_xticks(sorted(subset["k"].unique()))
    axes[0, 0].set_ylabel("PAC  (lower = more stable)")
    if len(metrics) > 1:
        axes[1, 0].set_ylabel("bootstrap ARI")
    if bar is not None:
        axes[0, -1].text(
            0.98,
            0.04,
            "meets bar",
            transform=axes[0, -1].transAxes,
            ha="right",
            fontsize=8,
            color=SOFT_PALETTE["teal"],
        )
    if len(names) > 1 or table["algorithm"].nunique() > 1:
        figure.legend(
            loc="lower center",
            ncol=min(8, 2 * len(names)),
            fontsize=7.5,
            bbox_to_anchor=(0.5, -0.07),
            frameon=True,
            edgecolor=LEGEND_EDGE,
        )
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_representation_ladder(
    rungs: Sequence[object],
    directory: str | Path,
    name: str = "representation_ladder",
    context: str = "",
) -> Path:
    """Draw each representation's stability and its agreement with the reference.

    The figure that makes redundancy impossible to miss: if a simple
    representation reaches the same agreement as the full model, the two bars
    sit level and the model's contribution to the partition is visibly nil.

    Args:
        rungs: Representation rungs, each carrying ``name``, ``pac`` and
            ``ari_vs_reference``.
        directory: Destination directory.
        name: File stem.
        context: Cohort, sample size or run the figure describes, drawn as a
            final title line so the figure reads on its own.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    labels = [rung.name for rung in rungs]  # type: ignore[attr-defined]
    pacs = [rung.pac for rung in rungs]  # type: ignore[attr-defined]
    agreements = [rung.ari_vs_reference for rung in rungs]  # type: ignore[attr-defined]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    positions = np.arange(len(labels))

    left.bar(positions, pacs, color=OKABE_ITO["blue"])
    left.set_xticks(positions)
    left.set_xticklabels(labels, rotation=25, ha="right")
    left.set_ylabel("PAC (lower is more stable)")
    left.set_title("Partition stability (PAC) by representation")

    right.bar(positions, agreements, color=OKABE_ITO["green"])
    right.axhline(1.0, color=OKABE_ITO["vermilion"], ls="--", lw=1.2)
    right.text(
        len(labels) - 0.5,
        1.01,
        "identical partition",
        color=OKABE_ITO["vermilion"],
        fontsize=8,
        ha="right",
    )
    right.set_xticks(positions)
    right.set_xticklabels(labels, rotation=25, ha="right")
    right.set_ylabel("ARI vs the frozen partition")
    right.set_ylim(-0.1, 1.15)
    right.set_title("Agreement (ARI) with the reference partition")
    if context:
        figure.suptitle(context, fontsize=10)

    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_confound_screen(
    screens: pd.DataFrame | Mapping[str, pd.DataFrame],
    directory: str | Path,
    name: str = "confound_screen",
    label_col: str = "covariate",
    effect_col: str = "effect_size",
    xlabel: str = "effect size (|rho| or eta-squared)",
) -> Path:
    """Draw each known axis's association with a partition or score, by role.

    Bars are coloured by role (biology, clinical, technical), drawn as a light
    fill under a saturated outline, with the Holm-corrected significance
    marks at their ends; a technical driver is obvious at a glance. One panel
    per population when given a mapping.

    Args:
        screens: Output of
            :func:`~mogformer.analysis.clustering.screen_covariates` or
            :func:`~mogformer.analysis.clustering.partition_audit`, or a mapping
            of panel title to one.
        directory: Destination directory.
        name: File stem.
        label_col: Column naming each axis.
        effect_col: Column of the effect size.
        xlabel: Label of the effect-size axis.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    panels = screens if isinstance(screens, Mapping) else {"": screens}
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.1 * len(panels), 0.36 * max(len(s) for s in panels.values()) + 1.4),
        squeeze=False,
    )
    for column, (panel, screen) in enumerate(panels.items()):
        ordered = screen.sort_values(effect_col).reset_index(drop=True)
        axis = axes[0, column]
        effects = ordered[effect_col].clip(lower=0).fillna(0)
        colours = [
            ROLE_COLOURS.get(role, ROLE_COLOURS["unspecified"])
            for role in ordered["role"]
        ]
        axis.barh(
            ordered[label_col],
            effects,
            color=[light(c, 0.6) for c in colours],
            edgecolor=colours,
            linewidth=1.3,
            height=0.68,
        )
        top = max(float(effects.max()), 0.05)
        for y, (effect, p_value) in enumerate(
            zip(effects, ordered["p_holm"], strict=True)
        ):
            axis.text(
                effect + top * 0.03,
                y,
                significance_stars(p_value),
                va="center",
                fontsize=8,
                color=SOFT_PALETTE["ink"],
            )
        axis.set_xlim(0, top * 1.25)
        axis.set_xlabel(xlabel)
        axis.set_title(panel)
        axis.grid(axis="y", visible=False)
    roles = [
        r for r in ROLE_COLOURS if any(r in set(s["role"]) for s in panels.values())
    ]
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=light(ROLE_COLOURS[r], 0.6),
            edgecolor=ROLE_COLOURS[r],
            lw=1.3,
        )
        for r in roles
    ]
    figure.legend(
        handles,
        roles,
        loc="lower center",
        ncol=len(roles),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        edgecolor=LEGEND_EDGE,
    )
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_response_curves(
    curves: pd.DataFrame,
    genes: Sequence[str],
    directory: str | Path,
    name: str = "response_curves",
    context: str = "",
) -> Path:
    """Draw predicted expression against injected value for selected genes.

    A downward curve is the model asserting that methylation silences that gene.

    Args:
        curves: Per-grid-point curves from
            :func:`~mogformer.analysis.probe_cis.run_cis_map`.
        genes: Genes to draw.
        directory: Destination directory.
        name: File stem.
        context: Cohort, sample size or run the figure describes, drawn as a
            final title line so the figure reads on its own.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    colours = list(OKABE_ITO.values())

    figure, axes = plt.subplots(figsize=(6.6, 4.4))
    for index, gene in enumerate(genes):
        subset = curves[curves["gene"] == gene].sort_values("injected_value")
        if subset.empty:
            continue
        axes.errorbar(
            subset["injected_value"],
            subset["mean_prediction"],
            yerr=subset["sem_prediction"],
            marker="o",
            ms=4,
            lw=1.8,
            capsize=2,
            label=gene,
            color=colours[index % len(colours)],
        )

    axes.axhline(0, color=OKABE_ITO["grey"], lw=0.8)
    axes.axvline(0, color=OKABE_ITO["grey"], lw=0.8, ls=":")
    axes.set_xlabel("injected value (standard deviations)")
    axes.set_ylabel("predicted expression (standardised)")
    set_figure_title(
        axes,
        "Predicted expression under an injected input (mean ± s.e.m.)",
        context,
    )
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_sign_concordance(
    result: SignConcordanceResult,
    directory: str | Path,
    name: str = "sign_concordance",
    context: str = "",
) -> Path:
    """Draw observed sign concordance against both bars it must clear.

    Drawing the permutation null and the trivial majority baseline together is
    the whole point. Against a coin flip the observed rate can look decisive
    while sitting below the constant predictor, and this figure makes that
    visible rather than leaving it to a p-value.

    Args:
        result: Outcome of
            :func:`~mogformer.analysis.probe_trans.sign_permutation_test`.
        directory: Destination directory.
        name: File stem.
        context: Cohort, sample size or run the figure describes, drawn as a
            final title line so the figure reads on its own.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    figure, axes = plt.subplots(figsize=(7.0, 3.6))

    passed = result.is_informative
    axes.barh(
        [0],
        [result.rate],
        color=OKABE_ITO["green"] if passed else OKABE_ITO["vermilion"],
        height=0.5,
        label="observed",
    )
    axes.axvline(
        result.null_mean_rate,
        color=OKABE_ITO["grey"],
        lw=2,
        label=f"permutation null ({result.null_mean_rate:.3f})",
    )
    axes.axvspan(
        result.null_mean_rate - 2 * result.null_sd_rate,
        result.null_mean_rate + 2 * result.null_sd_rate,
        color=OKABE_ITO["grey"],
        alpha=0.18,
    )
    axes.axvline(
        result.trivial_majority_baseline,
        color=OKABE_ITO["black"],
        ls="--",
        lw=2,
        label=f"trivial majority ({result.trivial_majority_baseline:.3f})",
    )
    axes.axvline(
        0.5, color=OKABE_ITO["sky"], ls=":", lw=1.4, label="coin flip (wrong null)"
    )

    axes.set_yticks([])
    axes.set_xlim(0.4, 1.0)
    axes.set_xlabel("sign concordance")
    verdict = "informative" if passed else "NOT informative"
    set_figure_title(
        axes,
        f"Trans sign concordance with curated edge signs — {verdict}\n"
        f"{result.concordant}/{result.n_edges} edges, "
        f"permutation p={result.perm_p:.3f}",
        context,
    )
    axes.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_trans_versus_coexpression(
    edges: pd.DataFrame,
    directory: str | Path,
    name: str = "trans_vs_coexpression",
    context: str = "",
) -> Path:
    """Draw learned trans slope against observed co-expression.

    Args:
        edges: Edge rows carrying ``slope``, ``observed_coexpression`` and
            ``edge_sign``.
        directory: Destination directory.
        name: File stem.
        context: Cohort, sample size or run the figure describes, drawn as a
            final title line so the figure reads on its own.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    figure, axes = plt.subplots(figsize=(6.0, 5.4))

    for sign, colour, label in (
        (1, OKABE_ITO["green"], "activation"),
        (-1, OKABE_ITO["vermilion"], "repression"),
    ):
        subset = edges[edges["edge_sign"] == sign]
        axes.scatter(
            subset["observed_coexpression"],
            subset["slope"],
            s=22,
            alpha=0.6,
            color=colour,
            label=label,
            edgecolors="none",
        )

    axes.axhline(0, color=OKABE_ITO["grey"], lw=0.8, ls=":")
    axes.axvline(0, color=OKABE_ITO["grey"], lw=0.8, ls=":")
    axes.set_xlabel("observed TF-target co-expression")
    axes.set_ylabel("learned trans slope")
    set_figure_title(
        axes,
        f"Learned trans slope against observed co-expression, n={len(edges)} edges",
        context,
    )
    axes.legend(fontsize=9, title="curated edge sign", title_fontsize=9)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_kaplan_meier(
    frame: pd.DataFrame,
    endpoints: Mapping[str, tuple[str, str]],
    group_col: str,
    directory: str | Path,
    name: str = "kaplan_meier",
    title: str = "",
) -> Path:
    """Draw survival curves per group, one panel per endpoint, risk table beneath.

    Each group gets its colour from :data:`CLUSTER_COLOURS` and a light 95% band;
    the log-rank p-value sits in the panel's corner. The numbers at risk under
    each panel are not decoration: a curve's right-hand tail can rest on a
    handful of patients, and without them a reader cannot tell where the
    estimate stops meaning anything.

    Args:
        frame: One row per patient.
        endpoints: Panel title to the (duration, event) columns it draws.
        group_col: Column of the group (stratum) per patient.
        directory: Destination directory.
        name: File stem.
        title: Figure title: the cohort and population.

    Returns:
        The path of the written PNG.

    Raises:
        ImportError: If lifelines is not installed.
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "survival figures need lifelines; install the 'analysis' extra"
        ) from error
    from mogformer.analysis.survival import logrank_p_value

    apply_house_style()
    groups = sorted(frame[group_col].dropna().unique())
    figure, axes = plt.subplots(
        2,
        len(endpoints),
        figsize=(5.0 * len(endpoints), 4.9),
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": [4.2, 0.22 * len(groups) + 0.3]},
    )
    for column, (panel, (duration_col, event_col)) in enumerate(endpoints.items()):
        curve, table = axes[0, column], axes[1, column]
        complete = frame[[duration_col, event_col, group_col]].dropna()
        subsets = []
        for index, group in enumerate(groups):
            subset = complete[complete[group_col] == group]
            colour = CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]
            fitter = KaplanMeierFitter().fit(subset[duration_col], subset[event_col])
            band = fitter.confidence_interval_survival_function_
            curve.fill_between(
                band.index,
                band.iloc[:, 0],
                band.iloc[:, 1],
                step="post",
                color=light(colour, 0.75),
                alpha=0.6,
                lw=0,
            )
            curve.step(
                fitter.survival_function_.index,
                fitter.survival_function_.iloc[:, 0],
                where="post",
                color=colour,
                lw=2.0,
                label=f"{group_col} {group}  (n={len(subset)})",
            )
            subsets.append((subset, colour))
        p_value = logrank_p_value(complete, duration_col, event_col, group_col)
        curve.text(
            0.97,
            0.95,
            f"log-rank p = {p_value:.1e}",
            transform=curve.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=SOFT_PALETTE["ink"],
        )
        curve.set_ylim(0, 1.03)
        curve.set_title(panel)
        curve.legend(loc="lower left", frameon=True, edgecolor=LEGEND_EDGE)
        curve.set_xlabel("months since diagnosis")
        # The risk table sits under the curve's own ticks, column for column.
        limit = complete[duration_col].max()
        horizons = [t for t in curve.get_xticks() if 0 <= t <= limit]
        for index, (subset, colour) in enumerate(subsets):
            for t in horizons:
                table.text(
                    t,
                    len(groups) - 1 - index,
                    str(int((subset[duration_col] >= t).sum())),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=colour,
                )
        table.set_xlim(curve.get_xlim())
        table.set_xticks([])
        table.set_ylim(-0.6, len(groups) - 0.4)
        table.set_yticks(range(len(groups))[::-1], [str(g) for g in groups])
        table.grid(False)
        table.tick_params(length=0)
        for spine in table.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("survival probability")
    axes[1, 0].set_ylabel("at risk", rotation=0, ha="right", va="center")
    if title:
        figure.suptitle(title, fontsize=11, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_hazard_forest(
    ladder: pd.DataFrame,
    directory: str | Path,
    name: str = "hazard_forest",
    title: str = "",
    primary_rung: str | None = None,
) -> Path:
    """Draw an adjustment ladder: one shaded row per rung, one marker per contrast.

    Every rung is drawn, in order, so a reader sees the whole estimand ladder
    rather than whichever rung happened to be most favourable. Each estimate's
    value and interval are printed at the right in its contrast's colour.

    Args:
        ladder: One row per (rung, contrast): ``rung`` (or ``model``),
            ``hazard_ratio``, ``ci_low``, ``ci_high``, and optionally
            ``stratum`` and ``reference`` naming the contrast.
        directory: Destination directory.
        name: File stem.
        title: Figure title: the endpoint and population.
        primary_rung: Rung drawn with diamonds, the pre-registered primary
            estimate; defaults to the last rung.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    rows = ladder.rename(columns={"model": "rung"}).copy()
    if "stratum" not in rows:
        rows["stratum"], rows["reference"] = 1, 0
    rungs = list(dict.fromkeys(rows["rung"]))
    primary_rung = primary_rung or rungs[-1]
    contrasts = sorted(rows["stratum"].unique())
    offsets = np.linspace(0.16, -0.16, len(contrasts)) if len(contrasts) > 1 else [0]
    figure, axes = plt.subplots(figsize=(7.4, 0.55 * len(rungs) + 1.6))
    upper = float(max(rows["ci_high"].max(), 2.0)) * 1.08
    lower = float(min(rows["ci_low"].min(), 0.9)) * 0.92
    for i, rung in enumerate(rungs):
        y = len(rungs) - 1 - i
        if i % 2 == 0:
            axes.axhspan(y - 0.5, y + 0.5, color="#F6F6F6", lw=0, zorder=0)
        for offset, contrast in zip(offsets, contrasts, strict=True):
            cell = rows[(rows["rung"] == rung) & (rows["stratum"] == contrast)].iloc[0]
            colour = CLUSTER_COLOURS[int(contrast) % len(CLUSTER_COLOURS)]
            axes.hlines(
                y + offset, cell["ci_low"], cell["ci_high"], color=colour, lw=1.8
            )
            axes.scatter(
                cell["hazard_ratio"],
                y + offset,
                marker="D" if rung == primary_rung else "o",
                s=46 if rung == primary_rung else 30,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=(
                    f"stratum {contrast} vs {int(cell['reference'])}"
                    if i == 0
                    else None
                ),
            )
            axes.text(
                upper * 1.04,
                y + offset,
                f"{cell['hazard_ratio']:.2f} "
                f"({cell['ci_low']:.2f}–{cell['ci_high']:.2f})",
                va="center",
                fontsize=7.5,
                color=colour,
            )
    axes.axvline(1.0, color=SOFT_PALETTE["grey"], lw=1.0, ls="--")
    axes.set_xscale("log")
    ticks = [
        t for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0) if lower <= t <= upper
    ]
    axes.set_xticks(ticks, [f"{t:g}" for t in ticks])
    axes.minorticks_off()
    axes.set_xlim(lower, upper)
    axes.set_yticks(range(len(rungs))[::-1], rungs)
    axes.set_ylim(-0.6, len(rungs) - 0.4)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("hazard ratio (95% CI, log scale)")
    axes.set_title(title)
    if len(contrasts) > 1:
        axes.legend(loc="upper left", frameon=True, edgecolor=LEGEND_EDGE)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_strata_composition(
    compositions: Mapping[str, pd.DataFrame],
    directory: str | Path,
    name: str = "strata_composition",
    legend_title: str = "PAM50",
) -> Path:
    """Draw what each stratum is made of, one panel per population.

    Args:
        compositions: Population to a cross-tabulation, strata as rows and
            categories as columns (``BRCA_`` prefixes are dropped).
        directory: Destination directory.
        name: File stem.
        legend_title: What the categories are.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    renamed = {
        population: table.rename(columns=lambda c: str(c).removeprefix("BRCA_"))
        for population, table in compositions.items()
    }
    present = {c for table in renamed.values() for c in table.columns}
    categories = [c for c in SUBTYPE_COLOURS if c in present] + sorted(
        present - set(SUBTYPE_COLOURS)
    )
    colours = {
        c: SUBTYPE_COLOURS.get(c, CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)])
        for i, c in enumerate(categories)
    }
    figure, axes = plt.subplots(
        1, len(renamed), figsize=(3.4 * len(renamed), 3.6), squeeze=False
    )
    for column, (population, table) in enumerate(renamed.items()):
        axis = axes[0, column]
        shares = table.div(table.sum(axis=1), axis=0)
        bottom = np.zeros(len(shares))
        for category in categories:
            if category not in shares.columns:
                continue
            values = shares[category].to_numpy()
            axis.bar(
                range(len(shares)),
                values,
                bottom=bottom,
                width=0.62,
                color=light(colours[category], 0.35),
                edgecolor="white",
                linewidth=1.0,
            )
            bottom += values
        for x, total in enumerate(table.sum(axis=1)):
            axis.text(x, 1.02, f"n={total}", ha="center", fontsize=8, color="#4D4D4D")
        axis.set_xticks(range(len(shares)), [f"stratum {i}" for i in shares.index])
        axis.set_ylim(0, 1.1)
        axis.set_title(population)
        axis.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("share of patients")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=light(colours[c], 0.35)) for c in categories
    ]
    figure.legend(
        handles,
        categories,
        loc="center right",
        title=legend_title,
        frameon=True,
        edgecolor=LEGEND_EDGE,
    )
    figure.tight_layout(rect=(0, 0, 0.9, 1))
    return save_figure(figure, directory, name)


def plot_partition_crosstab(
    crosstab: pd.DataFrame,
    directory: str | Path,
    name: str = "partition_crosstab",
    title: str = "",
) -> Path:
    """Draw how two partitions of the same patients cross, as a soft heatmap.

    Args:
        crosstab: Counts, one partition as rows and the other as columns;
            the index and column names label the axes.
        directory: Destination directory.
        name: File stem.
        title: What is compared, with its agreement score.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    colours = matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_blue", [(1.0, 1.0, 1.0), light(SOFT_PALETTE["blue"], 0.15)]
    )
    figure, axes = plt.subplots(figsize=(4.0, 3.4))
    values = crosstab.to_numpy()
    axes.imshow(values, cmap=colours, vmin=0)
    for (i, j), value in np.ndenumerate(values):
        axes.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=9,
            color="white" if value > values.max() * 0.6 else SOFT_PALETTE["ink"],
        )
    axes.set_xticks(range(crosstab.shape[1]), crosstab.columns)
    axes.set_yticks(range(crosstab.shape[0]), crosstab.index)
    axes.set_xlabel(str(crosstab.columns.name or ""))
    axes.set_ylabel(str(crosstab.index.name or ""))
    axes.set_title(title)
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return save_figure(figure, directory, name)


#: Colour of each omics in every figure.
OMICS_COLOURS: dict[str, str] = {
    "rna": SOFT_PALETTE["blue"],
    "cnv": SOFT_PALETTE["amber"],
    "methy": SOFT_PALETTE["mauve"],
}

#: Input value from low to high in attribution figures.
VALUE_COLOURMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "soft_value", [SOFT_PALETTE["teal"], SOFT_PALETTE["mauve"], SOFT_PALETTE["coral"]]
)


def plot_attribution_beeswarm(
    attributions: np.ndarray,
    inputs: np.ndarray,
    genes: Sequence[str],
    directory: str | Path,
    name: str = "attribution",
    title: str = "",
    top: int = 18,
    negative_label: str = "lowers the score",
    positive_label: str = "raises the score",
) -> Path:
    """Draw per-patient attributions of the leading features as a beeswarm.

    One row per (gene, omics), ranked by mean absolute attribution; one point
    per patient at its attribution, coloured by the patient's input value
    (teal low, coral high). A row whose coral points sit right of zero is a
    feature whose high values raise the score.

    Args:
        attributions: Patients x genes x omics.
        inputs: The same patients' standardised inputs.
        genes: Gene names.
        directory: Destination directory.
        name: File stem.
        title: What score is attributed, in which population.
        top: Features drawn.
        negative_label: What a negative attribution means.
        positive_label: What a positive attribution means.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    n_patients, _, n_omics = attributions.shape
    flat = attributions.reshape(n_patients, -1)
    values = inputs.reshape(n_patients, -1)
    order = np.argsort(-np.abs(flat).mean(axis=0))[:top]
    labels = [f"{genes[i // n_omics]}  ({MODALITY_ORDER[i % n_omics]})" for i in order]
    rng = np.random.default_rng(0)
    figure, axes = plt.subplots(figsize=(7.6, 0.34 * top + 1.4))
    for row, feature in enumerate(order):
        y = top - 1 - row
        value = values[:, feature]
        low, high = np.percentile(value, [5, 95])
        colour = np.clip((value - low) / (high - low + 1e-9), 0, 1)
        axes.scatter(
            flat[:, feature],
            y + rng.uniform(-0.28, 0.28, n_patients),
            c=colour,
            cmap=VALUE_COLOURMAP,
            s=7,
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )
    axes.axvline(0, color=SOFT_PALETTE["grey"], lw=1.0, zorder=2)
    axes.set_yticks(range(top)[::-1], labels, fontsize=8)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel(f"←  {negative_label}     |     {positive_label}  →\nattribution")
    axes.set_title(title)
    scalar = plt.cm.ScalarMappable(cmap=VALUE_COLOURMAP)
    bar = figure.colorbar(scalar, ax=axes, fraction=0.03, pad=0.02, ticks=[0, 1])
    bar.ax.set_yticklabels(["low", "high"])
    bar.set_label("input value", fontsize=8)
    bar.outline.set_visible(False)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_omics_reliance(
    gates: pd.DataFrame,
    attribution_shares: Mapping[str, Mapping[str, float]],
    directory: str | Path,
    name: str = "omics_reliance",
    title: str = "",
) -> Path:
    """Draw the fusion gate's weight per omics beside each omics' attribution share.

    Args:
        gates: Mean gate weight per omics (columns), one row per stratum (and
            ``all``).
        attribution_shares: Read-out name to each omics' share of its absolute
            attribution.
        directory: Destination directory.
        name: File stem.
        title: Population and encoder.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    omics = [o for o in OMICS_COLOURS if o in gates.columns]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    width = 0.8 / len(omics)
    for k, o in enumerate(omics):
        positions = np.arange(len(gates)) + (k - (len(omics) - 1) / 2) * width
        left.bar(
            positions,
            gates[o],
            width=width * 0.92,
            color=light(OMICS_COLOURS[o], 0.6),
            edgecolor=OMICS_COLOURS[o],
            linewidth=1.3,
            label=o,
        )
    left.axhline(1 / len(omics), color=SOFT_PALETTE["grey"], ls="--", lw=1.0)
    left.set_xticks(
        range(len(gates)),
        [f"stratum {i}" if str(i) != "all" else "all" for i in gates.index],
    )
    left.set_ylabel("mean fusion gate weight")
    left.set_title("Fusion gate by stratum")
    left.grid(axis="x", visible=False)
    left.legend(frameon=True, edgecolor=LEGEND_EDGE)
    readouts = list(attribution_shares)
    bottom = np.zeros(len(readouts))
    for o in omics:
        shares = np.array([attribution_shares[r][o] for r in readouts])
        right.bar(
            range(len(readouts)),
            shares,
            bottom=bottom,
            width=0.55,
            color=light(OMICS_COLOURS[o], 0.35),
            edgecolor="white",
        )
        for x, (b, s) in enumerate(zip(bottom, shares, strict=True)):
            right.text(x, b + s / 2, f"{s:.0%}", ha="center", va="center", fontsize=8)
        bottom += shares
    right.set_xticks(range(len(readouts)), [f"{r} score" for r in readouts])
    right.set_ylabel("share of |attribution|")
    right.set_ylim(0, 1)
    right.set_title("Attribution by omics")
    right.grid(axis="x", visible=False)
    figure.suptitle(title, fontsize=11, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_block_prognosis(
    programs: pd.DataFrame,
    directory: str | Path,
    name: str = "program_prognosis",
    title: str = "",
) -> Path:
    """Draw each gene program block's out-of-fold C-index, labelled by its genes.

    Args:
        programs: One row per program: ``program``, ``relapse_c_index``,
            ``relapse_c_index_sd`` and ``top_attended_genes``.
        directory: Destination directory.
        name: File stem.
        title: Population and endpoint.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    ordered = programs.sort_values("relapse_c_index").reset_index(drop=True)
    best = ordered["relapse_c_index"].max()
    colours = [
        SOFT_PALETTE["coral"] if c == best else SOFT_PALETTE["blue"]
        for c in ordered["relapse_c_index"]
    ]
    labels = [
        f"P{int(program)}: " + ", ".join(str(genes).split(", ")[:3])
        for program, genes in zip(
            ordered["program"], ordered["top_attended_genes"], strict=True
        )
    ]
    figure, axes = plt.subplots(figsize=(7.2, 0.34 * len(ordered) + 1.4))
    axes.barh(
        range(len(ordered)),
        ordered["relapse_c_index"] - 0.5,
        left=0.5,
        xerr=ordered["relapse_c_index_sd"],
        color=[light(c, 0.6) for c in colours],
        edgecolor=colours,
        linewidth=1.3,
        height=0.68,
        error_kw={"ecolor": SOFT_PALETTE["grey"], "lw": 1.0},
    )
    axes.axvline(0.5, color=SOFT_PALETTE["grey"], ls="--", lw=1.0)
    axes.set_yticks(range(len(ordered)), labels, fontsize=8)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("out-of-fold C-index for relapse (block alone; 0.5 = chance)")
    axes.set_title(title)
    figure.tight_layout()
    return save_figure(figure, directory, name)
