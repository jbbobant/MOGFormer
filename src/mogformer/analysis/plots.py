"""Figures for the downstream analysis phases.

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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mogformer.analysis.clustering import ConsensusResult, order_by_linkage
from mogformer.analysis.probe_trans import SignConcordanceResult
from mogformer.evaluation.plots import OKABE_ITO, apply_house_style, save_figure

logger = logging.getLogger(__name__)

#: Cluster colours, matching the project's existing figures.
CLUSTER_COLOURS: tuple[str, ...] = (OKABE_ITO["blue"], OKABE_ITO["vermilion"])

#: Bar colour per covariate role in the confound screen.
ROLE_COLOURS: dict[str, str] = {
    "biology": OKABE_ITO["green"],
    "clinical": OKABE_ITO["blue"],
    "technical": OKABE_ITO["vermilion"],
    "unspecified": OKABE_ITO["grey"],
}


def plot_consensus_heatmap(
    result: ConsensusResult, directory: str | Path, name: str = "consensus_heatmap"
) -> Path:
    """Draw a consensus matrix with patients ordered by linkage.

    Two clean blocks with near-zero off-diagonal mass is what a reproducible
    partition looks like; mid-range grey between the blocks is the ambiguity the
    accompanying statistic counts.

    Args:
        result: Consensus clustering result to draw.
        directory: Destination directory.
        name: File stem.

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
    axes.set_title(
        f"Consensus at K={result.n_clusters}\n"
        f"PAC={result.pac:.3f}, n={len(result.labels)}"
    )
    axes.grid(False)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="co-cluster rate")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_stability_by_k(
    results: Mapping[int, ConsensusResult],
    directory: str | Path,
    name: str = "stability_by_k",
) -> Path:
    """Draw stability against the number of clusters.

    Args:
        results: Mapping of cluster count to consensus result.
        directory: Destination directory.
        name: File stem.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    counts = sorted(results)
    pacs = [results[k].pac for k in counts]

    figure, axes = plt.subplots(figsize=(6.0, 3.8))
    axes.plot(counts, pacs, "o-", color=OKABE_ITO["blue"], lw=2)
    best = counts[int(np.argmin(pacs))]
    axes.scatter(
        [best],
        [min(pacs)],
        s=140,
        facecolors="none",
        edgecolors=OKABE_ITO["vermilion"],
        lw=2,
        zorder=4,
    )
    axes.annotate(
        f"most stable: K={best}",
        (best, min(pacs)),
        textcoords="offset points",
        xytext=(10, 10),
        color=OKABE_ITO["vermilion"],
        fontsize=9,
    )
    axes.set_xticks(counts)
    axes.set_xlabel("number of clusters")
    axes.set_ylabel("PAC (lower is more stable)")
    axes.set_title("Partition stability under resampling")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_representation_ladder(
    rungs: Sequence[object], directory: str | Path, name: str = "representation_ladder"
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
    left.set_title("Stability by representation")

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
    right.set_title("Does the partition need the model?")

    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_confound_screen(
    screen: pd.DataFrame, directory: str | Path, name: str = "confound_screen"
) -> Path:
    """Draw covariate effect sizes, coloured by role.

    Args:
        screen: Output of
            :func:`~mogformer.analysis.clustering.screen_covariates`.
        directory: Destination directory.
        name: File stem.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    rows = screen.sort_values("effect_size")
    colours = [
        ROLE_COLOURS.get(role, ROLE_COLOURS["unspecified"]) for role in rows["role"]
    ]

    figure, axes = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 2.0))
    positions = np.arange(len(rows))
    axes.barh(positions, rows["effect_size"], color=colours)

    for position, (_, row) in zip(positions, rows.iterrows(), strict=True):
        marker = " *" if row["p_holm"] < 0.05 else ""
        axes.text(
            row["effect_size"] + 0.01,
            float(position),
            f"{row['effect_size']:.2f}{marker}",
            va="center",
            fontsize=8,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(rows["covariate"])
    axes.set_xlabel("effect size (|rho| or eta-squared)")
    axes.set_title("What the partition tracks  (* Holm-significant)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour)
        for role, colour in ROLE_COLOURS.items()
        if role in set(rows["role"])
    ]
    axes.legend(
        handles,
        [role for role in ROLE_COLOURS if role in set(rows["role"])],
        fontsize=8,
        loc="lower right",
    )
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_response_curves(
    curves: pd.DataFrame,
    genes: Sequence[str],
    directory: str | Path,
    name: str = "response_curves",
) -> Path:
    """Draw predicted expression against injected value for selected genes.

    A downward curve is the model asserting that methylation silences that gene.

    Args:
        curves: Per-grid-point curves from
            :func:`~mogformer.analysis.probe_cis.run_cis_map`.
        genes: Genes to draw.
        directory: Destination directory.
        name: File stem.

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
    axes.set_title("Interventional response — downward means silencing")
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_sign_concordance(
    result: SignConcordanceResult,
    directory: str | Path,
    name: str = "sign_concordance",
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
    axes.set_title(
        f"Trans sign concordance — {verdict}\n"
        f"{result.concordant}/{result.n_edges} edges, permutation p={result.perm_p:.3f}"
    )
    axes.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_trans_versus_coexpression(
    edges: pd.DataFrame,
    directory: str | Path,
    name: str = "trans_vs_coexpression",
) -> Path:
    """Draw learned trans slope against observed co-expression.

    Args:
        edges: Edge rows carrying ``slope``, ``observed_coexpression`` and
            ``edge_sign``.
        directory: Destination directory.
        name: File stem.

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
    axes.set_title("Graph routing, or an echo of co-expression?")
    axes.legend(fontsize=9)
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_kaplan_meier(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    directory: str | Path,
    name: str = "kaplan_meier",
    p_value: float | None = None,
) -> Path:
    """Draw survival curves per group with a risk table beneath.

    The risk table is not decoration: at 35 events a curve's right-hand tail
    rests on a handful of patients, and without the counts a reader cannot tell
    where the estimate stops meaning anything.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator.
        group_col: Binary group column.
        directory: Destination directory.
        name: File stem.
        p_value: Optional log-rank p-value to annotate.

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

    apply_house_style()
    complete = frame[[duration_col, event_col, group_col]].dropna()
    groups = sorted(complete[group_col].unique())

    figure, axes = plt.subplots(figsize=(6.6, 4.8))
    horizons = np.linspace(0, complete[duration_col].max(), 6)
    risk_rows = []

    for index, group in enumerate(groups):
        subset = complete[complete[group_col] == group]
        fitter = KaplanMeierFitter(label=f"{group} (n={len(subset)})")
        fitter.fit(subset[duration_col], subset[event_col])
        fitter.plot_survival_function(
            ax=axes, color=CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)], ci_show=True
        )
        risk_rows.append([int((subset[duration_col] >= t).sum()) for t in horizons])

    if p_value is not None:
        axes.text(
            0.98,
            0.04,
            f"log-rank p = {p_value:.3f}",
            transform=axes.transAxes,
            ha="right",
            fontsize=9,
        )

    axes.set_xlabel("months")
    axes.set_ylabel("survival probability")
    axes.set_ylim(0, 1.02)
    axes.set_title("Survival by group")

    table = "\n".join(
        f"{group}: " + "  ".join(f"{n:>4d}" for n in counts)
        for group, counts in zip(groups, risk_rows, strict=True)
    )
    figure.text(0.01, -0.06, f"at risk\n{table}", fontsize=7, family="monospace")
    figure.tight_layout()
    return save_figure(figure, directory, name)


def plot_hazard_forest(
    ladder: pd.DataFrame, directory: str | Path, name: str = "hazard_forest"
) -> Path:
    """Draw the adjustment ladder as a forest plot.

    Every rung is drawn, in order, so a reader sees the whole estimand ladder
    rather than whichever rung happened to be most favourable.

    Args:
        ladder: Output of
            :func:`~mogformer.analysis.survival.adjustment_ladder`.
        directory: Destination directory.
        name: File stem.

    Returns:
        The path of the written PNG.
    """
    apply_house_style()
    figure, axes = plt.subplots(figsize=(7.4, 0.55 * len(ladder) + 2.0))
    positions = np.arange(len(ladder))[::-1]

    axes.hlines(
        positions,
        ladder["ci_low"],
        ladder["ci_high"],
        color=OKABE_ITO["blue"],
        lw=2.6,
    )
    axes.scatter(
        ladder["hazard_ratio"], positions, color=OKABE_ITO["black"], s=42, zorder=4
    )
    axes.axvline(1.0, color=OKABE_ITO["vermilion"], ls="--", lw=1.4)

    for position, (_, row) in zip(positions, ladder.iterrows(), strict=True):
        axes.text(
            axes.get_xlim()[1],
            position,
            f"  {row['hazard_ratio']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
            f"   {int(row['events'])} events",
            va="center",
            fontsize=8,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{row['model']}\n({row['estimand']})" for _, row in ladder.iterrows()],
        fontsize=8,
    )
    axes.set_xscale("log")
    axes.set_xlabel("hazard ratio (log scale)")
    axes.set_title("Adjustment ladder — every rung, in order")
    figure.tight_layout()
    return save_figure(figure, directory, name)
