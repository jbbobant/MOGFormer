"""Trans sign concordance under a bias-preserving permutation null.

On TCGA, 60.0% of regulatory edges have their direction matched, which a
binomial test against a coin flip scores at p = 9.3e-10. A coin flip is the
wrong null twice over: the edge set is 77%
activating, and the model's slopes are not sign-balanced either. Both push
concordance up before the model has learned anything about direction.

Two corrections apply together and the module refuses to report one without the
other. A permutation null that shuffles edge labels preserves both biases, so
it says whether the concordance is *significant*. The trivial majority baseline
— always answer "activating" — says whether it is *useful*. Under the corrected
analysis the TCGA result is neither: p is around 0.15, and 60% loses to the
constant predictor's 77%.

Nothing here calls the model. It re-analyses the per-edge slopes a stored
probe sweep already wrote, which is what makes the correction cheap enough to apply
to both cohorts and both masking modes at once.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mogformer.analysis.plots import (
    plot_sign_concordance,
    plot_trans_versus_coexpression,
)
from mogformer.analysis.probe_trans import (
    magnitude_tests,
    permuted_graph_retention,
    sign_permutation_test,
)
from mogformer.evaluation.plots import apply_house_style
from mogformer.experiments.paths import save_json

logger = logging.getLogger(__name__)

#: Column names earlier probe runs used, mapped onto the package's own.
COLUMN_ALIASES: Mapping[str, str] = {
    "obs_coexpr": "observed_coexpression",
    "observed_coexpr": "observed_coexpression",
    "coexpr": "observed_coexpression",
}

#: Columns the analysis cannot proceed without.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "edge_sign",
    "is_edge",
    "null",
    "mask_mode",
    "slope",
)

#: Label marking rows measured on the unmodified graph.
REAL_LABEL = "real"

#: Graph controls a sweep may carry, keyed by null label: what was shuffled.
#: The two answer different questions and are never pooled — shuffling the
#: regulatory graph alone isolates what the curated edges contribute, while
#: shuffling every graph input also removes the interaction network.
GRAPH_NULLS: Mapping[str, str] = {
    "permute_grn_only": "regulatory graph shuffled",
    "permute_graph": "interaction and regulatory graphs shuffled",
}

#: Permutations behind the null distribution.
N_PERMUTATIONS = 10_000


def load_edge_table(path: Path | str) -> pd.DataFrame:
    """Read a per-edge slope table and normalise its column names.

    Args:
        path: A CSV written by an earlier trans sweep.

    Returns:
        The table, with column names matching what the analysis expects.

    Raises:
        ValueError: If a column the analysis needs is absent.
    """
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={old: new for old, new in COLUMN_ALIASES.items() if old in frame}
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing column(s): {missing}")
    return frame


def rerun_sign_test(
    edges_csv: Path | str,
    output_dir: Path | str,
    cohort: str,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 20240712,
) -> dict[str, Any]:
    """Re-test sign concordance for every masking mode in one table.

    Args:
        edges_csv: Per-edge slopes from a trans sweep.
        output_dir: Directory to write tables and figures into.
        cohort: Cohort name, recorded in the outputs.
        n_permutations: Label shuffles behind the null.
        seed: Seed for the shuffling.

    Returns:
        A record of every mode tested, also written as ``sign_tests.json``.
    """
    apply_house_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_edge_table(edges_csv)
    record: dict[str, Any] = {
        "cohort": cohort,
        "source": str(edges_csv),
        "n_permutations": n_permutations,
        "modes": {},
    }

    for mask_mode in sorted(frame["mask_mode"].unique()):
        mode_rows = frame[frame["mask_mode"] == mask_mode]
        real_edges = mode_rows[
            (mode_rows["null"] == REAL_LABEL) & (mode_rows["is_edge"] == 1)
        ]
        if real_edges.empty:
            logger.warning("no real edges under mask mode %s", mask_mode)
            continue

        concordance = sign_permutation_test(
            real_edges, n_perm=n_permutations, seed=seed
        )
        entry: dict[str, Any] = {
            "sign_concordance": {
                "n_edges": concordance.n_edges,
                "concordant": concordance.concordant,
                "rate": concordance.rate,
                "null_mean_rate": concordance.null_mean_rate,
                "null_sd_rate": concordance.null_sd_rate,
                "perm_p": concordance.perm_p,
                "z": concordance.z,
                "frac_slope_positive": concordance.frac_slope_positive,
                "trivial_majority_baseline": concordance.trivial_majority_baseline,
                "beats_trivial_baseline": concordance.beats_trivial_baseline,
                "is_informative": concordance.is_informative,
            },
            "magnitude": magnitude_tests(mode_rows, mask_mode),
        }

        graph_nulls = {}
        for label in GRAPH_NULLS:
            permuted = mode_rows[
                (mode_rows["null"] == label) & (mode_rows["is_edge"] == 1)
            ]
            if not permuted.empty:
                graph_nulls[label] = permuted_graph_retention(real_edges, permuted)
        if graph_nulls:
            entry["graph_nulls"] = graph_nulls

        suffix = str(mask_mode)
        entry["figures"] = {
            "sign_concordance": str(
                plot_sign_concordance(
                    concordance,
                    output,
                    name=f"sign_concordance_{suffix}",
                    context=f"{cohort}, mask mode {suffix}",
                )
            ),
            "trans_vs_coexpression": str(
                plot_trans_versus_coexpression(
                    real_edges,
                    output,
                    name=f"trans_vs_coexpression_{suffix}",
                    context=f"{cohort}, mask mode {suffix}",
                )
            )
            if "observed_coexpression" in real_edges
            else None,
        }
        record["modes"][suffix] = entry
        real_edges.to_csv(output / f"edges_{suffix}.csv", index=False)

    save_json(record, output / "sign_tests.json")
    (output / "sign_tests.md").write_text(render_sign_tests(record), encoding="utf-8")
    logger.info("sign tests written to %s", output)
    return record


def render_sign_tests(record: Mapping[str, Any]) -> str:
    """Render the corrected sign test as Markdown, verdict first.

    Args:
        record: Output of :func:`rerun_sign_test`.

    Returns:
        A Markdown document.
    """
    modes: Mapping[str, Any] = record["modes"]
    lines = [
        f"# Trans sign concordance — {record['cohort']}",
        "",
        "Concordance is tested against a permutation null that shuffles edge",
        "labels, which preserves both the activating/repressing imbalance and",
        "the model's own slope-sign bias. It is also compared against the",
        "trivial majority baseline — always answering with the commoner edge",
        "sign. A result counts as informative only if it clears both.",
        "",
        "| mask mode | rate | null | perm p | trivial baseline | informative |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode, entry in modes.items():
        sign: Mapping[str, Any] = entry["sign_concordance"]
        lines.append(
            f"| `{mode}` | {sign['rate']:.3f} | {sign['null_mean_rate']:.3f} | "
            f"{sign['perm_p']:.3f} | {sign['trivial_majority_baseline']:.3f} | "
            f"{'yes' if sign['is_informative'] else '**no**'} |"
        )

    lines += [
        "",
        "## Effect magnitude on real edges versus matched controls",
        "",
        "This is the claim that stands independently of direction: whether the",
        "model routes *any* effect preferentially along real edges.",
        "",
        "| mask mode | edge median | control median | Mann-Whitney p | "
        "intervals excluding zero |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode, entry in modes.items():
        magnitude: Mapping[str, Any] = entry["magnitude"]
        if "edge_abs_median" not in magnitude:
            continue
        mann_whitney = magnitude.get("mw_p_edge_stronger")
        lines.append(
            f"| `{mode}` | {magnitude['edge_abs_median']:.4g} | "
            f"{magnitude['control_abs_median']:.4g} | "
            f"{'n/a' if mann_whitney is None else format(mann_whitney, '.3g')} | "
            f"{magnitude['frac_edges_interval_excludes_zero']:.3f} |"
        )

    retentions = [
        (mode, label, retention)
        for mode, entry in modes.items()
        for label, retention in entry.get("graph_nulls", {}).items()
    ]
    if retentions:
        lines += [
            "",
            "## How much of the effect is the graph's doing",
            "",
            "Permuting node identities and re-measuring. What survives was",
            "never routed through the graph that was shuffled.",
            "",
            "| mask mode | control | retained | attributable |",
            "| --- | --- | --- | --- |",
        ]
        for mode, label, retention in retentions:
            lines.append(
                f"| `{mode}` | {GRAPH_NULLS[label]} | "
                f"{retention['retained_pct']:.1f}% | "
                f"{retention['graph_attributable_pct']:.1f}% |"
            )

    return "\n".join(lines) + "\n"
