"""Judge each tested group of encoder changes by leaving it out.

Reads the evaluation suite's outputs for one cohort: the reference encoder
(three seeds, which set the noise band), the encoder with every group on, and
one encoder per left-out group. For each group, the effect of having it is
``all groups − all groups but this one`` on the primary readouts:

* the mean macro-F1 of the classification probes on the best read-out;
* held-out reconstruction under the reference mask, as loss over the
  gene-mean baseline (lower is better);
* the participation ratio of that read-out.

A group is kept when removing it worsens at least one readout by more than
one reference-seed standard deviation and improves none by as much; the rule
was fixed before the runs (``configs/encoder_ablation/README.md``). Writes
``ablation_forest.png``, ``ablation_effects.csv`` and ``encoder_card.md``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mogformer.evaluation.plots import OKABE_ITO, apply_house_style, save_figure

logger = logging.getLogger(__name__)

#: Groups left out one at a time, in the order they are reported.
GROUPS: tuple[str, ...] = (
    "masking",
    "summary_tokens",
    "identity_modulation",
    "fusion_residual",
)

#: Classification probes averaged into the primary probe readout, per cohort.
PRIMARY_TARGETS: dict[str, tuple[str, ...]] = {
    "tcga": ("PAM50", "ER status", "HER2 status"),
    "metabric": ("PAM50", "ER status", "HER2 status", "Integrative cluster"),
}


@dataclass(frozen=True)
class Readout:
    """One primary readout and the direction that counts as better.

    Attributes:
        column: Column of the readout table.
        label: Axis label.
        higher_is_better: Direction of improvement.
    """

    column: str
    label: str
    higher_is_better: bool


#: The primary readouts of the decision rule.
READOUTS: tuple[Readout, ...] = (
    Readout("probe_macro_f1", "mean macro-F1 of the classification probes", True),
    Readout("heldout_loss_ratio", "held-out loss / gene-mean baseline", False),
    Readout("participation_ratio", "participation ratio", True),
)


def encoder_readouts(evaluation: Path, cohort: str) -> pd.DataFrame:
    """Tabulate the primary readouts of every encoder run in one evaluation.

    For each run the read-out with the highest probe macro-F1 is used.

    Args:
        evaluation: An evaluation directory written by
            :func:`~mogformer.experiments.representation_evaluation.run_representation_evaluation`.
        cohort: ``"tcga"`` or ``"metabric"``.

    Returns:
        One row per run: the chosen read-out, the three primary readouts,
        cross-omics reconstruction (both heads) and the nuisance probe.
    """
    probes = pd.read_csv(evaluation / "probe_summary.csv")
    axes = pd.read_csv(evaluation / "axes_summary.csv").set_index("representation")
    reconstruction_path = evaluation / "reconstruction.csv"
    reconstruction = (
        pd.read_csv(reconstruction_path)
        if reconstruction_path.exists()
        else pd.DataFrame(columns=["run", "task", "omics", "head", "ratio_to_baseline"])
    )

    encoders = probes[probes["representation"].str.startswith("encoder_")].copy()
    parts = encoders["representation"].str.removeprefix("encoder_").str.split("__")
    encoders["readout"] = parts.str[0]
    encoders["run"] = parts.str[1].fillna(evaluation.name)

    rows = []
    for run, group in encoders.groupby("run"):
        primary = group[group["target"].isin(PRIMARY_TARGETS[cohort])]
        f1 = primary.groupby("readout")["mean"].mean()
        best = str(f1.idxmax())
        representation = group.loc[group["readout"] == best, "representation"].iloc[0]
        nuisance = group[(group["readout"] == best) & group["nuisance"]]["mean"]
        # A single-run evaluation names its one run after the evaluation.
        single = run == evaluation.name
        recon = (
            reconstruction if single else reconstruction[reconstruction["run"] == run]
        )

        def ratio(task: str, head: str, recon: pd.DataFrame = recon) -> float:
            cells = recon[(recon["task"] == task) & (recon["head"] == head)]
            return float(cells["ratio_to_baseline"].mean()) if len(cells) else np.nan

        rows.append(
            {
                "run": run,
                "readout": best,
                "probe_macro_f1": float(f1[best]),
                "heldout_loss_ratio": ratio("masked", "local"),
                "participation_ratio": float(
                    axes.loc[representation, "participation_ratio"]
                ),
                "cross_omics_ratio_local": ratio("omics_hidden", "local"),
                "cross_omics_ratio_global": ratio("omics_hidden", "global"),
                "nuisance_probe": float(nuisance.iloc[0]) if len(nuisance) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def group_effects(
    reference: pd.DataFrame,
    all_groups: pd.Series,
    without: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute each group's effect and apply the decision rule.

    Args:
        reference: Readouts of the reference encoder, one row per seed.
        all_groups: Readouts of the encoder with every group on.
        without: Group name to the readouts of the encoder without it.

    Returns:
        One row per group and readout: the effect (all groups minus without),
        the seed standard deviation, the effect in SD units signed so that
        positive means the group helps, and the group's verdict.
    """
    rows = []
    for group, left_out in without.items():
        for readout in READOUTS:
            effect = float(all_groups[readout.column] - left_out[readout.column])
            noise = float(reference[readout.column].std(ddof=1))
            helps = effect if readout.higher_is_better else -effect
            rows.append(
                {
                    "group": group,
                    "readout": readout.column,
                    "all_groups": float(all_groups[readout.column]),
                    "without_group": float(left_out[readout.column]),
                    "effect": effect,
                    "seed_sd": noise,
                    "helps_in_sd": helps / noise if noise > 0 else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    verdicts: dict[str, str] = {}
    for group_name, cells in table.groupby("group"):
        group = str(group_name)
        any_helps = bool((cells["helps_in_sd"] > 1).any())
        any_hurts = bool((cells["helps_in_sd"] < -1).any())
        verdicts[group] = (
            "keep"
            if any_helps and not any_hurts
            else "drop"
            if any_hurts and not any_helps
            else "mixed"
            if any_helps and any_hurts
            else "no difference"
        )
    table["verdict"] = table["group"].map(verdicts)
    return table


def _plot_forest(
    effects: pd.DataFrame, cohort: str, directory: Path, title: str
) -> Path:
    """Draw each group's effect per readout against the seed-noise band."""
    apply_house_style()
    groups = list(dict.fromkeys(effects["group"]))
    figure, axes_row = plt.subplots(
        1, len(READOUTS), figsize=(13, 0.5 * len(groups) + 1.8)
    )
    for axes, readout in zip(axes_row, READOUTS, strict=True):
        cells = effects[effects["readout"] == readout.column].set_index("group")
        noise = float(cells["seed_sd"].iloc[0])
        axes.axvspan(-noise, noise, color=OKABE_ITO["grey"], alpha=0.25, lw=0)
        axes.axvline(0, color="#222222", lw=0.8)
        for row, group in enumerate(groups):
            effect = float(cells.loc[group, "effect"])
            helps = float(cells.loc[group, "helps_in_sd"])
            colour = (
                OKABE_ITO["blue"]
                if helps > 1
                else OKABE_ITO["vermilion"]
                if helps < -1
                else OKABE_ITO["grey"]
            )
            axes.scatter(effect, row, color=colour, s=40, zorder=3)
        axes.set_yticks(range(len(groups)))
        axes.set_yticklabels(groups if axes is axes_row[0] else [])
        axes.invert_yaxis()
        direction = (
            "higher is better" if readout.higher_is_better else "lower is better"
        )
        axes.set_xlabel(f"all groups − without group\n{readout.label} ({direction})")
        axes.grid(axis="y", visible=False)
    figure.suptitle(
        f"{title}\nGrey band: ±1 SD over the reference encoder's 3 seeds; "
        "blue: the group helps beyond it, red: it hurts",
        fontsize=9,
    )
    figure.tight_layout()
    return save_figure(figure, directory, f"ablation_forest_{cohort}")


def run_encoder_ablation(
    evaluations: Path | str,
    output_dir: Path | str,
    cohort: str,
    reference: str,
    prefix: str,
    groups: Sequence[str] = GROUPS,
) -> dict[str, object]:
    """Summarise one cohort's leave-one-group-out runs.

    Args:
        evaluations: The ``representation_evaluation`` results directory.
        output_dir: Where the summary is written.
        cohort: ``"tcga"`` or ``"metabric"``.
        reference: Evaluation name of the reference encoder (several seeds).
        prefix: Evaluation-name prefix of the ablation runs: they are named
            ``<prefix>_all_groups``, ``<prefix>_without_<group>`` and, for the
            copy-number scale, ``<prefix>_all_groups_log_ratio_copy_number``.
        groups: Groups to report, those whose runs exist.

    Returns:
        The verdict per group and where the outputs are.

    Raises:
        FileNotFoundError: If the reference or all-groups evaluation is absent.
    """
    root = Path(evaluations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_table = encoder_readouts(root / reference, cohort)
    all_groups = encoder_readouts(root / f"{prefix}_all_groups", cohort).iloc[0]
    without = {}
    for group in groups:
        directory = root / f"{prefix}_without_{group}"
        if directory.exists():
            without[group] = encoder_readouts(directory, cohort).iloc[0]
        else:
            logger.warning("no evaluation of the encoder without %s yet", group)

    effects = group_effects(reference_table, all_groups, without)
    effects.to_csv(output / f"ablation_effects_{cohort}.csv", index=False)
    table = pd.concat(
        [
            reference_table.assign(encoder="reference"),
            all_groups.to_frame().T.assign(encoder="all groups"),
            *[
                row.to_frame().T.assign(encoder=f"without {group}")
                for group, row in without.items()
            ],
        ],
        ignore_index=True,
    )
    scale = root / f"{prefix}_all_groups_log_ratio_copy_number"
    if scale.exists():
        table = pd.concat(
            [
                table,
                encoder_readouts(scale, cohort).assign(
                    encoder="all groups, log-ratio CN"
                ),
            ],
            ignore_index=True,
        )
    table.to_csv(output / f"ablation_readouts_{cohort}.csv", index=False)
    figure = _plot_forest(
        effects,
        cohort,
        output,
        f"Leave-one-group-out ablation, {cohort.upper()} encoder (one seed per run)",
    )
    _write_card(table, effects, output / f"encoder_card_{cohort}.md", cohort)
    verdicts = effects.drop_duplicates("group").set_index("group")["verdict"].to_dict()
    record: dict[str, object] = {
        "cohort": cohort,
        "verdicts": verdicts,
        "figure": str(figure),
    }
    (output / f"record_{cohort}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _write_card(
    table: pd.DataFrame, effects: pd.DataFrame, path: Path, cohort: str
) -> None:
    """Write the readouts and the verdicts as Markdown."""
    from mogformer.experiments.representation_evaluation import markdown_table

    columns = [
        "encoder",
        "run",
        "readout",
        *(r.column for r in READOUTS),
        "cross_omics_ratio_local",
        "cross_omics_ratio_global",
        "nuisance_probe",
    ]
    lines = [
        f"# Encoder ablation — {cohort.upper()}",
        "",
        "## Primary readouts per encoder",
        "",
        markdown_table(table[columns].round(3)),
        "",
        "## Effect of each group (all groups − without it), against seed noise",
        "",
        markdown_table(effects.round(3)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "GROUPS",
    "PRIMARY_TARGETS",
    "READOUTS",
    "encoder_readouts",
    "group_effects",
    "run_encoder_ablation",
]
