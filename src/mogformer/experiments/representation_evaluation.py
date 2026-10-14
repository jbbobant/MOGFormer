"""Run the representation evaluation suite on frozen encoders or on inputs.

Two entry points share one evaluator:

* :func:`run_representation_evaluation` reads one or more pretraining runs of
  the same configuration (one per seed), scores every patient read-out each
  run saved against comparators of equal dimension built from the same
  inputs, and adds what only trained encoders have: held-out and cross-omics
  reconstruction against the gene-mean baseline, and seed stability;
* :func:`run_copy_number_scale_check` needs no encoder: it rebuilds the inputs
  with absolute and with ploidy-relative copy number and scores the
  comparators on both, side by side.

Both write per-fold scores, summaries, figures and a Markdown digest under
``representation_evaluation/<name>/``. No survival column is read.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from mogformer.analysis.clustering import run_consensus_clustering
from mogformer.analysis.representation_evaluation import (
    PROLIFERATION_GENES,
    ProbeTarget,
    clustering_checkpoint,
    comparator_representations,
    dominant_axis_share,
    linear_cka,
    participation_ratio,
    per_gene_rank_correlation,
    probe_panel,
    summarise_probe_panel,
)
from mogformer.data.omics import (
    OmicsData,
    apply_copy_number_representation,
    load_curated_genes,
    load_omics_bundle,
    load_ploidy,
)
from mogformer.data.preprocess import MultiOmicsTransformer
from mogformer.evaluation.plots import OKABE_ITO, apply_house_style, save_figure

logger = logging.getLogger(__name__)

#: Cohort archive of the self-supervised cohort, and its per-patient ploidy
#: and burden, as written by ``utils.build_cohort_bundle``.
SSL_COHORT_FILE = "cohort_ssl.npz"
PLOIDY_FILE = "ploidy.csv"

#: Patient read-outs a pretraining run saves, in order of preference.
READOUTS: tuple[str, ...] = ("summary", "summary_masked_mean", "gene_mean")

#: Plain-language name of each read-out, for figures.
READOUT_LABELS: dict[str, str] = {
    "summary": "summary token",
    "summary_masked_mean": "summary token, mean over masks",
    "gene_mean": "mean of gene states",
}

#: Plain-language name of each comparator, for figures.
COMPARATOR_LABELS: dict[str, str] = {
    "pca_concatenated": "PCA, all omics",
    "pca_per_omics": "PCA per omics",
    "raw_inputs": "Raw inputs",
    "mofa": "MOFA+ (at most 128 factors)",
}

#: Number of variance-ranked genes of the archived encoder's inputs, used to
#: rebuild them for the copy-number scale check.
ARCHIVED_TOP_K = 250

#: Two-cluster rule that reproduced the archived Luminal A split.
PLOIDY_SPLIT_THRESHOLD = 3.0

#: Partition for consensus clustering of seed stability.
SEED_STABILITY_K: tuple[int, ...] = (2, 3, 4, 5, 6)

INK = "#222222"

_PALETTE: tuple[str, ...] = (
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["grey"],
    OKABE_ITO["vermilion"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
    OKABE_ITO["black"],
)
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


@dataclass
class CohortContext:
    """What a cohort supplies to the suite, aligned to the representations.

    Attributes:
        cohort: Display name of the cohort.
        patients: Patient barcodes, in representation row order.
        covariates: Per-patient covariates (``burden``, ``proliferation`` and,
            where absolute copy number exists, ``ploidy``).
        targets: Probe targets, nuisance labels last.
        populations: Fixed populations of the clustering checkpoint, by name,
            as row indices.
        references: Known cohort-wide splits a stratification must not merely
            reproduce, ``-1`` where a patient has no value.
    """

    cohort: str
    patients: list[str]
    covariates: pd.DataFrame
    targets: list[ProbeTarget]
    populations: dict[str, np.ndarray]
    references: dict[str, np.ndarray]


def _stage(value: object) -> float:
    """Collapse an AJCC pathologic stage to I–IV as 1–4, NaN when unknown."""
    text = str(value).upper().replace("STAGE", "").strip()
    for numeral, number in (("IV", 4.0), ("III", 3.0), ("II", 2.0), ("I", 1.0)):
        if text.startswith(numeral):
            return number
    return float("nan")


def proliferation_score(data: OmicsData, log_counts: bool) -> pd.Series:
    """Score proliferation as the mean z-scored expression of its genes.

    Args:
        data: Cohort with raw expression in its ``rna`` block.
        log_counts: Take ``log1p`` first, for expression distributed as counts.

    Returns:
        One score per patient, indexed by barcode.
    """
    genes = [gene for gene in PROLIFERATION_GENES if gene in data.gene_names]
    columns = [data.gene_names.index(gene) for gene in genes]
    values = data.X[:, data.block_slice("rna")][:, columns].astype(float)
    if log_counts:
        values = np.log1p(np.clip(values, 0, None))
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    return pd.Series(z.mean(axis=1), index=data.patient_ids, name="proliferation")


def _two_means(values: np.ndarray, seed: int) -> np.ndarray:
    """Split one covariate in two by k-means, ``-1`` where it is missing."""
    labels = np.full(len(values), -1)
    known = ~np.isnan(values)
    labels[known] = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(
        values[known].reshape(-1, 1)
    )
    return labels


def tcga_context(patients: list[str], bundle: Path, seed: int = 42) -> CohortContext:
    """Assemble the TCGA covariates, probe targets and populations.

    Args:
        patients: Patient barcodes, in representation order.
        bundle: The TCGA bundle directory (``clinical.csv``,
            ``receptors.csv``, ``ploidy.csv``, ``cohort_ssl.npz``).
        seed: Seed of the two-means burden split.

    Returns:
        The cohort context.
    """
    data = load_omics_bundle(bundle / SSL_COHORT_FILE)
    copy_number = pd.read_csv(bundle / PLOIDY_FILE, index_col="patient_id")
    covariates = pd.DataFrame(
        {
            "ploidy": copy_number["ploidy"].reindex(patients),
            "burden": copy_number["burden"].reindex(patients),
            "proliferation": proliferation_score(data, log_counts=True).reindex(
                patients
            ),
        }
    )
    clinical = pd.read_csv(bundle / "clinical.csv").set_index("patient")
    clinical = clinical[~clinical.index.duplicated()].reindex(patients)
    receptors = pd.read_csv(bundle / "receptors.csv", index_col=0).reindex(patients)
    index = pd.Index(patients)

    def known(series: pd.Series) -> pd.Series:
        return series.where(series.isin(["Positive", "Negative"])).set_axis(index)

    targets = [
        ProbeTarget("PAM50", clinical["SUBTYPE"].set_axis(index), "classification"),
        ProbeTarget("ER status", known(receptors["er"]), "classification"),
        ProbeTarget("HER2 status", known(receptors["her2"]), "classification"),
        ProbeTarget(
            "Stage",
            clinical["clinical__diagnoses.ajcc_pathologic_stage"]
            .map(_stage)
            .set_axis(index),
            "regression",
        ),
        ProbeTarget(
            "Age",
            clinical["clinical__demographic.age_at_index"]
            .astype(float)
            .set_axis(index),
            "regression",
        ),
        ProbeTarget("Proliferation", covariates["proliferation"], "regression"),
        ProbeTarget("CN burden", covariates["burden"], "regression"),
        ProbeTarget("Ploidy", covariates["ploidy"], "regression"),
        ProbeTarget(
            "Tissue source site",
            pd.Series([p.split("-")[1] for p in patients], index=index),
            "classification",
            nuisance=True,
        ),
    ]
    populations = {
        "TCGA Luminal A": np.flatnonzero(clinical["SUBTYPE"].to_numpy() == "BRCA_LumA"),
        "TCGA ER+/HER2−": np.flatnonzero(
            receptors["er_positive_her2_negative"].fillna(False).to_numpy(dtype=bool)
        ),
    }
    references = {
        "ploidy": (covariates["ploidy"].to_numpy() >= PLOIDY_SPLIT_THRESHOLD).astype(
            int
        ),
        "burden": _two_means(covariates["burden"].to_numpy(), seed),
    }
    return CohortContext(
        "TCGA-BRCA", patients, covariates, targets, populations, references
    )


def metabric_context(
    patients: list[str], bundle: Path, seed: int = 42
) -> CohortContext:
    """Assemble the METABRIC covariates, probe targets and populations.

    METABRIC distributes ordinal copy-number calls, not absolute copies, so
    ploidy cannot be estimated and is absent; burden is the fraction of genes
    called other than neutral.

    Args:
        patients: Patient barcodes, in representation order.
        bundle: The METABRIC self-supervised bundle (``cohort_ssl.npz``,
            ``clinical.csv``).
        seed: Seed of the two-means burden split.

    Returns:
        The cohort context.
    """
    data = load_omics_bundle(bundle / SSL_COHORT_FILE)
    codes = data.X[:, data.block_slice("cnv")].astype(float)
    burden = pd.Series(
        np.nanmean(np.where(np.isnan(codes), np.nan, codes != 2), axis=1),
        index=data.patient_ids,
    )
    covariates = pd.DataFrame(
        {
            "burden": burden.reindex(patients),
            "proliferation": proliferation_score(data, log_counts=False).reindex(
                patients
            ),
        }
    )
    clinical = pd.read_csv(bundle / "clinical.csv", index_col=0).reindex(patients)
    index = pd.Index(patients)
    er = clinical["ER_IHC"].map({"Positve": "Positive", "Negative": "Negative"})
    her2 = clinical["HER2_SNP6"].map(
        {"GAIN": "Positive", "NEUTRAL": "Negative", "LOSS": "Negative"}
    )
    targets = [
        ProbeTarget(
            "PAM50", clinical["CLAUDIN_SUBTYPE"].set_axis(index), "classification"
        ),
        ProbeTarget("ER status", er.set_axis(index), "classification"),
        ProbeTarget("HER2 status", her2.set_axis(index), "classification"),
        ProbeTarget(
            "Integrative cluster",
            clinical["INTCLUST"].astype(str).set_axis(index),
            "classification",
        ),
        ProbeTarget(
            "Age",
            clinical["AGE_AT_DIAGNOSIS"].astype(float).set_axis(index),
            "regression",
        ),
        ProbeTarget("Proliferation", covariates["proliferation"], "regression"),
        ProbeTarget("CN burden", covariates["burden"], "regression"),
        ProbeTarget(
            "Cohort (processing batch)",
            clinical["COHORT"].astype(str).set_axis(index),
            "classification",
            nuisance=True,
        ),
    ]
    populations = {
        "METABRIC ER+/HER2−": np.flatnonzero(
            ((er == "Positive") & (her2 == "Negative")).to_numpy()
        ),
        "METABRIC Luminal A": np.flatnonzero(
            (clinical["CLAUDIN_SUBTYPE"] == "LumA").to_numpy()
        ),
    }
    references = {"burden": _two_means(covariates["burden"].to_numpy(), seed)}
    return CohortContext(
        "METABRIC", patients, covariates, targets, populations, references
    )


#: Context builder per cohort name.
COHORT_CONTEXTS = {"tcga": tcga_context, "metabric": metabric_context}


def load_encoder_run(
    run_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Read a pretraining run's patient read-outs and the inputs it consumed.

    Args:
        run_dir: Directory holding ``probe_inputs.pt`` and either
            ``embeddings/<read-out>.parquet`` files or a single
            ``C_all.parquet`` (the summary token of an archived run).

    Returns:
        The read-outs by name (patients x dimensions), the standardised inputs
        (patients x genes x omics) and the patient identifiers, all aligned.

    Raises:
        ValueError: If a read-out lists different patients from the inputs.
    """
    inputs = torch.load(run_dir / "probe_inputs.pt", weights_only=False)
    patients = [str(p) for p in inputs["patient_ids"]]
    files = {
        name: run_dir / "embeddings" / f"{name}.parquet"
        for name in READOUTS
        if (run_dir / "embeddings" / f"{name}.parquet").exists()
    } or {"summary": run_dir / "C_all.parquet"}

    readouts = {}
    for name, path in files.items():
        table = pd.read_parquet(path)
        if "patient_id" in table.columns and table["patient_id"].dtype == object:
            table = table.set_index("patient_id").reindex(patients)
        if len(table) != len(patients) or table.isna().all(axis=1).any():
            raise ValueError(f"{path} and the inputs cover different patients")
        columns = [c for c in table.columns if str(c).startswith("c_")]
        readouts[name] = table[columns].to_numpy(dtype=float)
    return readouts, inputs["inputs"].numpy().astype(float), patients


def standardised_inputs(
    data: OmicsData,
    curated_genes: Sequence[str],
    top_k: int = ARCHIVED_TOP_K,
) -> np.ndarray:
    """Select and standardise a cohort's inputs as the archived encoder did.

    Args:
        data: Cohort, copy number already on the scale to evaluate.
        curated_genes: Genes force-included after variance ranking.
        top_k: Size of the variance-ranked pool.

    Returns:
        Standardised inputs, patients x selected genes x omics.
    """
    preprocessor = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=top_k,
        curated_genes=list(curated_genes),
    ).fit(data.X)
    features = preprocessor.transform(data.X)
    n_selected = len(preprocessor.get_selected_gene_names())
    return np.stack(
        [features[:, k * n_selected : (k + 1) * n_selected] for k in range(3)], axis=-1
    )


def _styles(names: Sequence[str], labels: Mapping[str, str]) -> dict[str, tuple]:
    """Give every representation a label, a colour and a marker."""
    return {
        name: (labels.get(name, name), _PALETTE[i % 8], _MARKERS[i % 8])
        for i, name in enumerate(names)
    }


def _plot_probe_panel(
    summary: pd.DataFrame, styles: Mapping[str, tuple], directory: Path, title: str
) -> Path:
    """Draw every probe's mean and corrected interval per representation."""
    apply_house_style()
    targets = list(dict.fromkeys(summary["target"]))
    names = [name for name in styles if name in set(summary["representation"])]
    figure, axes = plt.subplots(figsize=(9, 0.32 * len(targets) * len(names) + 1.8))
    offsets = np.linspace(0.3, -0.3, len(names))
    for row, target in enumerate(targets):
        y = len(targets) - 1 - row
        for offset, name in zip(offsets, names, strict=True):
            cell = summary[
                (summary["target"] == target) & (summary["representation"] == name)
            ]
            if cell.empty:
                continue
            label, colour, marker = styles[name]
            axes.hlines(
                y + offset, cell["nb_ci95_lo"], cell["nb_ci95_hi"], color=colour, lw=1.6
            )
            axes.scatter(
                cell["mean"],
                y + offset,
                color=colour,
                marker=marker,
                s=30,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label=label if row == 0 else None,
            )
    labels = []
    for target in targets:
        rows = summary[summary["target"] == target]
        metric = "macro-F1" if rows["task"].iloc[0] == "classification" else "R²"
        nuisance = ", lower is better" if bool(rows["nuisance"].iloc[0]) else ""
        labels.append(f"{target} ({metric}{nuisance})")
    axes.set_yticks(range(len(targets))[::-1])
    axes.set_yticklabels(labels, color=INK)
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("linear-probe score (mean, Nadeau–Bengio 95% CI over 25 estimates)")
    axes.set_title(title, color=INK, fontsize=10)
    axes.legend(loc="lower right", fontsize=7)
    figure.tight_layout()
    return save_figure(figure, directory, "probe_panel")


def _plot_axes_summary(
    axes_table: pd.DataFrame, styles: Mapping[str, tuple], directory: Path, title: str
) -> Path:
    """Draw effective dimension and dominant-axis shares per representation."""
    apply_house_style()
    names = [n for n in styles if n in set(axes_table["representation"])]
    shares = [c for c in ("ploidy_share", "burden_share") if c in axes_table.columns]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 0.42 * len(names) + 1.6))
    for i, name in enumerate(names):
        _, colour, _ = styles[name]
        row = axes_table[axes_table["representation"] == name].iloc[0]
        left.barh(i, row["participation_ratio"], color=colour)
        left.text(
            row["participation_ratio"],
            i,
            f" {row['participation_ratio']:.1f}",
            va="center",
            fontsize=8,
            color=INK,
        )
        for j, column in enumerate(shares):
            right.barh(
                i + (j - (len(shares) - 1) / 2) * 0.36,
                row[column],
                height=0.34,
                color=colour,
                hatch="" if column == "ploidy_share" else "//",
                edgecolor="white",
            )
    for ax in (left, right):
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([styles[n][0] for n in names], color=INK, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    left.set_xlabel("participation ratio (effective dimensions)")
    right.set_xlabel(
        "share of total variance explained ("
        + ", ".join(
            "solid: ploidy" if c == "ploidy_share" else "hatched: burden"
            for c in shares
        )
        + ")"
    )
    right.set_xlim(0, 1)
    figure.suptitle(title, color=INK, fontsize=10, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, directory, "axes_summary")


def _plot_checkpoint(
    checkpoint: pd.DataFrame,
    styles: Mapping[str, tuple],
    reference: str,
    directory: Path,
    title: str,
) -> Path:
    """Draw consensus PAC and agreement with a known split per population."""
    apply_house_style()
    populations = list(dict.fromkeys(checkpoint["population"]))
    figure, axes_row = plt.subplots(
        1, len(populations), figsize=(4.8 * len(populations), 3.6), squeeze=False
    )
    for axes, population in zip(axes_row[0], populations, strict=True):
        part = checkpoint[checkpoint["population"] == population]
        for name in styles:
            rows = part[part["representation"] == name]
            if rows.empty:
                continue
            label, colour, marker = styles[name]
            axes.plot(
                rows["k"],
                rows["pac"],
                color=colour,
                marker=marker,
                label=f"{label}: PAC",
            )
            axes.plot(
                rows["k"],
                rows[f"ari_{reference}"],
                color=colour,
                marker=marker,
                ls=":",
                label=f"{label}: ARI with {reference} split",
            )
        n = int(part["n_patients"].iloc[0])
        axes.set_title(f"{population} (n = {n})", color=INK, fontsize=9)
        axes.set_xlabel("number of clusters K")
        axes.set_xticks(sorted(part["k"].unique()))
        axes.set_ylim(-0.05, 1.05)
    axes_row[0][0].set_ylabel("PAC (solid; lower is more stable)\nARI (dotted)")
    axes_row[0][-1].legend(fontsize=6, loc="upper right")
    figure.suptitle(title, color=INK, fontsize=10, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, directory, "clustering_checkpoint")


def axes_table_for(
    representations: Mapping[str, np.ndarray], covariates: pd.DataFrame
) -> pd.DataFrame:
    """Tabulate effective dimension and dominant-axis shares (E3, E4).

    Args:
        representations: Name to patients x dimensions.
        covariates: Per-patient covariates; ``ploidy`` and ``burden`` are used
            where present.

    Returns:
        One row per representation.
    """
    names = [c for c in ("ploidy", "burden") if c in covariates.columns]
    rows = []
    for name, values in representations.items():
        shares = dominant_axis_share(
            values, {c: covariates[c].to_numpy(dtype=float) for c in names}
        ).set_index("covariate")
        row: dict[str, object] = {
            "representation": name,
            "dimensions": values.shape[1],
            "participation_ratio": participation_ratio(values),
            "pc1_variance_share": shares["pc1_variance_share"].iloc[0],
        }
        for covariate in names:
            row[f"{covariate}_pc1_r2"] = shares.loc[covariate, "pc1_r2"]
            row[f"{covariate}_share"] = shares.loc[covariate, "variance_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_representations(
    representations: Mapping[str, np.ndarray],
    context: CohortContext,
    output: Path,
    title: str,
    reference: str,
    checkpoint_names: Sequence[str],
    labels: Mapping[str, str],
    n_resample: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Score representations on probes, axes and the clustering checkpoint.

    Args:
        representations: Name to patients x dimensions, rows aligned with
            ``context.patients``.
        context: The cohort's covariates, targets and populations.
        output: Directory written into.
        title: Figure and digest title, naming cohort and inputs.
        reference: Representation every paired probe difference is taken to.
        checkpoint_names: Representations clustered in the checkpoint.
        labels: Display name per representation.
        n_resample: Consensus resamples per K.
        seed: Seed of every partition and resampling.

    Returns:
        The probe summary, axes table and checkpoint, and the figure paths.
    """
    output.mkdir(parents=True, exist_ok=True)
    styles = _styles(list(representations), labels)

    per_fold = probe_panel(representations, context.targets, seed=seed)
    per_fold.to_csv(output / "probe_scores_per_fold.csv", index=False)
    summary = summarise_probe_panel(per_fold, reference=reference)
    summary.to_csv(output / "probe_summary.csv", index=False)

    axes_table = axes_table_for(representations, context.covariates)
    axes_table.to_csv(output / "axes_summary.csv", index=False)

    checkpoint = pd.concat(
        [
            clustering_checkpoint(
                representations[name],
                context.populations,
                context.references,
                n_resample=n_resample,
                seed=seed,
            ).assign(representation=name)
            for name in checkpoint_names
        ],
        ignore_index=True,
    )
    checkpoint.to_csv(output / "clustering_checkpoint.csv", index=False)

    split = "ploidy" if "ploidy" in context.references else "burden"
    checkpoint_styles = {name: styles[name] for name in checkpoint_names}
    figures = {
        "probe_panel": str(
            _plot_probe_panel(summary, styles, output, f"Linear probes: {title}")
        ),
        "axes_summary": str(
            _plot_axes_summary(axes_table, styles, output, f"Dominant axes: {title}")
        ),
        "clustering_checkpoint": str(
            _plot_checkpoint(
                checkpoint,
                checkpoint_styles,
                split,
                output,
                f"Clustering checkpoint: {title}",
            )
        ),
    }
    return {
        "summary": summary,
        "axes": axes_table,
        "checkpoint": checkpoint,
        "figures": figures,
    }


def seed_stability(
    readouts_by_seed: Mapping[str, Mapping[str, np.ndarray]],
    populations: Mapping[str, np.ndarray],
    n_resample: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare encoders trained from different seeds (E7).

    Args:
        readouts_by_seed: Seed label to read-out name to embedding.
        populations: Fixed populations clustered for partition agreement.
        n_resample: Consensus resamples per K.
        seed: Seed of the clustering.

    Returns:
        One row per read-out, pair of seeds and population: linear CKA over
        the whole cohort, and the ARI between the two seeds' consensus
        partitions at every K.
    """
    rows = []
    seeds = list(readouts_by_seed)
    readouts = list(next(iter(readouts_by_seed.values())))
    for readout in readouts:
        labels: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for run in seeds:
            for population, indices in populations.items():
                results = run_consensus_clustering(
                    readouts_by_seed[run][readout][indices],
                    n_clusters_list=SEED_STABILITY_K,
                    n_resample=n_resample,
                    seed=seed,
                )
                labels[(run, population)] = {k: r.labels for k, r in results.items()}
        for first, second in combinations(seeds, 2):
            cka = linear_cka(
                readouts_by_seed[first][readout], readouts_by_seed[second][readout]
            )
            for population in populations:
                row: dict[str, object] = {
                    "readout": readout,
                    "seed_a": first,
                    "seed_b": second,
                    "linear_cka": cka,
                    "population": population,
                }
                for k in SEED_STABILITY_K:
                    row[f"ari_k{k}"] = adjusted_rand_score(
                        labels[(first, population)][k], labels[(second, population)][k]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def reconstruction_table(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """Gather held-out and cross-omics reconstruction of every run (E5, E6).

    Args:
        run_dirs: Pretraining runs, each with ``heldout_reconstruction.json``.

    Returns:
        One row per run, task, omics and head, with the gene-mean baseline
        beside it; empty when no run recorded one.
    """
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "heldout_reconstruction.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in record["rows"]:
            rows.append({"run": run_dir.name, **row})
    return pd.DataFrame(rows)


def run_representation_evaluation(
    run_dirs: Sequence[Path | str],
    bundle: Path | str,
    output_dir: Path | str,
    cohort: str = "tcga",
    name: str | None = None,
    n_resample: int = 200,
    seed: int = 42,
    with_mofa: bool = True,
) -> dict[str, object]:
    """Evaluate the encoders of one configuration against their comparators.

    Every read-out of every run is probed; comparators are built from the
    first run's inputs (all runs of one configuration share them). The
    checkpoint clusters the first run's read-outs and PCA of all omics.

    Args:
        run_dirs: Pretraining runs of one configuration, one per seed.
        bundle: The cohort's bundle directory.
        output_dir: Where results are written; a subdirectory per evaluation.
        cohort: ``"tcga"`` or ``"metabric"``.
        name: Subdirectory name; the first run's name when omitted.
        n_resample: Consensus resamples per K in the clustering checkpoint.
        with_mofa: Add MOFA+ to the comparators.
        seed: Seed of every partition and resampling.

    Returns:
        A record of what was written and the headline numbers.

    Raises:
        ValueError: If the runs were not trained on the same patients and
            inputs, or the cohort is unknown.
    """
    if cohort not in COHORT_CONTEXTS:
        raise ValueError(f"unknown cohort {cohort!r}; {sorted(COHORT_CONTEXTS)}")
    runs = [Path(run) for run in run_dirs]
    bundle = Path(bundle)
    output = Path(output_dir) / (name or runs[0].name)

    readouts_by_run: dict[str, dict[str, np.ndarray]] = {}
    inputs, patients = None, None
    for run in runs:
        readouts, run_inputs, run_patients = load_encoder_run(run)
        if inputs is None:
            inputs, patients = run_inputs, run_patients
        elif run_patients != patients or run_inputs.shape != inputs.shape:
            raise ValueError(f"{run} was trained on other patients or inputs")
        readouts_by_run[run.name] = readouts
    assert inputs is not None and patients is not None

    first = runs[0].name
    many = len(runs) > 1
    representations: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for run_name, readouts in readouts_by_run.items():
        for readout, values in readouts.items():
            key = f"encoder_{readout}" + (f"__{run_name}" if many else "")
            representations[key] = values
            labels[key] = f"Encoder, {READOUT_LABELS[readout]}" + (
                f" ({run})" if many else ""
            )
    width = next(iter(representations.values())).shape[1]
    representations.update(
        comparator_representations(inputs, width, seed=seed, with_mofa=with_mofa)
    )
    labels.update(COMPARATOR_LABELS)
    reference = "encoder_summary" + (f"__{first}" if many else "")

    context = COHORT_CONTEXTS[cohort](patients, bundle, seed)
    title = f"{context.cohort}, {len(patients)} patients, encoder {output.name}"
    first_readouts = [
        key
        for key in representations
        if key.startswith("encoder_") and (not many or key.endswith(f"__{first}"))
    ]
    evaluation = evaluate_representations(
        representations,
        context,
        output,
        title,
        reference,
        [*first_readouts, "pca_concatenated"],
        labels,
        n_resample=n_resample,
        seed=seed,
    )

    reconstruction = reconstruction_table(runs)
    if not reconstruction.empty:
        reconstruction.to_csv(output / "reconstruction.csv", index=False)
    stability = pd.DataFrame()
    if many:
        stability = seed_stability(
            readouts_by_run,
            context.populations,
            n_resample=max(n_resample // 2, 50),
            seed=seed,
        )
        stability.to_csv(output / "seed_stability.csv", index=False)

    _write_markdown(
        evaluation, reconstruction, stability, output / "evaluation_summary.md", title
    )
    record = {
        "runs": [str(run) for run in runs],
        "cohort": context.cohort,
        "n_patients": len(patients),
        "embedding_dimensions": int(width),
        "figures": evaluation["figures"],
        "participation_ratio": {
            row["representation"]: round(float(row["participation_ratio"]), 2)
            for _, row in cast(pd.DataFrame, evaluation["axes"]).iterrows()
        },
    }
    (output / "record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def run_copy_number_scale_check(
    bundle: Path | str,
    output_dir: Path | str,
    n_resample: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Score the comparators on absolute and on ploidy-relative copy number.

    The inputs are rebuilt both ways from the same archive with the archived
    encoder's preprocessing, so the only difference is the copy-number scale.
    Also measures, gene by gene, how strongly expression follows copy number
    on each scale over every gene of the cohort.

    Args:
        bundle: The TCGA bundle directory.
        output_dir: Where results are written (``ploidy_relative_inputs/``).
        n_resample: Consensus resamples per K in the clustering checkpoint.
        seed: Seed of every partition and resampling.

    Returns:
        A record of the headline numbers.
    """
    bundle = Path(bundle)
    output = Path(output_dir) / "ploidy_relative_inputs"
    data = load_omics_bundle(bundle / SSL_COHORT_FILE)
    ploidy = load_ploidy(bundle / PLOIDY_FILE)
    curated = load_curated_genes(bundle / "curated_genes.txt", data.gene_names)
    scales = {
        "absolute": data,
        "ploidy_relative": apply_copy_number_representation(
            data, "ploidy_relative", ploidy
        ),
    }

    representations: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for scale, cohort in scales.items():
        inputs = standardised_inputs(cohort, curated)
        logger.info("%s copy number: %d selected genes", scale, inputs.shape[1])
        for name, values in comparator_representations(inputs, 128, seed=seed).items():
            key = f"{name}__{scale}"
            representations[key] = values
            labels[key] = f"{COMPARATOR_LABELS[name]}, {scale.replace('_', '-')} CN"

    context = tcga_context(list(data.patient_ids), bundle, seed)
    title = (
        f"TCGA-BRCA, {data.n_patients} patients, comparators on absolute vs "
        "ploidy-relative copy number"
    )
    evaluation = evaluate_representations(
        representations,
        context,
        output,
        title,
        "pca_concatenated__absolute",
        ["pca_concatenated__absolute", "pca_concatenated__ploidy_relative"],
        labels,
        n_resample=n_resample,
        seed=seed,
    )

    expression = np.log1p(np.clip(data.X[:, data.block_slice("rna")], 0, None))
    dosage = pd.DataFrame(
        {
            "gene": data.gene_names,
            "rho_absolute": per_gene_rank_correlation(
                expression, data.X[:, data.block_slice("cnv")]
            ),
            "rho_ploidy_relative": per_gene_rank_correlation(
                expression,
                scales["ploidy_relative"].X[:, data.block_slice("cnv")],
            ),
        }
    ).dropna()
    dosage.to_csv(output / "expression_copy_number_correlation.csv", index=False)
    difference = dosage["rho_ploidy_relative"] - dosage["rho_absolute"]
    test = wilcoxon(difference, alternative="greater")
    dosage_record = {
        "n_genes": len(dosage),
        "median_rho_absolute": float(dosage["rho_absolute"].median()),
        "median_rho_ploidy_relative": float(dosage["rho_ploidy_relative"].median()),
        "share_of_genes_rising": float((difference > 0).mean()),
        "wilcoxon_one_sided_p": float(test.pvalue),
    }
    _plot_dosage(dosage, dosage_record, output, title)

    _write_markdown(
        evaluation,
        pd.DataFrame(),
        pd.DataFrame(),
        output / "evaluation_summary.md",
        title,
        extra={"Expression–copy-number rank correlation, per gene": dosage_record},
    )
    record = {
        "n_patients": data.n_patients,
        "figures": evaluation["figures"],
        "dosage": dosage_record,
        "axes": cast(pd.DataFrame, evaluation["axes"])
        .round(4)
        .to_dict(orient="records"),
    }
    (output / "record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _plot_dosage(
    dosage: pd.DataFrame, record: Mapping[str, float], directory: Path, title: str
) -> Path:
    """Draw per-gene expression–copy-number correlation on both scales."""
    apply_house_style()
    figure, axes = plt.subplots(figsize=(5.2, 5))
    axes.scatter(
        dosage["rho_absolute"],
        dosage["rho_ploidy_relative"],
        s=2,
        alpha=0.25,
        color=OKABE_ITO["blue"],
        rasterized=True,
    )
    low = float(min(dosage["rho_absolute"].min(), dosage["rho_ploidy_relative"].min()))
    axes.plot([low, 1], [low, 1], color=OKABE_ITO["vermilion"], lw=1, ls="--")
    axes.set_xlabel("Spearman ρ(expression, absolute copy number)")
    axes.set_ylabel("Spearman ρ(expression, ploidy-relative copy number)")
    axes.set_title(
        f"{title}\nn = {record['n_genes']} genes; ρ rises for "
        f"{100 * record['share_of_genes_rising']:.0f}% "
        f"(Wilcoxon p = {record['wilcoxon_one_sided_p']:.1e})",
        color=INK,
        fontsize=8,
    )
    figure.tight_layout()
    return save_figure(figure, directory, "expression_copy_number_correlation")


def _write_markdown(
    evaluation: Mapping[str, object],
    reconstruction: pd.DataFrame,
    stability: pd.DataFrame,
    path: Path,
    title: str,
    extra: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """Write the result tables as one readable Markdown file."""
    summary = evaluation["summary"]
    axes_table = evaluation["axes"]
    checkpoint = evaluation["checkpoint"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(axes_table, pd.DataFrame)
    assert isinstance(checkpoint, pd.DataFrame)
    pivot = summary.pivot(index="target", columns="representation", values="mean")
    lines = [
        f"# Representation evaluation — {title}",
        "",
        "## Linear probes (mean over 25 estimates; macro-F1 or R²)",
        "",
        markdown_table(pivot.round(3).reset_index()),
        "",
        "## Effective dimension and dominant axes",
        "",
        markdown_table(axes_table.round(3)),
    ]
    best = checkpoint.loc[
        checkpoint.groupby(["population", "representation"])["pac"].idxmin()
    ]
    lines += [
        "",
        "## Clustering checkpoint (most stable K per population; structure only)",
        "",
        markdown_table(best.round(3)),
    ]
    if not reconstruction.empty:
        lines += [
            "",
            "## Held-out and cross-omics reconstruction (Huber loss; lower is better)",
            "",
            markdown_table(reconstruction.round(4)),
        ]
    if not stability.empty:
        lines += [
            "",
            "## Seed stability (linear CKA; ARI of consensus partitions)",
            "",
            markdown_table(stability.round(3)),
        ]
    for heading, values in (extra or {}).items():
        lines += ["", f"## {heading}", ""]
        lines += [f"- {key}: {value:.4g}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as a GitHub Markdown table without extra dependencies."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


__all__ = [
    "COHORT_CONTEXTS",
    "READOUTS",
    "CohortContext",
    "axes_table_for",
    "evaluate_representations",
    "load_encoder_run",
    "markdown_table",
    "metabric_context",
    "proliferation_score",
    "reconstruction_table",
    "run_copy_number_scale_check",
    "run_representation_evaluation",
    "seed_stability",
    "standardised_inputs",
    "tcga_context",
]
