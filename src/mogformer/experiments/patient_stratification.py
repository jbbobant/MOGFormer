"""Patient stratification under the pre-registered protocol.

The protocol (``configs/patient_stratification/README.md``) fixes the
encoders, populations, clustering procedure and bar before anything is
clustered. This module carries it out, structure only: no outcome column is
read here. Outcomes are analysed separately, under the locked outcome protocol.

For each population, every representation is reduced the same way
(standardised, then the principal components explaining 90% of the variance)
and clustered by consensus at K = 2…6 with k-means and a Gaussian mixture. A
stratification exists only if one (K, algorithm) pair is stable under
resampling (PAC), under bootstrap refits, and across the encoder's three seeds.
The representation ladder asks whether simpler inputs give the same split, and
the audit asks whether the split restates a known axis.
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
import yaml
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

from mogformer.analysis.clustering import (
    block_separation,
    bootstrap_agreement,
    fit_partition,
    partition_audit,
    reduce_for_clustering,
    run_consensus_clustering,
)
from mogformer.analysis.plots import (
    plot_confound_screen,
    plot_partition_crosstab,
    plot_stability_by_k,
    plot_strata_composition,
)
from mogformer.analysis.representation_evaluation import mofa_factors
from mogformer.data.omics import MODALITY_ORDER
from mogformer.experiments.representation_evaluation import (
    CohortContext,
    load_encoder_run,
    metabric_context,
    tcga_context,
)

logger = logging.getLogger(__name__)

#: Label of patients without a PAM50 call in the composition figure.
NOT_LABELLED = "not labelled"

#: Representation holding the encoder's primary partition.
ENCODER = "encoder"

#: Width of one program block in the encoder's embedding.
PROGRAM_BLOCK_WIDTH = 32

#: Role of each known axis in the novelty audit.
AXIS_ROLES: dict[str, str] = {
    "PAM50": "biology",
    "Integrative cluster": "biology",
    "ER status": "biology",
    "HER2 status": "biology",
    "Proliferation": "biology",
    "ESR1 expression": "biology",
    "CN burden": "biology",
    "Ploidy": "biology",
    "Age": "clinical",
    "NPI": "clinical",
    "Stage": "clinical",
    "Cohort (processing batch)": "technical",
    "Tissue source site": "technical",
}


@dataclass(frozen=True)
class StratificationProtocol:
    """Settings of the pre-registered protocol, read from ``protocol.yaml``.

    Attributes:
        cohorts: Per cohort: ``bundle``, ``runs`` (first seed primary) and
            ``populations``.
        discovery_population: Population of the outcome test.
        replication_population: Population that replicates its strata.
        readout: Encoder read-out clustered.
        n_clusters: Cluster counts tried.
        algorithms: Clustering algorithms tried.
        n_resample: Consensus resamples.
        subsample_frac: Share of patients per resample.
        n_bootstrap: Bootstrap refits for the bootstrap ARI.
        variance_explained: Variance the kept components must explain.
        max_components: Upper bound on kept components.
        bar: Thresholds ``pac``, ``bootstrap_ari`` and ``seed_ari``.
        burden_reproduces_ari: ARI above which a burden split is said to
            reproduce the encoder's partition.
        seed: Seed of every random step.
    """

    cohorts: dict[str, dict]
    discovery_population: str
    replication_population: str
    readout: str
    n_clusters: tuple[int, ...]
    algorithms: tuple[str, ...]
    n_resample: int
    subsample_frac: float
    n_bootstrap: int
    variance_explained: float
    max_components: int
    bar: dict[str, float]
    burden_reproduces_ari: float
    seed: int


def load_protocol(path: str | Path) -> StratificationProtocol:
    """Read the protocol's settings.

    Args:
        path: The ``protocol.yaml`` file.

    Returns:
        The settings.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw["n_clusters"] = tuple(raw["n_clusters"])
    raw["algorithms"] = tuple(raw["algorithms"])
    return StratificationProtocol(**raw)


@dataclass
class CohortInputs:
    """Everything one cohort contributes, aligned to its patients.

    Attributes:
        name: Cohort key.
        patients: Patient barcodes.
        embeddings: Encoder read-out per run, first run primary.
        inputs: Standardised encoder inputs, patients x genes x omics.
        genes: Gene axis of ``inputs``.
        context: Covariates, known axes and populations.
        axes: Known axes for the audit, one column each.
        comparators: Representations of the ladder, whole cohort.
    """

    name: str
    patients: list[str]
    embeddings: list[np.ndarray]
    inputs: np.ndarray
    genes: list[str]
    context: CohortContext
    axes: pd.DataFrame
    comparators: dict[str, np.ndarray]


def _known_axes(
    cohort: str,
    context: CohortContext,
    inputs: np.ndarray,
    genes: list[str],
    bundle: Path,
) -> pd.DataFrame:
    """Collect the audit's known axes: the probe targets plus cohort extras."""
    axes = pd.DataFrame({t.name: t.values.to_numpy() for t in context.targets})
    if "ESR1" in genes:
        axes["ESR1 expression"] = inputs[:, genes.index("ESR1"), 0]
    if cohort == "metabric":
        clinical = pd.read_csv(bundle / "clinical.csv", index_col=0)
        axes["NPI"] = clinical["NPI"].reindex(context.patients).to_numpy(dtype=float)
    return axes


def load_cohort_inputs(
    cohort: str, settings: Mapping, results_root: Path, seed: int
) -> CohortInputs:
    """Load a cohort's encoder runs, inputs, known axes and comparators.

    Comparators are fitted on the whole cohort without labels, as the encoder
    was, then restricted to each population.

    Args:
        cohort: ``"metabric"`` or ``"tcga"``.
        settings: The cohort's protocol entry.
        results_root: Directory holding the pretraining runs.
        seed: Seed of MOFA+.

    Returns:
        The cohort's inputs.

    Raises:
        ValueError: If the runs list different patients.
    """
    bundle = Path(settings["bundle"])
    embeddings, patients, inputs = [], None, None
    for run in settings["runs"]:
        readouts, run_inputs, run_patients = load_encoder_run(results_root / run)
        if patients is None:
            patients, inputs = run_patients, run_inputs
        elif run_patients != patients:
            raise ValueError(f"{run} covers different patients from the first run")
        embeddings.append(readouts["summary"])
    assert patients is not None and inputs is not None
    probe = torch.load(
        results_root / settings["runs"][0] / "probe_inputs.pt", weights_only=False
    )
    genes = [str(g) for g in probe["gene_order"]]
    context = (tcga_context if cohort == "tcga" else metabric_context)(
        patients, bundle, seed
    )
    flat = inputs.reshape(len(patients), -1)
    comparators = {
        "pca_inputs": flat,
        "mofa": mofa_factors(inputs, 32, seed=seed),
        **{f"pca_{omics}": inputs[:, :, k] for k, omics in enumerate(MODALITY_ORDER)},
        "burden": context.covariates[["burden"]].to_numpy(dtype=float),
    }
    if "ploidy" in context.covariates:
        comparators["ploidy"] = context.covariates[["ploidy"]].to_numpy(dtype=float)
    axes = _known_axes(cohort, context, inputs, genes, bundle)
    return CohortInputs(
        cohort, patients, embeddings, inputs, genes, context, axes, comparators
    )


def _cluster_representation(
    name: str,
    values: np.ndarray,
    algorithm: str,
    protocol: StratificationProtocol,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Consensus-cluster one reduced representation with one algorithm."""
    results = run_consensus_clustering(
        values,
        n_clusters_list=protocol.n_clusters,
        n_resample=protocol.n_resample,
        subsample_frac=protocol.subsample_frac,
        seed=protocol.seed,
        algorithm=algorithm,
    )
    rows, labels = [], {}
    for k, result in results.items():
        rows.append(
            {
                "representation": name,
                "algorithm": algorithm,
                "k": k,
                "pac": result.pac,
                "bootstrap_ari": bootstrap_agreement(
                    values,
                    result.labels,
                    k,
                    algorithm,
                    n_boot=protocol.n_bootstrap,
                    seed=protocol.seed,
                ),
                "smallest_cluster": int(np.bincount(result.labels).min()),
            }
        )
        labels[k] = result.labels
    return rows, labels


def stability_table(
    representations: Mapping[str, np.ndarray],
    seed_embeddings: Sequence[np.ndarray],
    protocol: StratificationProtocol,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], np.ndarray]]:
    """Cluster every representation and score each (K, algorithm) pair.

    Args:
        representations: Name to data for one population; ``"encoder"`` is
            the primary seed's embedding.
        seed_embeddings: The other seeds' embeddings of the same population.
        protocol: The settings.
        n_jobs: Parallel workers over (representation, algorithm) pairs.

    Returns:
        The stability table (PAC, bootstrap ARI, seed ARI for the encoder,
        whether the bar is met) and the labels by (representation,
        algorithm, K).
    """
    reduced = {
        name: reduce_for_clustering(
            values, protocol.variance_explained, protocol.max_components
        )
        for name, values in representations.items()
    }
    tasks = [(name, algorithm) for name in reduced for algorithm in protocol.algorithms]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_cluster_representation)(name, reduced[name], algorithm, protocol)
        for name, algorithm in tasks
    )
    rows, labels = [], {}
    for (name, algorithm), (task_rows, task_labels) in zip(tasks, outputs, strict=True):
        rows += task_rows
        for k, value in task_labels.items():
            labels[(name, algorithm, k)] = value

    others = [
        reduce_for_clustering(e, protocol.variance_explained, protocol.max_components)
        for e in seed_embeddings
    ]
    table = pd.DataFrame(rows)
    seed_ari = []
    for representation, algorithm, k in zip(
        table["representation"].astype(str),
        table["algorithm"].astype(str),
        table["k"].astype(int),
        strict=True,
    ):
        if representation != ENCODER:
            seed_ari.append(float("nan"))
            continue
        primary = labels[(ENCODER, algorithm, k)]
        seed_ari.append(
            min(
                adjusted_rand_score(
                    primary, fit_partition(o, k, algorithm, seed=protocol.seed)
                )
                for o in others
            )
        )
    table["seed_ari"] = seed_ari
    bar = protocol.bar
    table["meets_bar"] = (
        (table["pac"] <= bar["pac"])
        & (table["bootstrap_ari"] >= bar["bootstrap_ari"])
        & (
            (table["seed_ari"] >= bar["seed_ari"])
            | (table["representation"] != ENCODER)
        )
    )
    return table, labels


def choose_partition(table: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Pick the encoder's reported partition under the protocol's rule.

    Among encoder rows meeting the bar, the lowest PAC wins; ties go to the
    smaller K, then to k-means. If none meets it, the lowest-PAC candidate is
    returned as the best available partition.

    Args:
        table: The stability table of one population.

    Returns:
        The chosen row and whether it meets the bar.
    """
    encoder = table[table["representation"] == ENCODER].copy()
    encoder["algorithm_order"] = (encoder["algorithm"] != "kmeans").astype(int)
    passing = encoder[encoder["meets_bar"]]
    pool = passing if len(passing) else encoder
    row = pool.sort_values(["pac", "k", "algorithm_order"]).iloc[0]
    return row, bool(len(passing))


def differential_features(
    inputs: np.ndarray, genes: Sequence[str], labels: np.ndarray, top: int = 25
) -> pd.DataFrame:
    """Rank each omics' genes by how they separate each stratum from the rest.

    Args:
        inputs: Standardised inputs of the population, patients x genes x
            omics.
        genes: Gene names.
        labels: Stratum per patient.
        top: Genes kept per stratum and omics, by significance.

    Returns:
        One row per kept (stratum, omics, gene): the mean difference (in
        standard units) and Benjamini–Hochberg-adjusted Mann–Whitney p-value.
    """
    rows = []
    for stratum in np.unique(labels):
        inside = labels == stratum
        for k, omics in enumerate(MODALITY_ORDER):
            values = inputs[:, :, k]
            p = mannwhitneyu(values[inside], values[~inside], axis=0).pvalue
            order = np.argsort(p)
            ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
            adjusted = np.empty_like(p)
            adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
            difference = values[inside].mean(axis=0) - values[~inside].mean(axis=0)
            for g in order[:top]:
                rows.append(
                    {
                        "stratum": int(stratum),
                        "omics": omics,
                        "gene": genes[g],
                        "difference": float(difference[g]),
                        "p_bh": float(adjusted[g]),
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Replication:
    """Discovery strata carried to the replication population.

    Attributes:
        shared_genes: Genes on both encoders' axes, the classifier's inputs.
        ari_assigned_vs_own: Agreement between the assigned strata and the
            replication population's own clustering.
        assigned: Assigned discovery stratum per replication patient.
        crosstab: Assigned strata against the own clustering.
    """

    shared_genes: int
    ari_assigned_vs_own: float
    assigned: np.ndarray
    crosstab: pd.DataFrame


def replicate_strata(
    discovery: CohortInputs,
    discovery_rows: np.ndarray,
    discovery_labels: np.ndarray,
    replication: CohortInputs,
    replication_rows: np.ndarray,
    replication_labels: np.ndarray,
    seed: int,
) -> Replication:
    """Carry the discovery strata to the replication population.

    A multinomial logistic model is trained on the discovery population's
    standardised inputs over the genes both encoders share, then assigns the
    replication patients, whose agreement with their own clustering is the
    replication score.

    Args:
        discovery: The discovery cohort.
        discovery_rows: Rows of the discovery population.
        discovery_labels: Its strata.
        replication: The replication cohort.
        replication_rows: Rows of the replication population.
        replication_labels: Its own partition at the same K and algorithm.
        seed: Seed of the classifier.

    Returns:
        The replication: shared genes, agreement, assigned strata and their
        cross-tabulation with the own strata.
    """
    shared = [g for g in discovery.genes if g in set(replication.genes)]
    d_index = [discovery.genes.index(g) for g in shared]
    r_index = [replication.genes.index(g) for g in shared]
    x_train = discovery.inputs[discovery_rows][:, d_index, :].reshape(
        len(discovery_rows), -1
    )
    x_test = replication.inputs[replication_rows][:, r_index, :].reshape(
        len(replication_rows), -1
    )
    model = LogisticRegression(C=0.05, max_iter=5000, random_state=seed)
    assigned = model.fit(x_train, discovery_labels).predict(x_test)
    return Replication(
        shared_genes=len(shared),
        ari_assigned_vs_own=float(adjusted_rand_score(replication_labels, assigned)),
        assigned=np.asarray(assigned),
        crosstab=pd.crosstab(
            pd.Series(assigned, name="assigned discovery stratum"),
            pd.Series(replication_labels, name="own clustering"),
        ),
    )


def run_patient_stratification(
    protocol_path: str | Path,
    results_root: str | Path,
    output_dir: str | Path,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Carry out the stratification protocol, structure only.

    Args:
        protocol_path: The ``protocol.yaml`` file.
        results_root: Directory holding the pretraining runs.
        output_dir: Where the outputs are written.
        n_jobs: Parallel workers for the consensus clustering.

    Returns:
        The record: per population the reported partition, whether it meets
        the bar, the ladder agreement, the best audited axis, and the
        replication score.
    """
    protocol = load_protocol(protocol_path)
    results_root, output = Path(results_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cohorts = {
        name: load_cohort_inputs(name, settings, results_root, protocol.seed)
        for name, settings in protocol.cohorts.items()
    }
    tables: list[pd.DataFrame] = []
    audits: dict[str, pd.DataFrame] = {}
    compositions: dict[str, pd.DataFrame] = {}
    record: dict[str, object] = {}
    chosen: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]] = {}
    differential, blocks, strata = [], [], []
    for cohort_name, cohort in cohorts.items():
        for population in protocol.cohorts[cohort_name]["populations"]:
            rows = cohort.context.populations[population]
            logger.info("%s: %d patients", population, len(rows))
            representations = {ENCODER: cohort.embeddings[0][rows]}
            representations.update(
                {name: values[rows] for name, values in cohort.comparators.items()}
            )
            table, labels = stability_table(
                representations,
                [e[rows] for e in cohort.embeddings[1:]],
                protocol,
                n_jobs=n_jobs,
            )
            sensitivity = run_consensus_clustering(
                cohort.embeddings[0][rows],
                n_clusters_list=protocol.n_clusters,
                n_resample=protocol.n_resample,
                seed=protocol.seed,
            )
            table["raw_embedding_kmeans_pac"] = [
                sensitivity[k].pac
                if representation == ENCODER and algorithm == "kmeans"
                else float("nan")
                for representation, algorithm, k in zip(
                    table["representation"].astype(str),
                    table["algorithm"].astype(str),
                    table["k"].astype(int),
                    strict=True,
                )
            ]
            table.insert(0, "population", population)
            tables.append(table)

            row, meets = choose_partition(table)
            k, algorithm = int(row["k"]), str(row["algorithm"])
            partition = labels[(ENCODER, algorithm, k)]
            ladder = {
                name: float(
                    adjusted_rand_score(partition, labels[(name, algorithm, k)])
                )
                for name in representations
                if name != ENCODER
            }
            audit = partition_audit(
                partition, cohort.axes.iloc[rows].reset_index(drop=True), AXIS_ROLES
            )
            audit.insert(0, "population", population)
            audits[population] = audit
            pam50 = (
                cohort.axes.iloc[rows]["PAM50"]
                .astype(object)
                .fillna(NOT_LABELLED)
                .astype(str)
                .to_numpy()
            )
            compositions[population] = pd.crosstab(
                pd.Series(partition, name="stratum"), pd.Series(pam50, name="PAM50")
            )
            features = differential_features(
                cohort.inputs[rows], cohort.genes, partition
            )
            features.insert(0, "population", population)
            differential.append(features)
            separation = block_separation(
                cohort.embeddings[0][rows], partition, PROGRAM_BLOCK_WIDTH
            )
            separation.insert(0, "population", population)
            blocks.append(separation)
            strata.append(
                pd.DataFrame(
                    {
                        "population": population,
                        "patient": np.asarray(cohort.patients)[rows],
                        "stratum": partition,
                    }
                )
            )
            chosen[population] = (rows, partition, row)
            best_axis = audit.iloc[0]
            burden_agreement = max(ladder.get("burden", 0.0), ladder.get("ploidy", 0.0))
            record[population] = {
                "n_patients": len(rows),
                "k": k,
                "algorithm": algorithm,
                "pac": float(row["pac"]),
                "bootstrap_ari": float(row["bootstrap_ari"]),
                "seed_ari": float(row["seed_ari"]),
                "meets_bar": meets,
                "stratum_sizes": np.bincount(partition).tolist(),
                "ladder_ari": ladder,
                "reproduced_by_burden_or_ploidy": burden_agreement
                >= protocol.burden_reproduces_ari,
                "best_known_axis": str(best_axis["axis"]),
                "best_known_axis_effect": float(best_axis["effect"]),
            }
            logger.info(
                "%s: K=%d %s, PAC %.3f, bootstrap ARI %.2f, seed ARI %.2f, bar %s",
                population,
                k,
                algorithm,
                row["pac"],
                row["bootstrap_ari"],
                row["seed_ari"],
                "met" if meets else "not met",
            )

    discovery_rows, discovery_labels, discovery_row = chosen[
        protocol.discovery_population
    ]
    replication_rows, _, _ = chosen[protocol.replication_population]
    replication_cohort = next(
        c
        for c in cohorts.values()
        if protocol.replication_population in c.context.populations
    )
    discovery_cohort = next(
        c
        for c in cohorts.values()
        if protocol.discovery_population in c.context.populations
    )
    own = fit_partition(
        reduce_for_clustering(
            replication_cohort.embeddings[0][replication_rows],
            protocol.variance_explained,
            protocol.max_components,
        ),
        int(discovery_row["k"]),
        str(discovery_row["algorithm"]),
        seed=protocol.seed,
    )
    replication = replicate_strata(
        discovery_cohort,
        discovery_rows,
        discovery_labels,
        replication_cohort,
        replication_rows,
        own,
        protocol.seed,
    )
    record["replication"] = {
        "population": protocol.replication_population,
        "shared_genes": replication.shared_genes,
        "ari_assigned_vs_own": replication.ari_assigned_vs_own,
        "assigned_sizes": np.bincount(replication.assigned).tolist(),
    }

    stability = pd.concat(tables, ignore_index=True)
    stability.to_csv(output / "stability_by_k.csv", index=False)
    pd.concat(audits.values(), ignore_index=True).to_csv(
        output / "strata_versus_known_axes.csv", index=False
    )
    pd.concat(differential, ignore_index=True).to_csv(
        output / "differential_features.csv", index=False
    )
    pd.concat(blocks, ignore_index=True).to_csv(
        output / "block_separation.csv", index=False
    )
    pd.concat(strata, ignore_index=True).to_csv(output / "strata.csv", index=False)
    replication.crosstab.to_csv(output / "tcga_replication.csv")
    plot_stability_by_k(stability, output, bar=protocol.bar)
    plot_confound_screen(
        dict(audits),
        output,
        name="strata_versus_known_axes",
        label_col="axis",
        effect_col="effect",
        xlabel="AMI (categorical) · η² (continuous)",
    )
    plot_strata_composition(compositions, output)
    plot_partition_crosstab(
        replication.crosstab,
        output,
        name="tcga_replication",
        title=f"{protocol.replication_population}  ·  "
        f"ARI {replication.ari_assigned_vs_own:.2f}",
    )
    (output / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
