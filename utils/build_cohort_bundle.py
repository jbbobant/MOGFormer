"""Assemble the compact cohort bundle every experiment reads.

The raw inputs are around 1.6 GB of CSV, most of which is never used: STRING
ships the whole human interactome and only the induced subgraph over this
cohort's genes matters, and re-parsing and re-aligning the omics matrices
dominates the start-up of every run.

This tool reduces those inputs to what a run actually reads, without moving
the leakage boundary. Every reduction here is cohort-level and fold-blind:
patient and gene alignment, the interactome restricted to the gene universe,
the frozen embedding table. Everything that must be fit per fold — variable
gene selection, imputation, scaling, the induced graph — is untouched and
still happens inside the fold, at run time.

The bundle is written under ``data/bundle/``, which is git-ignored, and is
where ``mogformer.cli.experiments`` looks by default.

METABRIC is bundled once per classification target, each in its own
directory with its own stratified partition: the targets label different
patients, and a partition stratified on one label is not stratified on another.

Usage:
    uv run python -m utils.build_cohort_bundle --cohort tcga
    uv run python -m utils.build_cohort_bundle --cohort metabric
    uv run python -m utils.build_cohort_bundle --cohort all
    uv run python -m utils.build_cohort_bundle --cohort tcga_ssl
    uv run python -m utils.build_cohort_bundle --cohort metabric_ssl

The self-supervised cohorts are bundled separately (``cohort_ssl.npz``): their
membership is defined by which omics were measured, not by a label, and their
copy number is kept on its native scale (absolute copies in TCGA, ordinal codes
in METABRIC) so that every copy-number representation can be derived from the
one archive at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mogformer.data.folds import make_folds, save_folds
from mogformer.data.gene_embeddings import (
    load_gene_embedding_table,
    save_gene_embedding_table,
)
from mogformer.data.omics import (
    M_VALUE_EPS,
    beta_to_m_values,
    load_omics,
    ploidy_estimate,
    save_omics_bundle,
)
from mogformer.data.receptors import resolve_receptor_status
from mogformer.graph.string_graph import (
    EVIDENCE_CHANNELS,
    StringGraphCache,
    parse_string_channel_edges,
    save_string_edges,
)

logger = logging.getLogger("build_cohort_bundle")

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where bundles are written. Everything under it is reproducible from the
#: repository plus the raw data, so none of it is committed.
DATA_ROOT = REPO_ROOT / "data" / "bundle"

#: Files copied verbatim because they are already small.
SMALL_INPUTS: tuple[tuple[str, str], ...] = (
    ("data/clean/folds.json", "folds.json"),
    ("data/clean/collectri_signed_edges.tsv", "collectri_signed_edges.tsv"),
    ("data/raw/CGenes.txt", "curated_genes.txt"),
    ("data/raw/PAM50.txt", "pam50_genes.txt"),
    ("data/clean/survival_features_common.csv", "survival_features.csv"),
    ("data/clean/data_clinical.csv", "clinical.csv"),
)


#: GDC follow-up table holding the TCGA-BRCA receptor tests, and the TCGA 2012
#: clinical table that fills its gaps.
RECEPTOR_FOLLOW_UP = "data/download/clinical.project-tcga-brca.2026-10-03/follow_up.tsv"
RECEPTOR_LEGACY = "data/download/data_clinical_sample.txt"


@dataclass(frozen=True)
class CohortSource:
    """One methylation representation of one cohort.

    Two representations exist because the archived ablation used both, and the
    difference is a data choice rather than a model setting: beta values are
    bounded proportions, M-values their logit. Each therefore produces its own
    aligned matrix and must be bundled separately.

    Attributes:
        key: Name of the bundle file, without extension.
        rna_file: Expression matrix, relative to the repository root.
        cnv_file: Copy-number matrix.
        methy_file: Methylation matrix.
        clin_file: Clinical table carrying the label column.
        label_col: Column holding the subtype label.
        keep_unlabelled: Keep patients without a label, for label-free cohorts.
    """

    key: str
    rna_file: str
    cnv_file: str
    methy_file: str
    clin_file: str
    label_col: str = "SUBTYPE"
    keep_unlabelled: bool = False


#: The TCGA methylation representations the transformer variants ask for.
TCGA_SOURCES: tuple[CohortSource, ...] = (
    CohortSource(
        key="cohort_beta",
        rna_file="data/raw/data_rna_seq_v2_rsem.csv",
        cnv_file="data/raw/data_cnv.csv",
        methy_file="data/raw/data_methylation450.csv",
        clin_file="data/clean/data_clinical.csv",
    ),
    CohortSource(
        key="cohort_m",
        rna_file="data/raw/data_rna_seq_v2_rsem.csv",
        cnv_file="data/raw/data_cnv.csv",
        methy_file="data/raw/data_methylation_M.csv",
        clin_file="data/clean/data_clinical.csv",
    ),
)

#: METABRIC promoter methylation as released (gene-level beta values), and the
#: M-values derived from it with the transform TCGA's M-values were made with.
METABRIC_BETA_FILE = "data/METABRIC/clean/gene_promoter_methylation_common_beta.csv"
METABRIC_M_FILE = "data/METABRIC/clean/gene_promoter_methylation_common_M.csv"

#: METABRIC omics, aligned the same way as TCGA's, with methylation as
#: M-values so that both cohorts use one representation.
METABRIC_SOURCE = CohortSource(
    key="cohort_m",
    rna_file="data/METABRIC/clean/data_rna_common.csv",
    cnv_file="data/METABRIC/clean/data_cnv_common.csv",
    methy_file=METABRIC_M_FILE,
    clin_file="data/METABRIC/clean/data_clinical.csv",
)

#: METABRIC classification targets: bundle directory and label column. They
#: differ in what the label is derived from — expression centroids (PAM50),
#: copy number with expression (integrative clusters), copy number alone (HER2
#: SNP6 calls) and protein staining (ER immunohistochemistry).
METABRIC_TARGETS: Mapping[str, str] = {
    "pam50": "SUBTYPE",
    "intclust": "INTCLUST",
    "her2_snp6": "HER2_SNP6",
    "er_ihc": "ER_IHC",
}

#: Small files each METABRIC target directory carries.
METABRIC_SMALL_INPUTS: tuple[tuple[str, str], ...] = (
    ("data/clean/collectri_signed_edges.tsv", "collectri_signed_edges.tsv"),
    ("data/raw/CGenes.txt", "curated_genes.txt"),
    ("data/raw/PAM50.txt", "pam50_genes.txt"),
    ("data/METABRIC/clean/data_clinical.csv", "clinical.csv"),
)

#: TCGA self-supervised cohort: every patient with gene-level absolute copy
#: number, expression and promoter methylation (M-values), labelled or not.
TCGA_SSL_SOURCE = CohortSource(
    key="cohort_ssl",
    rna_file="data/clean/data_rna_common.csv",
    cnv_file="data/clean/data_cnv_common.csv",
    methy_file="data/clean/gene_promoter_methylation_common_meta.csv",
    clin_file="data/clean/data_clinical.csv",
    keep_unlabelled=True,
)
TCGA_SSL_COHORT_RULE = (
    "every TCGA-BRCA patient with gene-level absolute copy number "
    "(data/clean/data_cnv_common.csv), RNA-seq expression and promoter "
    "methylation; no label required"
)

#: METABRIC self-supervised cohort: every patient with all three omics.
METABRIC_SSL_SOURCE = CohortSource(
    key="cohort_ssl",
    rna_file="data/METABRIC/clean/data_rna_common.csv",
    cnv_file="data/METABRIC/clean/data_cnv_common.csv",
    methy_file=METABRIC_M_FILE,
    clin_file="data/METABRIC/clean/data_clinical.csv",
    keep_unlabelled=True,
)
METABRIC_SSL_COHORT_RULE = (
    "every METABRIC patient with expression, ordinal copy-number calls (0-4) "
    "and promoter methylation (M-values); no label required"
)

#: STRING's per-channel scores, from which the self-supervised prior is
#: rebuilt without the co-expression channel.
STRING_DETAILED_LINKS = "data/clean/9606.protein.links.detailed.v12.5.txt"

#: Name of the channel-filtered, confidence-weighted edge list.
EVIDENCE_EDGES_FILE = "string_edges_experimental_database.tsv"

#: Partition settings, identical to the TCGA partition's.
N_SPLITS = 5
N_REPEATS = 5
FOLD_SEED = 42

#: STRING inputs, parsed once over the union of every cohort's gene universe.
STRING_LINKS = "data/clean/9606.protein.links.v12.0.txt"
STRING_ALIASES = "data/clean/9606.protein.aliases.v12.0.txt"
STRING_THRESHOLD = 400

#: Pretrained gene vectors, for the variants that use a frozen identity.
EMBEDDING_MATRIX = "data/raw/FROGS-ARCHS4/FROGS-ARCHS4_emb.csv"
EMBEDDING_SYMBOLS = "data/raw/FROGS-ARCHS4/FROGS-ARCHS4_symbollist.txt"

#: Archived self-supervised runs the trans probe re-measures from. Only the
#: three files the probe reads are copied; the rest of each run directory is
#: figures and parquet no run reads.
ENCODER_SOURCES: Mapping[str, str] = {
    "tcga": "results/SSL/SURV_CLEAN/TCGA/E600_p60_gated_meta",
    "metabric": "results/SSL/SURV_CLEAN/METABRIC/E800_p100_gated_meta",
}

#: What an archived run must carry for :func:`load_archived_encoder` to work.
ENCODER_FILES: tuple[str, ...] = (
    "encoder_frozen.pt",
    "graph_tensors.pt",
    "probe_inputs.pt",
    "gene_universe.json",
    "provenance.json",
)


def digest(path: Path, block_size: int = 1 << 20) -> str:
    """Return a short content hash, so a bundle can be told apart from a copy.

    Args:
        path: File to hash.
        block_size: Bytes read per iteration.

    Returns:
        The first sixteen hexadecimal characters of the SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def build_cohorts(
    sources: Sequence[CohortSource], destination: Path
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Align each representation and write it as a bundle.

    Args:
        sources: Representations to build.
        destination: Directory the bundles are written into.

    Returns:
        Tuple of per-bundle metadata and the union of every gene universe,
        which is what the interactome must be reduced over.
    """
    metadata: dict[str, dict[str, object]] = {}
    universe: set[str] = set()

    for source in sources:
        matrix = REPO_ROOT / source.methy_file
        if not matrix.exists():
            logger.warning("skipping %s: %s is absent", source.key, source.methy_file)
            continue

        logger.info("aligning %s (this reads ~0.5 GB of CSV)", source.key)
        data = load_omics(
            raw_dir=REPO_ROOT,
            rna_file=source.rna_file,
            cnv_file=source.cnv_file,
            methy_file=source.methy_file,
            clin_file=source.clin_file,
            label_col=source.label_col,
            exclude=(),
            keep_unlabelled=source.keep_unlabelled,
        )
        path = save_omics_bundle(data, destination / f"{source.key}.npz")
        universe |= set(data.gene_names)
        metadata[source.key] = {
            "file": path.name,
            "sha256_16": digest(path),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "n_patients": data.n_patients,
            "n_genes": data.n_genes,
            "classes": data.class_counts(),
            "methylation_source": source.methy_file,
        }
        logger.info(
            "%s: %d patients x %d genes -> %.1f MB",
            source.key,
            data.n_patients,
            data.n_genes,
            path.stat().st_size / 1e6,
        )

    return metadata, sorted(universe)


def build_graph(universe: Sequence[str], destination: Path) -> dict[str, object]:
    """Reduce the interactome to the cohort's gene universe.

    Args:
        universe: Every gene any representation of the cohort contains.
        destination: Directory the edge list is written into.

    Returns:
        Metadata describing the reduction.
    """
    logger.info("parsing STRING over %d genes (this reads ~0.8 GB)", len(universe))
    cache = StringGraphCache(
        str(REPO_ROOT / STRING_LINKS),
        str(REPO_ROOT / STRING_ALIASES),
        universe,
        STRING_THRESHOLD,
    )
    path = save_string_edges(cache.edges, destination / "string_edges.tsv")
    return {
        "file": path.name,
        "sha256_16": digest(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_edges": cache.n_universe_edges,
        "n_universe_genes": len(universe),
        "confidence_threshold": STRING_THRESHOLD,
        "source": [STRING_LINKS, STRING_ALIASES],
    }


def build_embeddings(destination: Path) -> dict[str, object] | None:
    """Compress the frozen gene vectors into a single archive.

    Args:
        destination: Directory the archive is written into.

    Returns:
        Metadata describing the table, or None if the source is absent.
    """
    matrix = REPO_ROOT / EMBEDDING_MATRIX
    symbols = REPO_ROOT / EMBEDDING_SYMBOLS
    if not matrix.exists() or not symbols.exists():
        logger.warning("skipping gene embeddings: %s is absent", EMBEDDING_MATRIX)
        return None

    table = load_gene_embedding_table(matrix, symbols)
    path = save_gene_embedding_table(table, destination / "gene_embeddings.npz")
    return {
        "file": path.name,
        "sha256_16": digest(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_genes": int(table.shape[0]),
        "d_embedding": int(table.shape[1]),
        "source": EMBEDDING_MATRIX,
    }


def copy_encoders(destination: Path) -> dict[str, dict[str, object]]:
    """Copy the archived encoders the trans probe re-measures from.

    The external-cohort sweep cannot be re-analysed from stored slopes the way
    TCGA's can, so the frozen encoder itself is kept. Seventeen megabytes for
    both cohorts, against retraining them — which would produce a second
    encoder and make the two cohorts incomparable.

    Args:
        destination: Directory the ``encoders/`` tree is written under.

    Returns:
        Metadata keyed by cohort name.
    """
    metadata: dict[str, dict[str, object]] = {}
    for cohort, relative in ENCODER_SOURCES.items():
        source = REPO_ROOT / relative
        if not source.is_dir():
            logger.warning("skipping %s encoder: %s is absent", cohort, relative)
            continue

        target = destination / "encoders" / cohort
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        copied: list[str] = []
        for name in ENCODER_FILES:
            candidate = source / name
            if candidate.exists():
                shutil.copy2(candidate, target / name)
                copied.append(name)
            elif name != "provenance.json":
                logger.warning("%s encoder is missing %s", cohort, name)

        size = sum((target / name).stat().st_size for name in copied)
        metadata[cohort] = {
            "directory": f"encoders/{cohort}",
            "files": copied,
            "size_mb": round(size / 1e6, 1),
            "source": relative,
        }
        logger.info("%s encoder: %d files, %.1f MB", cohort, len(copied), size / 1e6)
    return metadata


def copy_small_inputs(
    destination: Path, inputs: Sequence[tuple[str, str]] = SMALL_INPUTS
) -> dict[str, dict[str, object]]:
    """Copy the inputs that are already small enough to ship verbatim.

    Args:
        destination: Directory the copies are written into.
        inputs: ``(source, destination name)`` pairs.

    Returns:
        Metadata keyed by destination filename.
    """
    metadata: dict[str, dict[str, object]] = {}
    for relative, name in inputs:
        source = REPO_ROOT / relative
        if not source.exists():
            logger.warning("skipping %s: absent", relative)
            continue
        target = destination / name
        shutil.copy2(source, target)
        metadata[name] = {
            "sha256_16": digest(target),
            "size_kb": round(target.stat().st_size / 1e3, 1),
            "source": relative,
        }
    return metadata


def build_receptors(destination: Path) -> dict[str, object] | None:
    """Write the per-patient ER, PR and HER2 table beside the TCGA cohorts.

    Args:
        destination: The TCGA bundle directory.

    Returns:
        The table's manifest entry, or None when the GDC follow-up table has
        not been downloaded.
    """
    follow_up = REPO_ROOT / RECEPTOR_FOLLOW_UP
    if not follow_up.exists():
        return None
    legacy = REPO_ROOT / RECEPTOR_LEGACY
    table = resolve_receptor_status(follow_up, legacy if legacy.exists() else None)
    target = destination / "receptors.csv"
    table.to_csv(target)
    return {
        "file": target.name,
        "sha256_16": digest(target),
        "sources": [RECEPTOR_FOLLOW_UP]
        + ([RECEPTOR_LEGACY] if legacy.exists() else []),
        "n_patients": len(table),
        "er_positive_her2_negative": int(table["er_positive_her2_negative"].sum()),
    }


def build_copy_number_covariates(cnv_file: str, destination: Path) -> dict[str, object]:
    """Write each patient's ploidy and copy-number burden beside the cohort.

    Both are read over every gene of the absolute copy-number matrix, never
    over a selected subset, and neither is a model input: ploidy feeds the
    ``ploidy_relative`` scale and both serve as covariates of the evaluation.

    Args:
        cnv_file: Absolute copy number, genes by patients, relative to the
            repository root.
        destination: Directory ``ploidy.csv`` is written into.

    Returns:
        The table's manifest entry.
    """
    cnv = pd.read_csv(REPO_ROOT / cnv_file, index_col=0)
    cnv = cnv[cnv.index.notna()]
    table = pd.DataFrame(
        {
            "ploidy": ploidy_estimate(cnv),
            # Fraction of genes whose rounded copy number is not two.
            "burden": (cnv.round() != 2).where(cnv.notna()).mean(axis=0),
        }
    )
    table.index.name = "patient_id"
    target = destination / "ploidy.csv"
    table.to_csv(target)
    return {
        "file": target.name,
        "sha256_16": digest(target),
        "source": cnv_file,
        "n_genes": int(cnv.shape[0]),
        "n_patients": int(cnv.shape[1]),
        "ploidy": "mean absolute copy number over every gene of the source",
        "burden": "fraction of genes whose rounded absolute copy number is not 2",
        "ploidy_range": [
            round(float(table["ploidy"].min()), 2),
            round(float(table["ploidy"].max()), 2),
        ],
    }


def build_evidence_graph(
    universe: Sequence[str], destination: Path
) -> dict[str, object]:
    """Rebuild STRING edges from experimental and database evidence only.

    Args:
        universe: Gene universe of the cohort.
        destination: Directory the edge list is written into.

    Returns:
        Metadata describing the reduction.
    """
    logger.info("parsing STRING channels over %d genes", len(universe))
    weights = parse_string_channel_edges(
        str(REPO_ROOT / STRING_DETAILED_LINKS),
        str(REPO_ROOT / STRING_ALIASES),
        universe,
        channels=EVIDENCE_CHANNELS,
        confidence_threshold=STRING_THRESHOLD,
    )
    path = save_string_edges(weights, destination / EVIDENCE_EDGES_FILE, weights)
    return {
        "file": path.name,
        "sha256_16": digest(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_edges": len(weights),
        "n_universe_genes": len(universe),
        "n_genes_with_an_edge": len({gene for edge in weights for gene in edge}),
        "channels": list(EVIDENCE_CHANNELS),
        "score": "STRING combination of the listed channels, scaled to (0, 1]",
        "confidence_threshold": STRING_THRESHOLD,
        "source": [STRING_DETAILED_LINKS, STRING_ALIASES],
    }


def build_ssl_cohort(
    source: CohortSource, cohort_rule: str, destination: Path
) -> dict[str, object]:
    """Bundle a self-supervised cohort and its evidence-only interactome.

    Args:
        source: Where the cohort's matrices are.
        cohort_rule: Which patients the cohort contains, in words.
        destination: Directory the archive and edge list are written into.

    Returns:
        The ``ssl`` section of the bundle manifest.

    Raises:
        FileNotFoundError: If the cohort's matrices are absent.
    """
    destination.mkdir(parents=True, exist_ok=True)
    cohorts, universe = build_cohorts([source], destination)
    if not cohorts:
        raise FileNotFoundError(f"the {source.key} matrices are absent")
    return {
        "cohort": {
            **cohorts[source.key],
            "cohort_rule": cohort_rule,
            "sources": [source.rna_file, source.cnv_file, source.methy_file],
            "copy_number_scale": "as distributed; representation applied at load",
        },
        "graph": build_evidence_graph(universe, destination),
    }


def _update_manifest(destination: Path, key: str, section: object) -> None:
    """Add or replace one section of a bundle's manifest."""
    path = destination / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest[key] = section
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_tcga_ssl(destination: Path) -> dict[str, object]:
    """Add the TCGA self-supervised cohort to the TCGA bundle.

    Args:
        destination: The TCGA bundle directory.

    Returns:
        A summary in the shape :func:`report` reads.
    """
    section = build_ssl_cohort(TCGA_SSL_SOURCE, TCGA_SSL_COHORT_RULE, destination)
    section["ploidy"] = build_copy_number_covariates(
        TCGA_SSL_SOURCE.cnv_file, destination
    )
    _update_manifest(destination, "ssl", section)
    return {"cohorts": {"cohort_ssl": section["cohort"]}, "graph": section["graph"]}


def build_metabric_ssl(destination: Path) -> dict[str, object]:
    """Bundle the METABRIC self-supervised cohort as a bundle of its own.

    Args:
        destination: Directory to write into.

    Returns:
        The manifest written.
    """
    section = build_ssl_cohort(
        METABRIC_SSL_SOURCE, METABRIC_SSL_COHORT_RULE, destination
    )
    manifest = {
        "cohort": "METABRIC, self-supervised",
        "built_by": "utils/build_cohort_bundle.py",
        "cohorts": {"cohort_ssl": section["cohort"]},
        "graph": section["graph"],
        "files": copy_small_inputs(destination, METABRIC_SMALL_INPUTS),
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_tcga(destination: Path) -> dict[str, object]:
    """Build the whole TCGA bundle.

    Args:
        destination: Directory to write into.

    Returns:
        The manifest describing everything written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    cohorts, universe = build_cohorts(TCGA_SOURCES, destination)
    if not cohorts:
        raise FileNotFoundError("no TCGA representation could be built")

    manifest: dict[str, object] = {
        "cohort": "TCGA-BRCA",
        "built_by": "utils/build_cohort_bundle.py",
        "cohorts": cohorts,
        "graph": build_graph(universe, destination),
        "files": copy_small_inputs(destination),
        "encoders": copy_encoders(destination),
    }
    embeddings = build_embeddings(destination)
    if embeddings is not None:
        manifest["gene_embeddings"] = embeddings
    receptors = build_receptors(destination)
    if receptors is not None:
        manifest["receptors"] = receptors

    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def write_m_values(beta_file: str, m_file: str) -> dict[str, object]:
    """Derive an M-value matrix from a gene-level beta matrix.

    Args:
        beta_file: Genes-by-patients beta values, relative to the repository.
        m_file: Where the M-values are written, relative to the repository.

    Returns:
        Metadata describing the derived file.
    """
    beta = pd.read_csv(REPO_ROOT / beta_file, index_col=0)
    m_values = beta_to_m_values(beta)
    target = REPO_ROOT / m_file
    m_values.to_csv(target)
    logger.info(
        "wrote M-values for %d genes x %d patients to %s",
        m_values.shape[0],
        m_values.shape[1],
        m_file,
    )
    return {
        "file": m_file,
        "derived_from": beta_file,
        "transform": "log2(b / (1 - b))",
        "clip_eps": M_VALUE_EPS,
        "sha256_16": digest(target),
    }


def build_metabric(destination: Path) -> dict[str, object]:
    """Build one METABRIC bundle directory per classification target.

    Each directory has the standard layout — ``cohort_m.npz``,
    ``folds.json``, the interactome and the small inputs — so every experiment
    runs on it unchanged, pointed there with ``--bundle``.

    Args:
        destination: Directory the per-target directories are written under.

    Returns:
        The manifest describing everything written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    methylation = write_m_values(METABRIC_BETA_FILE, METABRIC_M_FILE)
    targets: dict[str, dict[str, object]] = {}
    universe: set[str] = set()
    for target, label_col in METABRIC_TARGETS.items():
        directory = destination / target
        directory.mkdir(parents=True, exist_ok=True)
        source = CohortSource(**{**METABRIC_SOURCE.__dict__, "label_col": label_col})
        cohorts, genes = build_cohorts([source], directory)
        if not cohorts:
            raise FileNotFoundError("the METABRIC matrices are absent")
        universe |= set(genes)

        data = load_omics(
            raw_dir=REPO_ROOT,
            rna_file=source.rna_file,
            cnv_file=source.cnv_file,
            methy_file=source.methy_file,
            clin_file=source.clin_file,
            label_col=label_col,
        )
        folds = make_folds(data.y, data.patient_ids, N_SPLITS, N_REPEATS, FOLD_SEED)
        save_folds(folds, data.patient_ids, directory / "folds.json")
        targets[target] = {
            "label_col": label_col,
            "cohort": cohorts[source.key],
            "files": copy_small_inputs(directory, METABRIC_SMALL_INPUTS),
        }

    graph = build_graph(sorted(universe), destination)
    for target in METABRIC_TARGETS:
        shutil.copy2(destination / "string_edges.tsv", destination / target)
    manifest: dict[str, object] = {
        "cohort": "METABRIC",
        "built_by": "utils/build_cohort_bundle.py",
        "partition": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": FOLD_SEED},
        "methylation": methylation,
        "targets": targets,
        "graph": graph,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for target, entry in targets.items():
        # Each target directory is a complete bundle in its own right.
        own = {
            "cohort": f"METABRIC, {entry['label_col']}",
            "built_by": "utils/build_cohort_bundle.py",
            "partition": manifest["partition"],
            "cohorts": {METABRIC_SOURCE.key: entry["cohort"]},
            "methylation": methylation,
            "graph": graph,
            "files": entry["files"],
        }
        (destination / target / "MANIFEST.json").write_text(
            json.dumps(own, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def report(manifest: dict[str, object], destination: Path) -> None:
    """Print what was built and how large it is.

    Args:
        manifest: Manifest returned by a build function.
        destination: Directory that was written.
    """
    total = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    print(f"\nbundle at {destination.relative_to(REPO_ROOT)} — {total / 1e6:.0f} MB")
    cohorts = manifest.get("cohorts")
    if isinstance(cohorts, dict):
        for key, entry in cohorts.items():
            print(
                f"  {key:12} {entry['n_patients']} patients x "
                f"{entry['n_genes']} genes  {entry['size_mb']} MB"
            )
    graph = manifest.get("graph")
    if isinstance(graph, dict):
        print(f"  {'graph':12} {graph['n_edges']} edges  {graph['size_mb']} MB")
    encoders = manifest.get("encoders")
    if isinstance(encoders, dict):
        for cohort, entry in encoders.items():
            print(f"  encoder/{cohort:4} {entry['size_mb']} MB")


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested bundles.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description="Build the cohort bundle.")
    parser.add_argument(
        "--cohort",
        choices=("tcga", "metabric", "all", "tcga_ssl", "metabric_ssl"),
        default="tcga",
        help="which cohort to bundle",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    builders = {
        "tcga": (build_tcga, DATA_ROOT / "tcga"),
        "metabric": (build_metabric, DATA_ROOT / "metabric"),
        "tcga_ssl": (build_tcga_ssl, DATA_ROOT / "tcga"),
        "metabric_ssl": (build_metabric_ssl, DATA_ROOT / "metabric" / "ssl"),
    }
    chosen = ["tcga", "metabric"] if args.cohort == "all" else [args.cohort]
    for cohort in chosen:
        builder, directory = builders[cohort]
        try:
            manifest = builder(directory)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        report(manifest, directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
