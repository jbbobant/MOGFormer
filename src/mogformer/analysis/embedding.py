"""Outcome-blind pretraining and patient embedding extraction.

This produces the representation every downstream analysis is built on: the
consensus clustering, the survival analysis and the interventional probes all
read the encoder frozen here.

Three properties of this step are load-bearing and easy to lose.

**The survival firewall.** No survival column is read here. The encoder never
sees an outcome, which is what makes the later confirmatory Cox model a genuine
test of the representation rather than a restatement of its training signal. The
loader is given the clinical table only for its subtype column.

**One frozen axis.** Gene selection and scaling are fit once and frozen, so that
embeddings, graph tensors and probe inputs stay mutually comparable across
every downstream analysis. They are fit on the patients the encoder trains and
stops on; the held-out patients are only transformed.

**Three disjoint splits.** Patients are split, with a seed of their own, into
a training split (the backward pass), a monitoring split (when to stop, scored
on one fixed mask draw) and a held-out split that is never trained on and
never used for stopping — the only place reconstruction measures
generalisation. Encoders trained with different seeds share the splits, so
their held-out scores are paired.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from mogformer.config import ExperimentConfig, save_config
from mogformer.data.batch_correction import (
    BATCH_CORRECTIONS,
    correct_batches,
    load_batch_labels,
)
from mogformer.data.gene_programs import PROGRAM_SOURCES, fit_coexpression_programs
from mogformer.data.omics import MODALITY_ORDER, OmicsData, load_curated_genes
from mogformer.data.preprocess import MultiOmicsTransformer
from mogformer.graph.grn import GRNSignedCache
from mogformer.graph.positional_encoding import GraphPositionalEncoding
from mogformer.graph.spd import compute_shortest_path_matrix
from mogformer.graph.string_graph import (
    StringGraphCache,
    load_string_edge_weights,
    load_string_edges,
)
from mogformer.models.layers.masking import MaskedMultiModalMasker
from mogformer.models.ssl import MOGFormerSSL
from mogformer.training.losses import participation_ratio
from mogformer.training.seed import seed_everything
from mogformer.training.trainer import MaskedReconstructionTrainer

logger = logging.getLogger(__name__)

#: Patient read-outs extracted from every frozen encoder. The summary token is
#: read without masking; its mean over random masks smooths what one view of
#: the patient happens to show; the mean of the final gene states bypasses the
#: summary token altogether. Which serves best is decided by the evaluation.
READOUTS: tuple[str, ...] = ("summary", "summary_masked_mean", "gene_mean")


def reference_masker() -> MaskedMultiModalMasker:
    """Return the mask every encoder is scored and read out under.

    15% of genes hidden whole and 15% with one omics hidden, expression
    weighted twice. Encoders may train under harder masks; held-out
    reconstruction and the masked read-out use this one, so encoders trained
    differently are compared on the same task.

    Returns:
        The masker.
    """
    return MaskedMultiModalMasker(0.15, 0.15, (2.0, 1.0, 1.0))


@dataclass
class FrozenAxes:
    """The study-wide gene axis and everything defined on it.

    Attributes:
        preprocessor: Fitted selection, imputation and scaling.
        selected_genes: Gene symbols in model column order.
        graph_pe: Positional encodings, shape ``(n_genes, pe_dim)``.
        spd: Shortest-path matrix, shape ``(n_genes, n_genes)``.
        grn: Signed regulatory matrix, or None.
        n_isolated: Genes without an interaction edge.
    """

    preprocessor: MultiOmicsTransformer
    selected_genes: list[str]
    graph_pe: torch.Tensor
    spd: torch.Tensor
    grn: torch.Tensor | None
    n_isolated: int


@dataclass
class PatientSplits:
    """Disjoint patient splits of a pretraining run, as cohort row indices.

    Attributes:
        train: Patients the backward pass sees.
        monitor: Patients scored to decide when to stop.
        heldout: Patients never trained on and never used for stopping.
    """

    train: np.ndarray
    monitor: np.ndarray
    heldout: np.ndarray


@dataclass
class PretrainingBundle:
    """Everything a downstream analysis needs to reuse a frozen encoder.

    Attributes:
        readouts: Patient embeddings by read-out name, each of shape
            ``(n_patients, d)`` in cohort row order.
        patient_ids: Patient barcodes, parallel to the embeddings.
        selected_genes: The frozen gene axis, in model column order.
        graph_pe: Positional encodings for that gene axis.
        spd: Shortest-path matrix for that gene axis.
        grn: Signed regulatory matrix, or None.
        probe_inputs: Standardised model inputs, shape
            ``(n_patients, n_genes, 3)``, exactly what the encoder consumed.
        observed: True where an input was measured rather than imputed, same
            shape as ``probe_inputs``.
        splits: The patient splits.
        best_val: Best monitoring loss reached during pretraining.
        reconstruction: Held-out and cross-omics reconstruction rows.
        record: Training record (epochs, timing, gates).
    """

    readouts: dict[str, torch.Tensor]
    patient_ids: list[str]
    selected_genes: list[str]
    graph_pe: torch.Tensor
    spd: torch.Tensor
    grn: torch.Tensor | None
    probe_inputs: torch.Tensor
    observed: torch.Tensor
    splits: PatientSplits
    best_val: float
    reconstruction: list[dict[str, Any]] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def embeddings(self) -> torch.Tensor:
        """Return the summary-token read-out."""
        return self.readouts["summary"]


def split_pretraining_patients(
    data: OmicsData,
    monitor_frac: float = 0.12,
    heldout_frac: float = 0.0,
    seed: int = 42,
    stratify_by_subtype: bool = True,
) -> PatientSplits:
    """Split patients into training, monitoring and held-out sets.

    Both fractions are of the whole cohort. Stratifying by subtype makes each
    split representative; PAM50 is not the survival outcome, so doing so does
    not touch the firewall.

    Args:
        data: Loaded cohort.
        monitor_frac: Share used only to decide when to stop.
        heldout_frac: Share never trained on and never used for stopping; 0
            for none.
        seed: Seed of the split, independent of the training seed.
        stratify_by_subtype: Stratify on the subtype label.

    Returns:
        The three sorted index arrays.
    """
    indices = np.arange(data.n_patients)
    labels = data.y if stratify_by_subtype else None
    heldout = np.array([], dtype=int)
    rest = indices
    if heldout_frac > 0:
        rest, heldout = train_test_split(
            indices, test_size=heldout_frac, random_state=seed, stratify=labels
        )
    train, monitor = train_test_split(
        rest,
        test_size=monitor_frac / (1.0 - heldout_frac),
        random_state=seed,
        stratify=None if labels is None else labels[rest],
    )
    logger.info(
        "pretraining split: %d train / %d monitor / %d held out",
        len(train),
        len(monitor),
        len(heldout),
    )
    return PatientSplits(np.sort(train), np.sort(monitor), np.sort(heldout))


def identify_subtype_patients(
    data: OmicsData,
    label: str,
    expected_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the indices of patients carrying one subtype label.

    Matching is exact rather than by substring, and the count is checked against
    a pre-registered expectation, because a silently renamed label produces a
    cohort of the wrong size that every later analysis then inherits.

    Args:
        data: Loaded cohort.
        label: Exact subtype name.
        expected_range: Inclusive bounds the count should fall within. A count
            outside them warns rather than raises, since a legitimately filtered
            cohort can be smaller.

    Returns:
        Indices of the matching patients.

    Raises:
        ValueError: If the label is absent from the cohort's label map.
    """
    if label not in data.label_map:
        raise ValueError(
            f"{label!r} not found among the cohort's labels: {list(data.label_map)}"
        )
    matched = np.where(data.y == data.label_map[label])[0]
    logger.info("%s: %d patients", label, len(matched))

    if expected_range is not None:
        low, high = expected_range
        if not low <= len(matched) <= high:
            warnings.warn(
                f"{label!r} matched {len(matched)} patients, outside the "
                f"pre-registered {low}-{high} range; check the label map before "
                "trusting anything downstream",
                stacklevel=2,
            )
    return matched


@torch.no_grad()
def extract_readouts(
    model: MOGFormerSSL,
    inputs: torch.Tensor,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 32,
    n_masks: int = 10,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Read every patient read-out of :data:`READOUTS` from a frozen encoder.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        graph_pe: Positional encodings for the frozen gene axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device the graph tensors live on.
        batch_size: Patients per forward pass.
        n_masks: Random masks averaged over for ``summary_masked_mean``.
        seed: Seed of those masks.

    Returns:
        Read-out name to embeddings of shape ``(n_patients, d)``, in input
        row order.
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    masker = reference_masker()
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in READOUTS}
    for start in range(0, inputs.shape[0], batch_size):
        batch = inputs[start : start + batch_size]
        rna, cnv, methy = (batch[..., k].to(device) for k in range(3))
        out = model(rna, cnv, methy, graph_pe, spd, grn, mask=False)
        collected["summary"].append(out["c"].cpu())
        collected["gene_mean"].append(out["H_genes"].mean(dim=1).cpu())
        masked = torch.zeros_like(out["c"])
        for _ in range(n_masks):
            mask_bool, _ = masker(batch, generator=generator)
            masked += model(
                rna,
                cnv,
                methy,
                graph_pe,
                spd,
                grn,
                mask=True,
                mask_bool=mask_bool.to(device),
            )["c"]
        collected["summary_masked_mean"].append((masked / n_masks).cpu())
    return {name: torch.cat(parts, 0) for name, parts in collected.items()}


def build_frozen_axes(
    config: ExperimentConfig,
    data: OmicsData,
    device: torch.device,
    fit_rows: np.ndarray | None = None,
) -> FrozenAxes:
    """Fit the study-wide preprocessing and build its graph tensors.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort.
        device: Where the graph tensors should live.
        fit_rows: Patients the preprocessing is fitted on; all when None.

    Returns:
        The frozen axes.
    """
    curated = load_curated_genes(config.data.curated_genes_file, data.gene_names)
    preprocessor = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=config.preprocess.top_k,
        curated_genes=curated,
        active_modalities=MODALITY_ORDER,
        mad_on_log_rna=config.preprocess.mad_on_log_rna,
        impute=config.preprocess.impute,
        rna_scale=config.preprocess.rna_scale,
        clip_z=config.preprocess.clip_z,
    )
    preprocessor.fit(data.X if fit_rows is None else data.X[fit_rows])
    selected = preprocessor.get_selected_gene_names()
    logger.info("frozen gene axis: %d genes", len(selected))

    if config.graph.edges_file is not None:
        string_cache = StringGraphCache.from_edges(
            load_string_edges(config.graph.edges_file),
            config.graph.string_threshold,
            weights=load_string_edge_weights(config.graph.edges_file),
        )
    else:
        string_cache = StringGraphCache(
            config.graph.ppi_file,
            config.graph.alias_file,
            universe_genes=data.gene_names,
            confidence_threshold=config.graph.string_threshold,
        )
    adjacency = torch.as_tensor(
        string_cache.induced_adjacency(selected, weighted=config.graph.weighted_pe),
        dtype=torch.float32,
    )
    graph_pe = GraphPositionalEncoding(
        pe_dim=config.graph.pe_dim,
        method=config.graph.pe_method,
        weighted=config.graph.weighted_pe,
    )(adjacency).to(device)
    spd = compute_shortest_path_matrix(
        adjacency, max_distance=config.graph.max_distance
    ).to(device)
    n_isolated = int(((adjacency > 0).sum(dim=1) == 0).sum().item())
    logger.info(
        "interaction graph on the frozen axis: %d edges, %d isolated genes",
        int((adjacency > 0).sum().item()) // 2,
        n_isolated,
    )

    grn = None
    if config.graph.grn_file is not None:
        grn_cache = GRNSignedCache(
            config.graph.grn_file, universe_genes=data.gene_names
        )
        grn_np = grn_cache.induced_signed_adjacency(selected)
        grn = torch.as_tensor(grn_np, dtype=torch.float32).to(device)
        logger.info(
            "regulatory edges on the frozen axis: %d (%d activating / %d repressing)",
            int((grn_np != 0).sum()),
            int((grn_np > 0).sum()),
            int((grn_np < 0).sum()),
        )

    return FrozenAxes(preprocessor, selected, graph_pe, spd, grn, n_isolated)


def encoder_settings(config: ExperimentConfig) -> dict[str, Any]:
    """Describe an encoder in the settings spelling stored beside its weights.

    :func:`mogformer.experiments.metabric.encoder_from_settings` rebuilds an
    encoder from exactly these keys, so a frozen encoder can be reloaded
    without its configuration file.

    Args:
        config: Resolved experiment configuration.

    Returns:
        The settings.
    """
    return {
        "d": config.model.d,
        "pe_dim": config.graph.pe_dim,
        "pe_method": config.graph.pe_method,
        "mini_heads": config.model.mini_heads,
        "global_heads": config.model.global_heads,
        "global_layers": config.model.global_layers,
        "dropout": config.model.dropout,
        "max_dist": config.graph.max_distance,
        "attention_bias_mode": config.model.attention_bias_mode,
        "gene_id_embedding": True,
        "attention_lambda_gate": config.model.lambda_gate,
        "fusion_type": config.model.fusion_type,
        "graph_relations": ["ppi", "grn"] if config.graph.use_grn else ["ppi"],
        "mask_gene_frac": config.model.mask_gene_frac,
        "mask_modality_frac": config.model.mask_modality_frac,
        "mask_modality_weights": list(config.model.mask_modality_weights),
        "global_offset": config.model.global_offset,
        "isolated_token": config.model.isolated_token,
        "identity_modulation": config.model.identity_modulation,
        "n_summary_tokens": config.model.n_summary_tokens,
        "global_head": config.model.global_head,
        "fusion_residual": config.model.fusion_residual,
        "neighbourhood_mask_share": config.model.neighbourhood_mask_share,
        "omics_mask_prob": config.model.omics_mask_prob,
        "program_source": config.model.program_source,
        "n_programs": config.model.n_programs,
        "program_width": config.model.program_width,
        "program_layers": config.model.program_layers,
        "program_input": config.model.program_input,
        "lr": config.train.lr,
        "weight_decay": config.train.weight_decay,
        "max_epochs": config.train.max_epochs,
        "patience": config.train.patience,
        "batch_size": config.train.batch_size,
        "seed": config.train.seed,
        "top_k": config.preprocess.top_k,
        "gate_lr_multiplier": config.train.gate_lr_multiplier,
    }


def build_encoder(
    config: ExperimentConfig, n_genes: int, use_grn: bool
) -> MOGFormerSSL:
    """Build an untrained encoder for a gene axis of ``n_genes``.

    The identity table has one row per selected gene: rows for genes that are
    never selected would never receive a gradient.

    Args:
        config: Resolved experiment configuration.
        n_genes: Genes on the frozen axis.
        use_grn: Enable the regulatory bias (a regulatory matrix exists).

    Returns:
        The encoder, on the CPU.
    """
    return MOGFormerSSL(
        d=config.model.d,
        pe_dim=config.graph.pe_dim,
        mini_heads=config.model.mini_heads,
        global_heads=config.model.global_heads,
        global_layers=config.model.global_layers,
        dropout=config.model.dropout,
        max_distance=config.graph.max_distance,
        attention_bias_mode=config.model.attention_bias_mode,
        use_grn=use_grn,
        gene_id_embedding=True,
        n_universe=n_genes,
        gene_ids=list(range(n_genes)),
        lambda_gate=config.model.lambda_gate,
        fusion_type=config.model.fusion_type,
        mask_gene_frac=config.model.mask_gene_frac,
        mask_modality_frac=config.model.mask_modality_frac,
        mask_modality_weights=tuple(config.model.mask_modality_weights),  # type: ignore[arg-type]
        global_offset=config.model.global_offset,
        isolated_token=config.model.isolated_token,
        identity_modulation=config.model.identity_modulation,
        n_summary_tokens=config.model.n_summary_tokens,
        global_head=config.model.global_head,
        fusion_residual=config.model.fusion_residual,
        neighbourhood_mask_share=config.model.neighbourhood_mask_share,
        omics_mask_prob=config.model.omics_mask_prob,
        n_programs=config.model.n_programs,
        program_width=config.model.program_width,
        program_layers=config.model.program_layers,
        program_input=config.model.program_input,
    )


@torch.no_grad()
def heldout_reconstruction(
    model: MOGFormerSSL,
    inputs: torch.Tensor,
    observed: torch.Tensor,
    splits: PatientSplits,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    delta: float = 1.0,
    batch_size: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Score reconstruction on held-out patients against the gene-mean baseline.

    Two tasks, both on patients the encoder never trained or stopped on:

    * ``masked``: one fixed draw of the reference mask
      (:func:`reference_masker`), whatever mask the encoder trained under;
    * ``omics_hidden``: one omics hidden entirely, for every gene at once, and
      reconstructed from the other two.

    The baseline predicts each gene's mean over the training patients; a head
    that does not beat it has learned the marginal distribution, not the
    patient. Only measured entries are scored.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs of the whole cohort, ``(n, n_genes, 3)``.
        observed: True where an input was measured, same shape.
        splits: The run's patient splits.
        graph_pe: Positional encodings.
        spd: Shortest-path matrix.
        grn: Signed regulatory matrix, or None.
        device: Where the model lives.
        delta: Huber transition point, as in training.
        batch_size: Patients per forward pass.
        seed: Seed of the fixed mask draw.

    Returns:
        One row per task, omics and head: Huber loss, the baseline's loss,
        their ratio and the number of entries scored. Empty without a held-out
        split.
    """
    if len(splits.heldout) == 0:
        return []
    model.eval()
    gene_mean = inputs[torch.as_tensor(splits.train)].mean(dim=0)
    held_inputs = inputs[torch.as_tensor(splits.heldout)]
    held_observed = observed[torch.as_tensor(splits.heldout)]
    generator = torch.Generator().manual_seed(seed)
    masks = {"masked": reference_masker()(held_inputs, generator=generator)[0]}
    for k, omics in enumerate(MODALITY_ORDER):
        hidden = torch.zeros_like(held_observed)
        hidden[..., k] = True
        masks[f"omics_hidden:{omics}"] = hidden

    rows: list[dict[str, Any]] = []
    for task, mask in masks.items():
        sums = {head: torch.zeros(3) for head in ("local", "global", "baseline")}
        counts = torch.zeros(3)
        for start in range(0, held_inputs.shape[0], batch_size):
            batch = held_inputs[start : start + batch_size]
            batch_mask = mask[start : start + batch_size]
            scored = batch_mask & held_observed[start : start + batch_size]
            rna, cnv, methy = (batch[..., k].to(device) for k in range(3))
            out = model(
                rna,
                cnv,
                methy,
                graph_pe,
                spd,
                grn,
                mask=True,
                mask_bool=batch_mask.to(device),
            )
            predictions = {
                "local": out["xhat_l"].cpu(),
                "global": out["xhat_g"].cpu(),
                "baseline": gene_mean.expand_as(batch),
            }
            weights = scored.float()
            for head, prediction in predictions.items():
                loss = F.huber_loss(prediction, batch, delta=delta, reduction="none")
                sums[head] += (loss * weights).sum(dim=(0, 1))
            counts += weights.sum(dim=(0, 1))
        task_name, _, only = task.partition(":")
        for k, omics in enumerate(MODALITY_ORDER):
            if (only and omics != only) or counts[k] == 0:
                continue
            baseline = float(sums["baseline"][k] / counts[k])
            for head in ("local", "global"):
                head_loss = float(sums[head][k] / counts[k])
                rows.append(
                    {
                        "task": task_name,
                        "omics": omics,
                        "head": head,
                        "loss": head_loss,
                        "gene_mean_baseline": baseline,
                        "ratio_to_baseline": head_loss / baseline,
                        "n_entries": int(counts[k]),
                        "n_patients": int(held_inputs.shape[0]),
                    }
                )
    return rows


def run_pretraining(
    config: ExperimentConfig,
    data: OmicsData,
    device: torch.device | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> PretrainingBundle:
    """Pretrain the encoder outcome-blind and extract the patient embeddings.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort, copy number on its configured scale. Must carry
            no survival column.
        device: Where to train; defaults to CUDA when available.
        provenance: Extra fields for the run's provenance record (machine,
            library versions).

    Returns:
        The frozen encoder's outputs and the axes they are defined on, also
        written to ``config.output_dir``.
    """
    seed_everything(config.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("pretraining %s on %s", config.name, device)

    splits = split_pretraining_patients(
        data,
        monitor_frac=config.train.monitor_frac,
        heldout_frac=config.train.heldout_frac,
        seed=config.train.split_seed,
    )
    fit_rows = np.concatenate([splits.train, splits.monitor])
    if config.preprocess.batch_correction is not None:
        if config.preprocess.batch_correction not in BATCH_CORRECTIONS:
            raise ValueError(
                f"unknown batch_correction {config.preprocess.batch_correction!r}; "
                f"available: {list(BATCH_CORRECTIONS)}"
            )
        if config.data.batch_file is None or config.data.batch_column is None:
            raise ValueError("batch correction needs data.batch_file and batch_column")
        batches = load_batch_labels(
            Path(config.data.raw_dir) / config.data.batch_file,
            config.data.batch_column,
            data.patient_ids,
        )
        data, _ = correct_batches(
            data, batches, fit_rows, config.preprocess.batch_modalities
        )
    axes = build_frozen_axes(config, data, device, fit_rows=fit_rows)
    n_selected = len(axes.selected_genes)

    features = axes.preprocessor.transform(data.X)
    observed_flat = axes.preprocessor.observed_mask(data.X)
    inputs = torch.stack(
        [
            torch.as_tensor(
                features[:, k * n_selected : (k + 1) * n_selected], dtype=torch.float32
            )
            for k in range(3)
        ],
        dim=-1,
    )
    observed = torch.stack(
        [
            torch.as_tensor(observed_flat[:, k * n_selected : (k + 1) * n_selected])
            for k in range(3)
        ],
        dim=-1,
    )
    logger.info(
        "inputs %s; %.3f%% of entries imputed and excluded from the loss",
        tuple(inputs.shape),
        100 * (1 - observed.float().mean().item()),
    )

    model = build_encoder(
        config, n_selected, use_grn=config.graph.use_grn and axes.grn is not None
    )
    model.masker.set_neighbourhoods((axes.spd == 1).cpu())
    program_sizes: list[int] = []
    if model.program_readout is not None:
        if config.model.program_source not in PROGRAM_SOURCES:
            raise ValueError(
                f"unknown program_source {config.model.program_source!r}; "
                f"available: {list(PROGRAM_SOURCES)}"
            )
        fit_rows = np.concatenate([splits.train, splits.monitor])
        assignment = fit_coexpression_programs(
            inputs[fit_rows].numpy(), config.model.n_programs
        )
        model.program_readout.set_assignment(torch.as_tensor(assignment))
        program_sizes = np.bincount(assignment).tolist()
        logger.info(
            "gene programs (%s): sizes %s", config.model.program_source, program_sizes
        )
    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info("encoder: %.2f M parameters", n_parameters / 1e6)

    # One mask draw on the monitoring split, seeded by the split seed so that
    # every training seed stops on the same criterion.
    monitor_rows = torch.as_tensor(splits.monitor)
    monitor_masks, _ = model.masker(
        inputs[monitor_rows],
        generator=torch.Generator().manual_seed(config.train.split_seed),
    )
    channels = [inputs[..., k] for k in range(3)]
    train_set = TensorDataset(*channels, observed)
    monitor_set = TensorDataset(
        *(channel[monitor_rows] for channel in channels),
        observed[monitor_rows],
        monitor_masks,
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(
        Subset(train_set, splits.train.tolist()),
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    monitor_loader = DataLoader(
        monitor_set, batch_size=config.train.batch_size, shuffle=False
    )

    trainer = MaskedReconstructionTrainer(
        model,
        train_loader,
        monitor_loader,
        device,
        axes.spd,
        axes.graph_pe,
        axes.grn,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        gate_lr_multiplier=config.train.gate_lr_multiplier,
        per_modality_loss=config.train.per_modality_loss,
        warmup_epochs=config.train.warmup_epochs,
        lr_schedule=config.train.lr_schedule,
        min_lr=config.train.min_lr,
        lambda_global=config.train.lambda_global,
        decorrelation_penalty=config.train.decorrelation_penalty,
        uniformity_weight=config.train.uniformity_weight,
        vicreg_weight=config.train.vicreg_weight,
        memory_bank_size=config.train.memory_bank_size,
        expander_width=config.train.expander_width,
        selection=config.train.selection,
        selection_tolerance=config.train.selection_tolerance,
        contrastive_weight=config.train.contrastive_weight,
        contrastive_temperature=config.train.contrastive_temperature,
        projection_width=config.train.projection_width,
        two_views=config.train.two_views,
        view_omics_prob=config.train.view_omics_prob,
    )
    started = time.perf_counter()
    best_val = trainer.fit_early(
        config.train.max_epochs, config.train.patience, config.train.log_every
    )
    minutes = (time.perf_counter() - started) / 60.0
    epochs = len(trainer.history.val)
    logger.info(
        "trained %d epochs in %.1f min (%.1f s/epoch); best monitoring loss %.4f",
        epochs,
        minutes,
        60.0 * minutes / max(epochs, 1),
        best_val,
    )

    # Freeze explicitly rather than relying on no_grad at every call site.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    readouts = extract_readouts(
        model,
        inputs,
        axes.graph_pe,
        axes.spd,
        axes.grn,
        device,
        batch_size=config.train.batch_size,
        n_masks=config.train.n_readout_masks,
        seed=config.train.split_seed,
    )
    for name, values in readouts.items():
        logger.info(
            "read-out %s: participation ratio %.2f", name, participation_ratio(values)
        )
    reconstruction = heldout_reconstruction(
        model,
        inputs,
        observed,
        splits,
        axes.graph_pe,
        axes.spd,
        axes.grn,
        device,
        batch_size=config.train.batch_size,
        seed=config.train.split_seed + 1,
    )
    for row in reconstruction:
        logger.info(
            "held-out %s %s (%s head): %.4f vs gene mean %.4f",
            row["task"],
            row["omics"],
            row["head"],
            row["loss"],
            row["gene_mean_baseline"],
        )

    best_epoch = int(np.argmin(trainer.history.val)) + 1 if epochs else 0
    record = {
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "selected_epoch": trainer.selected_epoch,
        "minutes": round(minutes, 2),
        "seconds_per_epoch": round(60.0 * minutes / max(epochs, 1), 2),
        "n_parameters": n_parameters,
        "n_selected_genes": n_selected,
        "n_isolated_genes": axes.n_isolated,
        "program_sizes": program_sizes,
        "attention_gates": trainer.attention_gates(),
        "imputed_share": float(1 - observed.float().mean().item()),
    }
    bundle = PretrainingBundle(
        readouts=readouts,
        patient_ids=list(data.patient_ids),
        selected_genes=list(axes.selected_genes),
        graph_pe=axes.graph_pe.cpu(),
        spd=axes.spd.cpu(),
        grn=None if axes.grn is None else axes.grn.cpu(),
        probe_inputs=inputs,
        observed=observed,
        splits=splits,
        best_val=best_val,
        reconstruction=reconstruction,
        record=record,
    )
    save_pretraining_bundle(
        bundle, model, axes.preprocessor, config, data, trainer, provenance
    )
    return bundle


def config_digest(config: ExperimentConfig) -> str:
    """Return a short hash of a resolved configuration's settings.

    Paths are left out, so the same settings read from another location hash
    the same.

    Args:
        config: Resolved experiment configuration.

    Returns:
        The first sixteen hexadecimal characters of a SHA-256 digest.
    """
    settings = asdict(config)
    settings.pop("results_dir", None)
    for section in ("data", "folds", "graph"):
        settings[section] = {
            key: value
            for key, value in settings[section].items()
            if not key.endswith(("_file", "_dir", "path"))
        }
    text = yaml.safe_dump(settings, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_pretraining_bundle(
    bundle: PretrainingBundle,
    model: MOGFormerSSL,
    preprocessor: MultiOmicsTransformer,
    config: ExperimentConfig,
    data: OmicsData,
    trainer: MaskedReconstructionTrainer | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write everything needed to reproduce or reuse the frozen encoder.

    The normalisation statistics matter as much as the weights: an
    interventional probe injects values in standardised units, and without the
    per-gene mean and scale those units cannot be interpreted biologically.

    Args:
        bundle: Extracted outputs.
        model: The frozen encoder.
        preprocessor: The fitted study-wide preprocessing.
        config: Resolved experiment configuration.
        data: Loaded cohort, for labels carried alongside the embeddings.
        trainer: The trainer, whose per-epoch history is written when given.
        provenance: Extra provenance fields.

    Returns:
        The directory written to.
    """
    root = config.output_dir
    (root / "embeddings").mkdir(parents=True, exist_ok=True)

    subtype = [data.inverse_label_map[int(code)] for code in data.y]
    for name, values in bundle.readouts.items():
        columns = [f"c_{i:03d}" for i in range(values.shape[1])]
        frame = pd.DataFrame(values.numpy(), columns=columns)
        frame.insert(0, "patient_id", bundle.patient_ids)
        # Carried for downstream stratification; never used during pretraining.
        frame["subtype"] = subtype
        _write_table(frame, root / "embeddings" / f"{name}.parquet")
        if name == "summary":
            _write_table(frame, root / "C_all.parquet")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_val": bundle.best_val,
            "config": encoder_settings(config),
        },
        root / "encoder_frozen.pt",
    )
    torch.save(
        {
            "spd": bundle.spd,
            "graph_pe": bundle.graph_pe,
            "grn": bundle.grn,
            "gene_order": bundle.selected_genes,
        },
        root / "graph_tensors.pt",
    )
    torch.save(
        {
            "inputs": bundle.probe_inputs,
            "observed": bundle.observed,
            "patient_ids": bundle.patient_ids,
            "gene_order": bundle.selected_genes,
        },
        root / "probe_inputs.pt",
    )
    if model.program_readout is not None:
        pd.DataFrame(
            {
                "gene": bundle.selected_genes,
                "program": model.program_readout.assignment.cpu().numpy(),
            }
        ).to_csv(root / "gene_programs.csv", index=False)
    (root / "gene_universe.json").write_text(
        json.dumps(
            {
                "selected_genes": bundle.selected_genes,
                "n_selected": len(bundle.selected_genes),
                "top_k": config.preprocess.top_k,
                "n_curated_selected": preprocessor.n_curated_selected_,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "norm_stats.json").write_text(
        json.dumps(_normalisation_stats(preprocessor), indent=2), encoding="utf-8"
    )
    (root / "heldout_reconstruction.json").write_text(
        json.dumps(
            {
                "estimand": "Huber loss per measured entry, held-out patients",
                "rows": bundle.reconstruction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if trainer is not None:
        history = trainer.history
        table = pd.DataFrame(
            {
                "epoch": np.arange(1, len(history.val) + 1),
                "learning_rate": history.learning_rate[: len(history.val)],
                "train": history.train,
                "monitor": history.val,
                "monitor_global": history.loss_global,
                "monitor_local": history.loss_local,
                "participation_ratio": history.participation_ratio,
                "rank_me": history.rank_me,
            }
        )
        if history.contrastive:
            table["train_contrastive"] = history.contrastive
        for k, omics in enumerate(MODALITY_ORDER):
            table[f"monitor_local_{omics}"] = [row[k] for row in history.per_modality]
            table[f"baseline_{omics}"] = [row[k] for row in history.baseline]
        table.to_csv(root / "training_history.csv", index=False)
    save_config(config, root / "config.yaml")

    patients = np.asarray(bundle.patient_ids)
    record = {
        "experiment": "ssl_pretraining",
        "run": config.name,
        "cohort_rule": config.data.cohort_rule,
        "cohort_file": config.data.cohort_file,
        "n_patients": len(bundle.patient_ids),
        "cnv_representation": config.preprocess.cnv_representation,
        "config_sha256_16": config_digest(config),
        "training_seed": config.train.seed,
        "split_seed": config.train.split_seed,
        "monitor_mask_seed": config.train.split_seed,
        "heldout_mask_seed": config.train.split_seed + 1,
        "readout_mask_seed": config.train.split_seed,
        "best_monitor_loss": bundle.best_val,
        "splits": {
            name: {"n": len(rows), "patient_ids": patients[rows].tolist()}
            for name, rows in (
                ("train", bundle.splits.train),
                ("monitor", bundle.splits.monitor),
                ("heldout", bundle.splits.heldout),
            )
        },
        **bundle.record,
        **dict(provenance or {}),
    }
    (root / "provenance.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    logger.info("wrote the frozen encoder and its outputs to %s", root)
    return root


def _normalisation_stats(
    preprocessor: MultiOmicsTransformer,
) -> dict[str, dict[str, list[float] | None]]:
    """Extract per-modality scaling and imputation statistics.

    Args:
        preprocessor: A fitted transformer.

    Returns:
        Mapping of modality to its mean, scale and imputation medians.
    """
    stats: dict[str, dict[str, list[float] | None]] = {}
    for modality in MODALITY_ORDER:
        scaler = preprocessor.scalers_[modality]
        stats[modality] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "median_impute": (
                preprocessor.medians_[modality].tolist()
                if preprocessor.impute
                else None
            ),
        }
    return stats


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame as parquet, falling back to CSV when unavailable.

    Args:
        frame: Table to write.
        path: Destination with a ``.parquet`` suffix.
    """
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError) as error:
        fallback = path.with_suffix(".csv")
        warnings.warn(
            f"could not write parquet ({error}); wrote {fallback} instead. "
            "Install pyarrow for the parquet artifact.",
            stacklevel=2,
        )
        frame.to_csv(fallback, index=False)


def load_pretraining_bundle(root: str | Path) -> dict[str, object]:
    """Read a saved run back for a downstream analysis.

    Args:
        root: A pretraining run directory.

    Returns:
        Mapping with the graph tensors, probe inputs, gene order and the
        encoder state dictionary.

    Raises:
        FileNotFoundError: If the directory lacks the expected artifacts.
        ValueError: If the gene order recorded in two artifacts disagrees.
    """
    base = Path(root)
    required = [
        base / "encoder_frozen.pt",
        base / "graph_tensors.pt",
        base / "probe_inputs.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run is incomplete, missing: {missing}")

    encoder = torch.load(base / "encoder_frozen.pt", map_location="cpu")
    graph = torch.load(base / "graph_tensors.pt", map_location="cpu")
    probe = torch.load(base / "probe_inputs.pt", map_location="cpu")

    if list(graph["gene_order"]) != list(probe["gene_order"]):
        raise ValueError(
            "gene order disagrees between the graph tensors and the probe "
            "inputs; the run was assembled from mismatched artifacts"
        )

    return {
        "state_dict": encoder["state_dict"],
        "spd": graph["spd"],
        "graph_pe": graph["graph_pe"],
        "grn": graph["grn"],
        "gene_order": list(graph["gene_order"]),
        "inputs": probe["inputs"],
        "patient_ids": list(probe["patient_ids"]),
    }


def subtype_slice(
    bundle: PretrainingBundle, indices: Sequence[int]
) -> tuple[torch.Tensor, list[str]]:
    """Take one subtype's rows out of an extracted embedding.

    Args:
        bundle: Extracted outputs.
        indices: Patient indices, as returned by
            :func:`identify_subtype_patients`.

    Returns:
        Tuple of the sliced summary embeddings and their patient barcodes.
    """
    selected = list(indices)
    return (
        bundle.embeddings[selected],
        [bundle.patient_ids[i] for i in selected],
    )
