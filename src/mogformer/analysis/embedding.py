"""Outcome-blind pretraining and patient embedding extraction.

This is the phase that produces the representation everything downstream is
built on: the consensus clustering, the survival analysis and the interventional
probes all read the encoder frozen here.

Two properties of this phase are load-bearing and easy to lose in a refactor.

**The survival firewall.** No survival column is read here. The encoder never
sees an outcome, which is what makes the later confirmatory Cox model a genuine
test of the representation rather than a restatement of its training signal. The
loader is given the clinical table only for its subtype column.

**Preprocessing is fit once, on everyone.** Unlike the supervised phase, which
refits inside every fold, this phase fits scaling and gene selection across the
whole cohort. That is deliberate, not an oversight: there is no held-out label to
leak into, and every downstream analysis needs one fixed gene axis and one fixed
scaling so that embeddings, graph tensors and probe inputs stay mutually
comparable. The leakage question reappears at the confirmatory stage, where
survival enters for the first time.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from mogformer.config import ExperimentConfig, save_config
from mogformer.data.omics import MODALITY_ORDER, OmicsData
from mogformer.data.preprocess import MultiOmicsTransformer
from mogformer.graph.grn import GRNSignedCache
from mogformer.graph.positional_encoding import GraphPositionalEncoding
from mogformer.graph.spd import compute_shortest_path_matrix
from mogformer.graph.string_graph import StringGraphCache
from mogformer.models.ssl import MOGFormerSSL
from mogformer.training.losses import participation_ratio
from mogformer.training.seed import seed_everything
from mogformer.training.trainer import MaskedReconstructionTrainer

logger = logging.getLogger(__name__)


@dataclass
class PretrainingBundle:
    """Everything a downstream phase needs to reuse a frozen encoder.

    Attributes:
        embeddings: Patient embeddings, shape ``(n_patients, d)``, in cohort
            row order.
        patient_ids: Patient barcodes, parallel to ``embeddings``.
        selected_genes: The frozen gene axis, in model column order.
        graph_pe: Positional encodings for that gene axis.
        spd: Shortest-path matrix for that gene axis.
        grn: Signed regulatory matrix, or None.
        probe_inputs: Standardised model inputs, shape
            ``(n_patients, n_genes, 3)``, exactly what the encoder consumed.
        best_val: Best monitoring loss reached during pretraining.
    """

    embeddings: torch.Tensor
    patient_ids: list[str]
    selected_genes: list[str]
    graph_pe: torch.Tensor
    spd: torch.Tensor
    grn: torch.Tensor | None
    probe_inputs: torch.Tensor
    best_val: float


def split_train_monitor(
    data: OmicsData,
    monitor_frac: float = 0.12,
    seed: int = 42,
    stratify_by_subtype: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split patients into a training and a monitoring slice.

    The monitoring slice exists only to decide when to stop; it carries no
    outcome and is not a test set. Stratifying by subtype makes it
    representative, and PAM50 is not the survival outcome, so doing so does not
    touch the firewall.

    Args:
        data: Loaded cohort.
        monitor_frac: Fraction held out for monitoring.
        seed: Seed for the split.
        stratify_by_subtype: Stratify the split on the subtype label.

    Returns:
        Tuple of sorted training and monitoring index arrays.
    """
    indices = np.arange(data.n_patients)
    train_idx, monitor_idx = train_test_split(
        indices,
        test_size=monitor_frac,
        random_state=seed,
        stratify=data.y if stratify_by_subtype else None,
    )
    logger.info(
        "pretraining split: %d train / %d monitor (%.1f%% held out for early "
        "stopping only)",
        len(train_idx),
        len(monitor_idx),
        100 * len(monitor_idx) / data.n_patients,
    )
    return np.sort(train_idx), np.sort(monitor_idx)


def identify_subtype_patients(
    data: OmicsData,
    label: str,
    expected_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the indices of patients carrying one subtype label.

    Matching is exact rather than by substring, and the count is checked against
    a pre-registered expectation, because a silently renamed label produces a
    cohort of the wrong size that every later phase then inherits.

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
def extract_embeddings(
    model: MOGFormerSSL,
    loader: DataLoader,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Read the patient summary token for every patient in a loader.

    Masking is off: this is the representation of the patient as observed, not
    of a partially hidden one. The loader must not shuffle, or the returned rows
    will not line up with the cohort's patient order.

    Args:
        model: Frozen encoder.
        loader: Unshuffled batches over the cohort.
        graph_pe: Positional encodings for the frozen gene axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device the graph tensors live on.

    Returns:
        Embeddings of shape ``(n_patients, d)`` in loader order.
    """
    model.eval()
    embeddings = []
    for batch in loader:
        rna, cnv, methy = (tensor.to(device) for tensor in tuple(batch)[:3])
        out = model(rna, cnv, methy, graph_pe, spd, grn, mask=False)
        embeddings.append(out["c"].cpu())
    return torch.cat(embeddings, 0)


def build_frozen_axes(
    config: ExperimentConfig, data: OmicsData, device: torch.device
) -> tuple[
    MultiOmicsTransformer, list[str], torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    """Fit the study-wide preprocessing and build its graph tensors.

    Everything produced here is frozen for the rest of the study, so that
    embeddings, probe inputs and graph tensors share one gene axis.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort.
        device: Where the graph tensors should live.

    Returns:
        Tuple of the fitted transformer, the selected genes, the positional
        encodings, the shortest-path matrix and the regulatory matrix or None.
    """
    from mogformer.data.omics import load_curated_genes

    curated = load_curated_genes(config.data.curated_genes_file, data.gene_names)
    preprocessor = MultiOmicsTransformer(
        n_genes=data.n_genes,
        gene_names=data.gene_names,
        top_k=config.preprocess.top_k,
        curated_genes=curated,
        active_modalities=MODALITY_ORDER,
        mad_on_log_rna=config.preprocess.mad_on_log_rna,
    )
    preprocessor.fit(data.X)
    selected = preprocessor.get_selected_gene_names()
    logger.info("frozen gene axis: %d genes", len(selected))

    string_cache = StringGraphCache(
        config.graph.ppi_file,
        config.graph.alias_file,
        universe_genes=data.gene_names,
        confidence_threshold=config.graph.string_threshold,
    )
    adjacency = torch.as_tensor(
        string_cache.induced_adjacency(selected), dtype=torch.float32
    )
    graph_pe = GraphPositionalEncoding(
        pe_dim=config.graph.pe_dim, method=config.graph.pe_method
    )(adjacency).to(device)
    spd = compute_shortest_path_matrix(
        adjacency, max_distance=config.graph.max_distance
    ).to(device)

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

    return preprocessor, selected, graph_pe, spd, grn


def run_pretraining(
    config: ExperimentConfig,
    data: OmicsData,
    monitor_frac: float = 0.12,
    device: torch.device | None = None,
) -> PretrainingBundle:
    """Pretrain the encoder outcome-blind and extract the patient embeddings.

    Args:
        config: Resolved experiment configuration.
        data: Loaded cohort. Must carry no survival column.
        monitor_frac: Fraction held out for early stopping only.
        device: Where to train; defaults to CUDA when available.

    Returns:
        The frozen encoder's outputs and the axes they are defined on.
    """
    seed_everything(config.train.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("pretraining on %s", device)

    preprocessor, selected, graph_pe, spd, grn = build_frozen_axes(config, data, device)
    n_selected = len(selected)
    universe_position = {gene: i for i, gene in enumerate(data.gene_names)}
    gene_ids = [universe_position[gene] for gene in selected]

    features = preprocessor.transform(data.X)
    rna, cnv, methy = (
        features[:, :n_selected],
        features[:, n_selected : 2 * n_selected],
        features[:, 2 * n_selected :],
    )
    dataset = TensorDataset(
        torch.as_tensor(rna, dtype=torch.float32),
        torch.as_tensor(cnv, dtype=torch.float32),
        torch.as_tensor(methy, dtype=torch.float32),
    )

    train_idx, monitor_idx = split_train_monitor(
        data, monitor_frac=monitor_frac, seed=config.train.seed
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    monitor_loader = DataLoader(
        Subset(dataset, monitor_idx.tolist()),
        batch_size=config.train.batch_size,
        shuffle=False,
    )
    # Unshuffled over the whole cohort, so extracted rows match patient order.
    full_loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False)

    model = MOGFormerSSL(
        d=config.model.d,
        pe_dim=config.graph.pe_dim,
        mini_heads=config.model.mini_heads,
        global_heads=config.model.global_heads,
        global_layers=config.model.global_layers,
        dropout=config.model.dropout,
        max_distance=config.graph.max_distance,
        attention_bias_mode=config.model.attention_bias_mode,
        use_grn=config.graph.use_grn and grn is not None,
        gene_id_embedding=True,
        n_universe=data.n_genes,
        gene_ids=gene_ids,
        lambda_gate=config.model.lambda_gate,
        fusion_type=config.model.fusion_type,
    ).to(device)

    trainer = MaskedReconstructionTrainer(
        model,
        train_loader,
        monitor_loader,
        device,
        spd,
        graph_pe,
        grn,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    best_val = trainer.fit_early(
        config.train.max_epochs, config.train.patience, config.train.log_every
    )

    # Freeze explicitly rather than relying on no_grad at every call site.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    embeddings = extract_embeddings(model, full_loader, graph_pe, spd, grn, device)
    if embeddings.shape[0] != data.n_patients:
        raise RuntimeError(
            f"extracted {embeddings.shape[0]} embeddings for "
            f"{data.n_patients} patients; the extraction loader must not shuffle"
        )
    logger.info(
        "extracted %s embeddings | participation ratio %.2f",
        tuple(embeddings.shape),
        participation_ratio(embeddings),
    )

    probe_inputs = torch.stack(
        [
            torch.as_tensor(rna, dtype=torch.float32),
            torch.as_tensor(cnv, dtype=torch.float32),
            torch.as_tensor(methy, dtype=torch.float32),
        ],
        dim=-1,
    )

    bundle = PretrainingBundle(
        embeddings=embeddings,
        patient_ids=list(data.patient_ids),
        selected_genes=list(selected),
        graph_pe=graph_pe.cpu(),
        spd=spd.cpu(),
        grn=None if grn is None else grn.cpu(),
        probe_inputs=probe_inputs,
        best_val=best_val,
    )
    save_pretraining_bundle(bundle, model, preprocessor, config, data)
    return bundle


def save_pretraining_bundle(
    bundle: PretrainingBundle,
    model: MOGFormerSSL,
    preprocessor: MultiOmicsTransformer,
    config: ExperimentConfig,
    data: OmicsData,
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

    Returns:
        The directory written to.
    """
    root = config.output_dir / "representation_bundle"
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    (root / "graph").mkdir(parents=True, exist_ok=True)

    columns = [f"c_{i:03d}" for i in range(bundle.embeddings.shape[1])]
    frame = pd.DataFrame(bundle.embeddings.numpy(), columns=columns)
    frame.insert(0, "patient_id", bundle.patient_ids)
    # Carried for downstream stratification; never used during pretraining.
    frame["subtype"] = [data.inverse_label_map[int(code)] for code in data.y]
    _write_table(frame, root / "embeddings" / "C_all.parquet")

    torch.save(
        {"state_dict": model.state_dict(), "best_val": bundle.best_val},
        root / "encoder_frozen.pt",
    )
    torch.save(
        {
            "spd": bundle.spd,
            "graph_pe": bundle.graph_pe,
            "grn": bundle.grn,
            "gene_order": bundle.selected_genes,
        },
        root / "graph" / "graph_tensors.pt",
    )
    torch.save(
        {
            "inputs": bundle.probe_inputs,
            "patient_ids": bundle.patient_ids,
            "gene_order": bundle.selected_genes,
        },
        root / "probe_inputs.pt",
    )

    (root / "gene_universe.json").write_text(
        json.dumps(
            {
                "selected_genes": bundle.selected_genes,
                "n_selected": len(bundle.selected_genes),
                "top_k": config.preprocess.top_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "norm_stats.json").write_text(
        json.dumps(_normalisation_stats(preprocessor), indent=2), encoding="utf-8"
    )
    save_config(config, root / "config.yaml")

    logger.info("wrote the representation bundle to %s", root)
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
    """Read a saved bundle back for a downstream phase.

    Args:
        root: The ``representation_bundle`` directory.

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
        base / "graph" / "graph_tensors.pt",
        base / "probe_inputs.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"bundle is incomplete, missing: {missing}")

    encoder = torch.load(base / "encoder_frozen.pt", map_location="cpu")
    graph = torch.load(base / "graph" / "graph_tensors.pt", map_location="cpu")
    probe = torch.load(base / "probe_inputs.pt", map_location="cpu")

    if list(graph["gene_order"]) != list(probe["gene_order"]):
        raise ValueError(
            "gene order disagrees between the graph tensors and the probe "
            "inputs; the bundle was assembled from mismatched runs"
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
        Tuple of the sliced embeddings and their patient barcodes.
    """
    selected = list(indices)
    return (
        bundle.embeddings[selected],
        [bundle.patient_ids[i] for i in selected],
    )
