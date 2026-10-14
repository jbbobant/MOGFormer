"""The trans probe re-measured on the external cohort from its frozen encoder.

TCGA's sign concordance can be re-tested from slopes a stored sweep already
wrote, which is why that half costs seconds. METABRIC's cannot: the archived
run saved the per-edge *co-expression*, not the per-edge slopes, so the sweep
has to be measured again. That means forward passes — roughly nine hundred
edges across nine intervention levels for fourteen hundred patients — which is
what the GPU is for.

Nothing is retrained. The archived encoder, its graph tensors and its probe
inputs are all on disk; this module rebuilds the model from the settings stored
beside its weights and re-measures. Retraining would introduce a second
encoder and make the two cohorts incomparable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mogformer.analysis.probe_trans import MASK_MODES, run_trans_sweep
from mogformer.experiments.paths import save_json
from mogformer.experiments.sign_test import rerun_sign_test
from mogformer.models.checkpoints import load_encoder_state, read_checkpoint
from mogformer.models.ssl import MOGFormerSSL

logger = logging.getLogger(__name__)

#: Files an archived self-supervised run leaves behind, all of which are read.
ENCODER_FILE = "encoder_frozen.pt"
GRAPH_FILE = "graph_tensors.pt"
PROBE_INPUTS_FILE = "probe_inputs.pt"

#: Null labels written into the sweep, matching what the sign test expects.
#: Only the regulatory graph is shuffled here; see ``GRAPH_NULLS``.
REAL_LABEL = "real"
PERMUTED_LABEL = "permute_grn_only"


@dataclass
class ArchivedEncoder:
    """An archived self-supervised run, ready to be probed again.

    Attributes:
        model: The encoder, weights loaded and in evaluation mode.
        inputs: Patient tensor of shape ``(n_patients, n_genes, 3)``.
        gene_order: Gene symbols in the tensor's own column order.
        graph_pe: Positional encodings for those genes.
        spd: Their shortest-path matrix.
        grn: Their signed regulatory adjacency, or None.
        config: Settings the run was trained under.
    """

    model: MOGFormerSSL
    inputs: torch.Tensor
    gene_order: list[str]
    graph_pe: torch.Tensor
    spd: torch.Tensor
    grn: torch.Tensor | None
    config: dict[str, Any]


def encoder_from_settings(
    config: Mapping[str, Any], n_genes: int, n_universe: int
) -> MOGFormerSSL:
    """Build an untrained encoder from the settings stored beside archived weights.

    Args:
        config: Settings in the archived spelling (``max_dist``,
            ``graph_relations``, ``attention_lambda_gate``, ...).
        n_genes: Genes on the gene axis.
        n_universe: Rows of the gene identity table; the axis's genes occupy
            its first ``n_genes`` positions.

    Returns:
        An encoder of the described shape, with fresh parameters.
    """
    relations = config.get("graph_relations", [])
    return MOGFormerSSL(
        d=config["d"],
        pe_dim=config["pe_dim"],
        mini_heads=config["mini_heads"],
        global_heads=config["global_heads"],
        global_layers=config["global_layers"],
        dropout=config["dropout"],
        max_distance=config.get("max_dist", config.get("max_distance", 5)),
        attention_bias_mode=config["attention_bias_mode"],
        use_grn="grn" in relations,
        gene_id_embedding=bool(config.get("gene_id_embedding", False)),
        n_universe=n_universe,
        gene_ids=list(range(n_genes)) if n_universe else None,
        lambda_gate=bool(config.get("attention_lambda_gate", False)),
        fusion_type=config.get("fusion_type", "gated"),
        mask_gene_frac=float(config.get("mask_gene_frac", 0.15)),
        mask_modality_frac=float(config.get("mask_modality_frac", 0.15)),
        mask_modality_weights=tuple(
            config.get("mask_modality_weights", (2.0, 1.0, 1.0))
        ),
        global_offset=bool(config.get("global_offset", True)),
        isolated_token=bool(config.get("isolated_token", False)),
        identity_modulation=bool(config.get("identity_modulation", False)),
        n_summary_tokens=int(config.get("n_summary_tokens", 1)),
        global_head=str(config.get("global_head", "film")),
        fusion_residual=bool(config.get("fusion_residual", False)),
        neighbourhood_mask_share=float(config.get("neighbourhood_mask_share", 0.0)),
        omics_mask_prob=float(config.get("omics_mask_prob", 0.0)),
        n_programs=int(config.get("n_programs", 0)),
        program_width=int(config.get("program_width", 32)),
        program_layers=int(config.get("program_layers", 2)),
        program_input=str(config.get("program_input", "final")),
    )


def load_archived_encoder(
    run_dir: str | Path, device: torch.device | str = "cpu"
) -> ArchivedEncoder:
    """Read an archived run and rebuild everything the probe needs.

    Args:
        run_dir: Directory holding the encoder, graph tensors and probe inputs.
        device: Where the model and tensors should live.

    Returns:
        The loaded run.

    Raises:
        FileNotFoundError: If the directory is missing one of the three files.
    """
    root = Path(run_dir)
    missing = [
        name
        for name in (ENCODER_FILE, GRAPH_FILE, PROBE_INPUTS_FILE)
        if not (root / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{root} is missing: {missing}")

    state_dict, config = read_checkpoint(root / ENCODER_FILE)
    graph = torch.load(root / GRAPH_FILE, map_location="cpu", weights_only=False)
    probe = torch.load(root / PROBE_INPUTS_FILE, map_location="cpu", weights_only=False)

    gene_order = [str(gene) for gene in graph["gene_order"]]
    inputs = probe["inputs"].float()
    if inputs.shape[1] != len(gene_order):
        raise ValueError(
            f"probe inputs carry {inputs.shape[1]} genes but the graph carries "
            f"{len(gene_order)}; these artifacts are from different runs"
        )

    torch_device = torch.device(device)
    identity_table = state_dict.get("lifter.gene_emb.weight")
    model = encoder_from_settings(
        config,
        len(gene_order),
        0 if identity_table is None else int(identity_table.shape[0]),
    )
    load_encoder_state(model, state_dict, strict=True)
    model.eval().to(torch_device)

    grn = graph.get("grn")
    logger.info(
        "loaded %s: %d patients x %d genes, %d layers, fusion %s",
        root.name,
        inputs.shape[0],
        inputs.shape[1],
        config["global_layers"],
        config.get("fusion_type"),
    )
    return ArchivedEncoder(
        model=model,
        inputs=inputs.to(torch_device),
        gene_order=gene_order,
        graph_pe=graph["graph_pe"].float().to(torch_device),
        spd=graph["spd"].to(torch_device),
        grn=None if grn is None else grn.float().to(torch_device),
        config=config,
    )


def permute_regulatory_graph(grn: torch.Tensor, seed: int = 20240712) -> torch.Tensor:
    """Shuffle which gene each regulatory edge points at.

    The decisive control for whether the graph is doing any work. Permuting
    node identities keeps the edge count, the sign balance and the degree
    distribution intact while destroying the correspondence to biology, so
    whatever effect survives was never graph-routed in the first place.

    Args:
        grn: Signed regulatory adjacency.
        seed: Seed for the permutation.

    Returns:
        The same matrix with rows and columns jointly permuted.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(grn.shape[0], generator=generator).to(grn.device)
    return grn[order][:, order]


def run_metabric_trans_sweep(
    run_dir: str | Path,
    output_dir: str | Path,
    device: torch.device | str | None = None,
    mask_modes: Sequence[str] = MASK_MODES,
    with_permuted_null: bool = True,
    seed: int = 20240712,
    max_factors: int | None = None,
) -> dict[str, Any]:
    """Measure trans effects on the external cohort, then test their signs.

    Args:
        run_dir: Archived self-supervised run to probe.
        output_dir: Directory to write the sweep and its tests into.
        device: Torch device, or None to detect.
        mask_modes: Masking modes to sweep.
        with_permuted_null: Also sweep a graph with permuted node identities.
        seed: Seed for the permutation and the bootstrap.
        max_factors: Probe only the first this-many factors, for smoke tests.
            Recorded in the output so a truncated sweep stays identifiable.

    Returns:
        A record naming what was written, including the corrected sign test.

    Raises:
        ValueError: If the archived run has no regulatory graph to probe.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("probing on %s", resolved)

    archived = load_archived_encoder(run_dir, resolved)
    if archived.grn is None:
        raise ValueError(
            f"{run_dir} carries no regulatory graph, so trans has no edges"
        )

    frames: list[pd.DataFrame] = []
    for mask_mode in mask_modes:
        logger.info("sweeping mask mode %s", mask_mode)
        frames.append(
            run_trans_sweep(
                archived.model,
                archived.inputs,
                archived.gene_order,
                archived.grn,
                archived.graph_pe,
                archived.spd,
                resolved,
                mask_mode=mask_mode,
                null_label=REAL_LABEL,
                seed=seed,
                max_factors=max_factors,
            )
        )

    if with_permuted_null:
        permuted = permute_regulatory_graph(archived.grn, seed)
        logger.info("sweeping the shuffled regulatory graph")
        frames.append(
            run_trans_sweep(
                archived.model,
                archived.inputs,
                archived.gene_order,
                permuted,
                archived.graph_pe,
                archived.spd,
                resolved,
                mask_mode=mask_modes[0],
                null_label=PERMUTED_LABEL,
                seed=seed,
                max_factors=max_factors,
            )
        )

    sweep = pd.concat(frames, ignore_index=True)
    edges_csv = output / "trans_edges.csv"
    sweep.to_csv(edges_csv, index=False)
    logger.info("wrote %d sweep rows to %s", len(sweep), edges_csv)

    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device": str(resolved),
        "n_patients": int(archived.inputs.shape[0]),
        "n_genes": len(archived.gene_order),
        "mask_modes": list(mask_modes),
        "with_permuted_null": with_permuted_null,
        "max_factors": max_factors,
        "truncated": max_factors is not None,
        "encoder_config": archived.config,
        "edges_csv": str(edges_csv),
        "sign_tests": rerun_sign_test(edges_csv, output, cohort="METABRIC"),
    }
    save_json(record, output / "metabric_trans.json")
    (output / "encoder_config.json").write_text(
        json.dumps(archived.config, indent=2) + "\n", encoding="utf-8"
    )
    return record
