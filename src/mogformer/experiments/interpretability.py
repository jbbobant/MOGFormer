"""What the encoder reads: attribution, omics reliance and gene programs.

Four views of one frozen encoder on one population:

* **risk attribution** — a ridge Cox model on the embedding gives each patient
  a relapse risk score; integrated gradients split that score over the
  inputs (gene by omics). It uses the outcome, so it is a post hoc analysis;
* **stratum attribution** — the same for a classifier separating the
  high-risk stratum from the rest, which uses no outcome;
* **omics reliance** — the fusion gate's weight on each omics, per stratum,
  beside each omics' share of each attribution;
* **gene programs** — the genes each program query attends to most, and how
  much relapse risk each program's block carries on its own (out-of-fold).

Every read-out is fitted on the same reduction of the embedding as the
stratification and survival analyses
(:func:`~mogformer.analysis.clustering.fit_reduction`), then mapped back to one
weight vector over the embedding, which is what attribution differentiates.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from mogformer.analysis.attribution import integrated_gradients
from mogformer.analysis.clustering import block_separation, fit_reduction
from mogformer.analysis.plots import (
    plot_attribution_beeswarm,
    plot_block_prognosis,
    plot_omics_reliance,
)
from mogformer.data.omics import MODALITY_ORDER
from mogformer.experiments.metabric import ArchivedEncoder, load_archived_encoder
from mogformer.experiments.outcome_analysis import (
    choose_ridge_penalty,
    fit_ridge_cox,
    outcome_frame,
    ridge_cox_out_of_fold,
)
from mogformer.experiments.patient_stratification import load_protocol
from mogformer.experiments.representation_evaluation import (
    metabric_context,
    tcga_context,
)
from mogformer.models.layers.program_readout import ProgramReadout

logger = logging.getLogger(__name__)

#: Width of one program block in the embedding.
PROGRAM_BLOCK_WIDTH = 32

#: Patients per forward pass when encoding a population.
ENCODE_BATCH = 32

#: Attribution columns reported in the record.
ATTRIBUTION_COLUMNS = (
    "gene",
    "omics",
    "mean_abs_attribution",
    "value_attribution_correlation",
)


@dataclass
class EncodedPopulation:
    """A population passed through the frozen encoder, nothing masked.

    Attributes:
        embedding: Patient embedding, patients x width.
        gates: Fusion gate weight per gene and omics, patients x genes x 3.
        fused: Fused per-gene tokens (what the program queries read),
            patients x genes x d.
    """

    embedding: np.ndarray
    gates: np.ndarray
    fused: torch.Tensor


@dataclass
class Survival:
    """Relapse follow-up of a population, restricted to complete rows.

    Attributes:
        complete: Rows with follow-up.
        time: Months to relapse or censoring, complete rows only.
        event: Relapse indicator, complete rows only.
    """

    complete: np.ndarray
    time: np.ndarray
    event: np.ndarray


def encode_population(
    encoder: ArchivedEncoder, inputs: torch.Tensor, device: torch.device
) -> EncodedPopulation:
    """Run a population through the frozen encoder and keep its internals.

    Args:
        encoder: The frozen encoder and its graph tensors.
        inputs: Standardised inputs of the population, patients x genes x 3.
        device: Where to compute.

    Returns:
        The embedding, the fusion gates and the fused gene tokens.
    """
    model = encoder.model.to(device).eval()
    embeddings: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    fused: list[torch.Tensor] = []
    hook = model.mini_transformer.register_forward_hook(
        lambda _module, _args, out: fused.append(out[0].detach().cpu())
    )
    graph = (
        encoder.graph_pe.to(device),
        encoder.spd.to(device),
        None if encoder.grn is None else encoder.grn.to(device),
    )
    with torch.no_grad():
        for start in range(0, len(inputs), ENCODE_BATCH):
            x = inputs[start : start + ENCODE_BATCH].to(device)
            out = model(x[..., 0], x[..., 1], x[..., 2], *graph, mask=False)
            embeddings.append(out["c"].cpu())
            gates.append(out["gate"].cpu())
    hook.remove()
    return EncodedPopulation(
        torch.cat(embeddings).numpy(), torch.cat(gates).numpy(), torch.cat(fused)
    )


def attribution_table(
    attributions: np.ndarray, inputs: np.ndarray, genes: list[str]
) -> pd.DataFrame:
    """Summarise attributions per (gene, omics).

    Args:
        attributions: Patients x genes x omics.
        inputs: The same patients' standardised inputs.
        genes: Gene names.

    Returns:
        One row per (gene, omics): mean |attribution|, mean attribution, and
        the correlation between the input value and its attribution (the
        sign of the effect: positive means higher values raise the score),
        sorted by mean |attribution|.
    """
    rows = []
    for k, omics in enumerate(MODALITY_ORDER):
        a, x = attributions[:, :, k], inputs[:, :, k]
        a_c, x_c = a - a.mean(axis=0), x - x.mean(axis=0)
        denominator = np.sqrt((a_c**2).sum(axis=0) * (x_c**2).sum(axis=0))
        direction = np.divide(
            (a_c * x_c).sum(axis=0),
            denominator,
            out=np.zeros(a.shape[1]),
            where=denominator > 0,
        )
        for g, gene in enumerate(genes):
            rows.append(
                {
                    "gene": gene,
                    "omics": omics,
                    "mean_abs_attribution": float(np.abs(a[:, g]).mean()),
                    "mean_attribution": float(a[:, g].mean()),
                    "value_attribution_correlation": float(direction[g]),
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values("mean_abs_attribution", ascending=False).reset_index(
        drop=True
    )


def read_out_direction(
    embedding: np.ndarray,
    variance: float,
    max_components: int,
    fit: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Fit a linear read-out on the reduced embedding; return it on the embedding.

    Args:
        embedding: Patients x width.
        variance: Variance the reduction's components must explain.
        max_components: Upper bound on the reduction's components.
        fit: From the component scores to one coefficient per component.

    Returns:
        One weight per embedding dimension.
    """
    reduction = fit_reduction(embedding, variance, max_components)
    return reduction.weights_on_features(fit(reduction.transform(embedding)))


def explain_score(
    name: str,
    direction: np.ndarray,
    encoder: ArchivedEncoder,
    inputs: torch.Tensor,
    genes: list[str],
    device: torch.device,
    steps: int,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    """Attribute one read-out to the inputs and summarise it.

    Args:
        name: Read-out name, used in file names.
        direction: Weights over the embedding.
        encoder: The frozen encoder.
        inputs: Standardised inputs of the population.
        genes: Gene names.
        device: Where to compute.
        steps: Integration points.

    Returns:
        The per-feature table, a summary for the record (omics shares,
        completeness gap, leading features), and the raw attributions.
    """
    attributions, gap = integrated_gradients(
        encoder.model,
        inputs,
        torch.as_tensor(direction, dtype=torch.float32),
        encoder.graph_pe,
        encoder.spd,
        encoder.grn,
        device,
        steps=steps,
    )
    table = attribution_table(attributions, inputs.numpy(), genes)
    totals = np.abs(attributions).sum(axis=(0, 1))
    summary: dict[str, object] = {
        "omics_share": {
            omics: float(totals[k] / totals.sum())
            for k, omics in enumerate(MODALITY_ORDER)
        },
        "median_abs_completeness_gap": float(np.median(np.abs(gap))),
        "top_features": table.head(15)[list(ATTRIBUTION_COLUMNS)].to_dict(
            orient="records"
        ),
    }
    logger.info("%s attribution: omics shares %s", name, summary["omics_share"])
    return table, summary, attributions


def gates_by_stratum(gates: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    """Mean fusion gate weight per omics, per stratum and overall.

    Args:
        gates: Patients x genes x 3.
        strata: Stratum per patient.

    Returns:
        One row per stratum plus ``all``; one column per omics.
    """
    table = pd.DataFrame(
        {o: gates[:, :, k].mean(axis=1) for k, o in enumerate(MODALITY_ORDER)}
    )
    summary = table.groupby(strata).mean()
    summary.loc["all"] = table.mean()
    return summary


def program_summary(
    readout: ProgramReadout,
    encoded: EncodedPopulation,
    genes: list[str],
    risk_table: pd.DataFrame,
    strata: np.ndarray,
    survival: Survival,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    """Describe each gene program: what it reads, what it predicts, what it splits.

    Args:
        readout: The encoder's program read-out.
        encoded: The population's embedding and fused tokens.
        genes: Gene names.
        risk_table: Per-feature risk attribution (see :func:`attribution_table`).
        strata: Stratum per patient.
        survival: Relapse follow-up.
        device: Where to compute.
        seed: Seed of the cross-validation.

    Returns:
        One row per program: its size, its most attended genes, its genes
        with the most risk attribution, its block's out-of-fold relapse
        C-index (mean and SD) and its between-strata variance share.
    """
    weights = []
    with torch.no_grad():
        for start in range(0, len(encoded.fused), ENCODE_BATCH):
            batch = encoded.fused[start : start + ENCODE_BATCH].to(device)
            weights.append(readout.attention_weights(batch).cpu())
    attention = torch.cat(weights).mean(dim=0).numpy()
    assignment = readout.assignment.cpu().numpy()
    risk = risk_table.groupby("gene")["mean_abs_attribution"].sum()
    separation: dict[int, float] = dict(
        block_separation(encoded.embedding, strata, PROGRAM_BLOCK_WIDTH)[
            ["block", "between_share"]
        ].itertuples(index=False, name=None)
    )
    rows = []
    for k in range(readout.n_programs):
        members = np.flatnonzero(assignment == k)
        attended = members[np.argsort(-attention[k, members])]
        block = encoded.embedding[
            :, k * PROGRAM_BLOCK_WIDTH : (k + 1) * PROGRAM_BLOCK_WIDTH
        ]
        block = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-9)
        c_index = ridge_cox_out_of_fold(
            pd.DataFrame(block[survival.complete]).add_prefix("b_"),
            survival.time,
            survival.event,
            n_splits=5,
            n_repeats=2,
            seed=seed,
        )
        member_genes = [genes[g] for g in members]
        rows.append(
            {
                "program": k,
                "n_genes": len(members),
                "top_attended_genes": ", ".join(genes[g] for g in attended[:10]),
                "top_risk_genes": ", ".join(
                    risk.reindex(member_genes).sort_values(ascending=False).index[:6]
                ),
                "relapse_c_index": float(np.mean(c_index)),
                "relapse_c_index_sd": float(np.std(c_index, ddof=1)),
                "strata_separation": separation[k],
            }
        )
    return pd.DataFrame(rows)


def run_interpretability(
    protocol_path: str | Path,
    results_root: str | Path,
    stratification_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    steps: int = 32,
) -> dict[str, object]:
    """Interpret the discovery cohort's primary encoder on its population.

    Args:
        protocol_path: The stratification ``protocol.yaml`` naming the runs.
        results_root: Directory holding the pretraining runs.
        stratification_dir: Output of the stratification whose strata are
            used.
        output_dir: Where outputs are written.
        seed: Seed of the resampling steps.
        steps: Integration points of integrated gradients.

    Returns:
        The record: the strata, the read-outs' attribution summaries, the
        fusion gates and the program table.

    Raises:
        ValueError: If the encoder has no program read-out.
    """
    protocol = load_protocol(protocol_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    population = protocol.discovery_population
    cohort = next(
        name for name, s in protocol.cohorts.items() if population in s["populations"]
    )
    settings = protocol.cohorts[cohort]
    run_dir = Path(results_root) / settings["runs"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_archived_encoder(run_dir, device)
    readout = encoder.model.program_readout
    if readout is None:
        raise ValueError(f"{run_dir} has no program read-out to interpret")
    probe = torch.load(run_dir / "probe_inputs.pt", weights_only=False)
    patients = [str(p) for p in probe["patient_ids"]]
    bundle = Path(settings["bundle"])
    context = (tcga_context if cohort == "tcga" else metabric_context)(
        patients, bundle, seed
    )
    rows = context.populations[population]
    inputs = encoder.inputs[rows].cpu()
    genes = list(encoder.gene_order)
    encoded = encode_population(encoder, inputs, device)

    population_patients = list(np.asarray(patients)[rows])
    strata_table = pd.read_csv(Path(stratification_dir) / "strata.csv")
    strata = (
        strata_table[strata_table["population"] == population]
        .set_index("patient")
        .loc[population_patients, "stratum"]
        .to_numpy(int)
    )
    # Outcome-blind choice of the stratum to explain: the most proliferative.
    proliferation = context.covariates["proliferation"].to_numpy()[rows]
    high_risk = int(pd.Series(proliferation).groupby(strata).mean().idxmax())

    clinical = pd.read_csv(bundle / "clinical.csv", index_col=0)
    frame = outcome_frame(
        clinical, population_patients, context.covariates["proliferation"]
    )
    complete = frame["rfs_months"].notna().to_numpy()
    survival = Survival(
        complete,
        frame["rfs_months"].to_numpy()[complete],
        frame["rfs_event"].to_numpy()[complete],
    )

    def cox_fit(scores: np.ndarray) -> np.ndarray:
        features = pd.DataFrame(scores[survival.complete]).add_prefix("pc_")
        penalty = choose_ridge_penalty(features, survival.time, survival.event, seed)
        model = fit_ridge_cox(features, survival.time, survival.event, penalty)
        return np.asarray(model.params_.to_numpy())

    def stratum_fit(scores: np.ndarray) -> np.ndarray:
        classifier = LogisticRegression(C=1.0, max_iter=2000)
        classifier.fit(scores, (strata == high_risk).astype(int))
        return np.asarray(classifier.coef_[0])

    record: dict[str, object] = {
        "population": population,
        "encoder_run": settings["runs"][0],
        "n_patients": len(rows),
        "strata_sizes": np.bincount(strata).tolist(),
        "high_risk_stratum": high_risk,
        "post_hoc": "risk attribution and program prognosis use relapse outcomes",
    }
    reads: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str, str, str]] = {
        "risk": (
            cox_fit,
            f"{population}: attribution of the relapse risk score (post hoc)",
            "lowers risk",
            "raises risk",
        ),
        "stratum": (
            stratum_fit,
            f"{population}: attribution of stratum {high_risk} vs the rest",
            "away from the stratum",
            "towards the stratum",
        ),
    }
    tables: dict[str, pd.DataFrame] = {}
    shares: dict[str, dict[str, float]] = {}
    for name, (fit, title, negative, positive) in reads.items():
        direction = read_out_direction(
            encoded.embedding, protocol.variance_explained, protocol.max_components, fit
        )
        table, summary, attributions = explain_score(
            name, direction, encoder, inputs, genes, device, steps
        )
        table.to_csv(output / f"{name}_attribution.csv", index=False)
        tables[name] = table
        shares[name] = dict(summary["omics_share"])  # type: ignore[call-overload]
        record[f"{name}_attribution"] = summary
        plot_attribution_beeswarm(
            attributions,
            inputs.numpy(),
            genes,
            output,
            name=f"{name}_attribution",
            title=title,
            negative_label=negative,
            positive_label=positive,
        )

    gates = gates_by_stratum(encoded.gates, strata)
    gates.to_csv(output / "fusion_gates_by_stratum.csv")
    record["fusion_gates"] = gates.round(4).to_dict(orient="index")
    plot_omics_reliance(
        gates, shares, output, title=f"{population}: which omics the encoder relies on"
    )

    programs = program_summary(
        readout, encoded, genes, tables["risk"], strata, survival, device, seed
    )
    programs.to_csv(output / "gene_programs.csv", index=False)
    record["programs"] = programs.round(4).to_dict(orient="records")
    plot_block_prognosis(
        programs,
        output,
        title=f"{population}: relapse risk carried by each gene program (post hoc)",
    )
    (output / "record.json").write_text(
        json.dumps(record, indent=2, default=float), encoding="utf-8"
    )
    return record
