"""Can in-silico perturbation recover regulatory signs when they are true?

On real tumours the trans probe does not recover CollecTRI's edge signs. That
failure has two readings: the probe's estimand cannot identify a sign from
observational data, or the curated signs are simply not expressed in bulk
tumours. This benchmark separates them. Cohorts are simulated from a structural
causal model built on the same curated edges (:mod:`mogformer.simulation.scm`),
so every sign is true in the data. The same encoder, trained with the same
settings on the same gene axis and graph tensors, is probed exactly as the real
one was, and scored with the same tests.

Every probe estimate is set beside three comparators on identical pairs —
observed co-expression, a ridge regression on everything left visible, and a
copy-number instrument — and against two truths: the curated sign, which is the
sign of the direct effect by construction, and the exact interventional effect
obtained by clamping the factor and re-solving the system.

The sweep crosses sample size, hidden confounding and whether the encoder is
given the regulatory graph.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from mogformer.analysis.effect_estimators import (
    ridge_conditional_effects,
    select_ridge_penalty,
    wald_ratio_effects,
)
from mogformer.analysis.probe_cis import DEFAULT_GRID
from mogformer.analysis.probe_trans import (
    permuted_graph_retention,
    run_trans_sweep,
    sign_permutation_test,
)
from mogformer.experiments.metabric import (
    GRAPH_FILE,
    PROBE_INPUTS_FILE,
    encoder_from_settings,
    permute_regulatory_graph,
)
from mogformer.experiments.paths import save_json
from mogformer.experiments.sign_test import REAL_LABEL, load_edge_table
from mogformer.models.checkpoints import read_checkpoint
from mogformer.simulation.scm import (
    GeneRegulatorySCM,
    SimulatedCohort,
    SimulationSettings,
)
from mogformer.training.seed import seed_everything
from mogformer.training.trainer import (
    ARCHIVED_GATE_LR_MULTIPLIER,
    MaskedReconstructionTrainer,
)

logger = logging.getLogger(__name__)

#: Estimates compared on every pair, by column name and display name.
ESTIMATORS: dict[str, str] = {
    "slope": "MOGFormer probe",
    "observed_coexpression": "co-expression",
    "ridge_effect": "ridge, all visible inputs",
    "instrumental_effect": "copy-number instrument",
}

#: Label of the probe sweep run on a regulatory graph with shuffled identities.
REGULATORY_NULL_LABEL = "permute_grn_only"


@dataclass(frozen=True)
class BenchmarkCondition:
    """One cell of the sweep.

    Attributes:
        n_patients: Simulated cohort size.
        confounding: Hidden-program share of exogenous variance.
        use_grn: Whether the encoder receives the regulatory graph.
    """

    n_patients: int
    confounding: float
    use_grn: bool

    @property
    def name(self) -> str:
        """Directory name stating the condition's three settings."""
        graph = "with_grn" if self.use_grn else "without_grn"
        return f"n{self.n_patients}_confounding{self.confounding:.1f}_{graph}"


@dataclass(frozen=True)
class BenchmarkPlan:
    """What the benchmark simulates, trains and probes.

    Attributes:
        simulation: Keyword arguments of :class:`SimulationSettings`, without
            the swept fields.
        conditions: Cells of the sweep, in run order.
        probe_patients: Patients the probe and the truth are averaged over.
        n_boot: Bootstrap resamples behind each probe slope's interval.
        n_permutations: Label shuffles behind each sign test.
        monitor_frac: Share of patients held out for early stopping.
        real_cohorts: ``(name, encoder, sweep CSV)`` per real cohort scored
            alongside, the encoder named as in the bundle's encoder directory.
        step_budget_patients: Cohort size at which the archived epoch budget
            and patience apply unchanged. Larger cohorts get proportionally
            fewer epochs, so every condition takes the same number of optimiser
            steps and sample size is not confounded with training compute.
            None keeps the archived epoch budget everywhere.
    """

    simulation: Mapping[str, Any]
    conditions: tuple[BenchmarkCondition, ...]
    probe_patients: int
    n_boot: int
    n_permutations: int
    monitor_frac: float
    real_cohorts: tuple[tuple[str, str, str], ...] = ()
    step_budget_patients: int | None = None


def load_benchmark_plan(path: Path | str) -> BenchmarkPlan:
    """Read the sweep from YAML.

    Args:
        path: A file with ``simulation``, ``conditions`` and ``probe`` keys.

    Returns:
        The plan.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    conditions = tuple(
        BenchmarkCondition(
            n_patients=int(cell["n_patients"]),
            confounding=float(cell["confounding"]),
            use_grn=bool(cell["use_grn"]),
        )
        for cell in raw["conditions"]
    )
    return BenchmarkPlan(
        simulation=dict(raw.get("simulation", {})),
        conditions=conditions,
        probe_patients=int(raw["probe"]["patients"]),
        n_boot=int(raw["probe"].get("n_boot", 1000)),
        n_permutations=int(raw["probe"]["n_permutations"]),
        monitor_frac=float(raw.get("monitor_frac", 0.12)),
        step_budget_patients=raw.get("step_budget_patients"),
        real_cohorts=tuple(
            (str(entry["name"]), str(entry["encoder"]), str(entry["sweep"]))
            for entry in raw.get("real_cohorts", [])
        ),
    )


#: Checkpoint of a condition's trained encoder, so a restart skips training.
ENCODER_CHECKPOINT = "encoder.pt"


def _encoder_for(
    settings: Mapping[str, Any], n_genes: int, use_grn: bool
) -> torch.nn.Module:
    """Build the archived architecture with the regulatory bias switched as asked."""
    relations = [r for r in settings.get("graph_relations", ["ppi"]) if r != "grn"]
    model_settings = {
        **settings,
        "graph_relations": relations + (["grn"] if use_grn else []),
    }
    return encoder_from_settings(model_settings, n_genes, n_genes)


def train_encoder(
    inputs: np.ndarray,
    settings: Mapping[str, Any],
    graph: Mapping[str, torch.Tensor],
    use_grn: bool,
    device: torch.device,
    monitor_frac: float,
    max_epochs: int | None = None,
    epoch_scale: float = 1.0,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Pretrain an encoder on a simulated cohort, exactly as on a real one.

    Args:
        inputs: Standardised values, shape ``(n_patients, n_genes, 3)``.
        settings: The archived encoder's settings, reused unchanged.
        graph: ``graph_pe``, ``spd`` and ``grn`` for the gene axis.
        use_grn: Give the encoder the regulatory attention bias.
        device: Where to train.
        monitor_frac: Share of patients held out for early stopping.
        max_epochs: Override of the epoch budget, for smoke tests only.
        epoch_scale: Factor applied to the archived epoch budget and patience.

    Returns:
        The frozen encoder and a record of its training.
    """
    seed = int(settings.get("seed", 42))
    seed_everything(seed)
    n_patients, n_genes, _ = inputs.shape
    model = _encoder_for(settings, n_genes, use_grn).to(device)

    order = np.random.default_rng(seed).permutation(n_patients)
    n_monitor = max(round(monitor_frac * n_patients), 1)
    tensors = [torch.as_tensor(inputs[..., channel]) for channel in range(3)]
    dataset = TensorDataset(*tensors)
    batch_size = int(settings["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, order[n_monitor:].tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    monitor_loader = DataLoader(
        torch.utils.data.Subset(dataset, order[:n_monitor].tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    trainer = MaskedReconstructionTrainer(
        model,
        train_loader,
        monitor_loader,
        device,
        graph["spd"],
        graph["graph_pe"],
        graph["grn"],
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
        gate_lr_multiplier=float(
            settings.get("gate_lr_multiplier", ARCHIVED_GATE_LR_MULTIPLIER)
        ),
    )
    started = time.perf_counter()
    budget = int(np.ceil(epoch_scale * int(settings["max_epochs"])))
    patience = int(np.ceil(epoch_scale * int(settings["patience"])))
    best = trainer.fit_early(
        budget if max_epochs is None else max_epochs,
        patience,
        int(settings.get("log_every", 10)),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    record = {
        "best_monitor_loss": float(best),
        "epochs": len(trainer.history.val),
        "epoch_budget": budget,
        "patience": patience,
        "optimiser_steps": len(trainer.history.val) * len(train_loader),
        "minutes": (time.perf_counter() - started) / 60.0,
    }
    logger.info(
        "trained %d epochs in %.1f min, best monitoring loss %.4f",
        record["epochs"],
        record["minutes"],
        best,
    )
    return model, record


def true_effects(
    scm: GeneRegulatorySCM,
    cohort: SimulatedCohort,
    factors: np.ndarray,
    targets: np.ndarray,
    max_patients: int,
) -> np.ndarray:
    """Exact interventional slope for every pair, one clamped solve per factor.

    Args:
        scm: The causal system the cohort was drawn from.
        cohort: The cohort, whose exogenous inputs the interventions reuse.
        factors: Factor gene index per pair.
        targets: Target gene index per pair.
        max_patients: Patients averaged over.

    Returns:
        The true effect per pair, in the probe's standardised units.
    """
    effects = np.empty(len(factors))
    for factor in np.unique(factors):
        pairs = np.nonzero(factors == factor)[0]
        effects[pairs] = scm.interventional_slopes(
            cohort, int(factor), targets[pairs], DEFAULT_GRID, max_patients
        )
    return effects


def score_estimates(
    pairs: pd.DataFrame, n_permutations: int, seed: int
) -> list[dict[str, Any]]:
    """Score every estimator against the curated sign and the true effect.

    Args:
        pairs: Probed pairs with every estimator's column, and ``true_effect``
            when a truth exists; without it the true-sign rate is NaN.
        n_permutations: Label shuffles behind each sign test.
        seed: Seed for the shuffling.

    Returns:
        One record per estimator.
    """
    edges = pairs[pairs["is_edge"] == 1]
    controls = pairs[pairs["is_edge"] == 0]
    truth_sign = (
        np.sign(edges["true_effect"].to_numpy()) if "true_effect" in edges else None
    )
    rows = []
    for column, label in ESTIMATORS.items():
        estimate = edges[column].to_numpy(dtype=float)
        usable = np.isfinite(estimate)
        test = sign_permutation_test(
            pd.DataFrame(
                {
                    "edge_sign": edges["edge_sign"].to_numpy()[usable],
                    "slope": estimate[usable],
                }
            ),
            n_perm=n_permutations,
            seed=seed,
        )
        control_magnitude = float(
            np.nanmedian(np.abs(controls[column].to_numpy(dtype=float)))
        )
        rows.append(
            {
                "estimator": label,
                "column": column,
                "n_edges": int(usable.sum()),
                "curated_sign_rate": test.rate,
                "curated_sign_null": test.null_mean_rate,
                "curated_sign_p": test.perm_p,
                "trivial_majority_baseline": test.trivial_majority_baseline,
                "is_informative": test.is_informative,
                "true_effect_sign_rate": float(
                    (np.sign(estimate[usable]) == truth_sign[usable]).mean()
                )
                if truth_sign is not None
                else float("nan"),
                "edge_to_control_magnitude": float(np.nanmedian(np.abs(estimate)))
                / control_magnitude
                if control_magnitude > 0
                else float("nan"),
            }
        )
    return rows


def run_condition(
    condition: BenchmarkCondition,
    plan: BenchmarkPlan,
    settings: Mapping[str, Any],
    graph: Mapping[str, Any],
    output: Path,
    device: torch.device,
    seed: int,
    max_epochs: int | None = None,
    max_factors: int | None = None,
) -> dict[str, Any]:
    """Simulate, train, probe and score one cell of the sweep.

    Args:
        condition: The cell.
        plan: The whole benchmark, for its shared settings.
        settings: The archived encoder's settings.
        graph: Graph tensors and gene order of the real gene axis.
        output: Directory for this cell's tables.
        device: Where to train and probe.
        seed: Seed of the causal structure, shared by every cell.
        max_epochs: Epoch override, for smoke tests only.
        max_factors: Probe only this many factors, for smoke tests only.

    Returns:
        The cell's record, also written as ``summary.json``.
    """
    output.mkdir(parents=True, exist_ok=True)
    grn = graph["grn"]
    scm = GeneRegulatorySCM(
        grn.cpu().numpy(),
        SimulationSettings(
            n_patients=condition.n_patients,
            confounding=condition.confounding,
            seed=seed,
            **plan.simulation,
        ),
        device=device,
    )
    cohort = scm.sample()
    epoch_scale = (
        1.0
        if plan.step_budget_patients is None
        else min(1.0, plan.step_budget_patients / condition.n_patients)
    )
    checkpoint = output / ENCODER_CHECKPOINT
    if checkpoint.exists():
        # The cohort is a deterministic draw from the seed, so a saved encoder
        # was trained on exactly these patients.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _encoder_for(settings, cohort.inputs.shape[1], condition.use_grn)
        model.load_state_dict(saved["state_dict"])
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        training = saved["training"]
        logger.info("restored the trained encoder from %s", checkpoint)
    else:
        model, training = train_encoder(
            cohort.inputs,
            settings,
            graph,
            condition.use_grn,
            device,
            plan.monitor_frac,
            max_epochs,
            epoch_scale,
        )
        torch.save({"state_dict": model.state_dict(), "training": training}, checkpoint)

    probe_inputs = torch.as_tensor(cohort.inputs[: plan.probe_patients]).to(device)
    gene_order = list(graph["gene_order"])
    sweep = run_trans_sweep(
        model,
        probe_inputs,
        gene_order,
        grn,
        graph["graph_pe"],
        graph["spd"],
        device,
        mask_mode="whole_gene",
        n_boot=plan.n_boot,
        seed=seed,
        max_factors=max_factors,
    )
    position = {gene: i for i, gene in enumerate(gene_order)}
    factors = sweep["tf"].map(position).to_numpy()
    targets = sweep["target"].map(position).to_numpy()
    probe_values = cohort.inputs[: plan.probe_patients]
    penalty = select_ridge_penalty(probe_values, seed=seed)
    sweep["ridge_effect"] = ridge_conditional_effects(
        probe_values, factors, targets, penalty
    )
    sweep["instrumental_effect"], sweep["first_stage_f"] = wald_ratio_effects(
        probe_values, factors, targets
    )
    sweep["true_effect"] = true_effects(
        scm, cohort, factors, targets, plan.probe_patients
    )
    sweep.to_csv(output / "pairs.csv", index=False)

    record: dict[str, Any] = {
        "condition": condition.name,
        "n_patients": condition.n_patients,
        "confounding": condition.confounding,
        "use_grn": condition.use_grn,
        "probe_patients": plan.probe_patients,
        "ridge_penalty": penalty,
        "training": training,
        "truncated": max_epochs is not None or max_factors is not None,
        "scores": score_estimates(sweep, plan.n_permutations, seed),
    }

    if condition.use_grn:
        shuffled = run_trans_sweep(
            model,
            probe_inputs,
            gene_order,
            permute_regulatory_graph(grn, seed),
            graph["graph_pe"],
            graph["spd"],
            device,
            mask_mode="whole_gene",
            null_label=REGULATORY_NULL_LABEL,
            n_boot=plan.n_boot,
            seed=seed,
            max_factors=max_factors,
        )
        shuffled.to_csv(output / "pairs_shuffled_grn.csv", index=False)
        record["regulatory_graph_retention"] = permuted_graph_retention(
            sweep[sweep["is_edge"] == 1], shuffled[shuffled["is_edge"] == 1]
        )

    save_json(record, output / "summary.json")
    return record


def score_real_cohort(
    encoder_dir: Path | str,
    sweep_csv: Path | str,
    cohort: str,
    n_permutations: int,
    seed: int = 20240712,
) -> dict[str, Any]:
    """Score the same estimators on a real cohort, where only curation is known.

    The probe's slopes come from the cohort's stored sweep; the comparators are
    computed on the probe inputs the encoder was probed with, on the same pairs.

    Args:
        encoder_dir: Archived encoder holding the probe inputs and gene order.
        sweep_csv: The cohort's per-pair trans sweep.
        cohort: Cohort name, recorded in the output.
        n_permutations: Label shuffles behind each sign test.
        seed: Seed for the shuffling and the penalty search.

    Returns:
        A record shaped like a simulated condition's.
    """
    root = Path(encoder_dir)
    probe = torch.load(root / PROBE_INPUTS_FILE, map_location="cpu", weights_only=False)
    values = probe["inputs"].double().numpy()
    values = np.nan_to_num(
        (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-12)
    )
    position = {str(gene): i for i, gene in enumerate(probe["gene_order"])}

    sweep = load_edge_table(sweep_csv)
    pairs = sweep[
        (sweep["null"] == REAL_LABEL) & (sweep["mask_mode"] == "whole_gene")
    ].copy()
    factors = pairs["tf"].map(position).to_numpy()
    targets = pairs["target"].map(position).to_numpy()
    penalty = select_ridge_penalty(values, seed=seed)
    pairs["ridge_effect"] = ridge_conditional_effects(values, factors, targets, penalty)
    pairs["instrumental_effect"], pairs["first_stage_f"] = wald_ratio_effects(
        values, factors, targets
    )
    shuffled = sweep[
        (sweep["null"] == REGULATORY_NULL_LABEL)
        & (sweep["mask_mode"] == "whole_gene")
        & (sweep["is_edge"] == 1)
    ]
    record: dict[str, Any] = {
        "condition": f"{cohort} (observed)",
        "n_patients": int(values.shape[0]),
        "confounding": float("nan"),
        "use_grn": True,
        "ridge_penalty": penalty,
        "scores": score_estimates(pairs, n_permutations, seed),
    }
    if not shuffled.empty:
        record["regulatory_graph_retention"] = permuted_graph_retention(
            pairs[pairs["is_edge"] == 1], shuffled
        )
    return record


def summarise_benchmark(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten per-condition records into one row per condition and estimator.

    Args:
        records: Outputs of :func:`run_condition`.

    Returns:
        The long table the figures read.
    """
    rows = []
    for record in records:
        retention = record.get("regulatory_graph_retention", {})
        training = record.get("training", {})
        for score in record["scores"]:
            rows.append(
                {
                    "condition": record["condition"],
                    "n_patients": record["n_patients"],
                    "confounding": record["confounding"],
                    "use_grn": record["use_grn"],
                    "encoder_monitor_loss": training.get("best_monitor_loss"),
                    "encoder_epochs": training.get("epochs"),
                    **score,
                    "grn_attributable_pct": retention.get("graph_attributable_pct")
                    if score["column"] == "slope"
                    else None,
                }
            )
    return pd.DataFrame(rows)


def run_simulation_benchmark(
    plan_path: Path | str,
    encoder_dir: Path | str,
    output_dir: Path | str,
    device: torch.device | str | None = None,
    only: Sequence[str] | None = None,
    seed: int = 20240712,
    max_epochs: int | None = None,
    max_factors: int | None = None,
) -> pd.DataFrame:
    """Run every cell of the sweep that has not already been run.

    Cells already holding a ``summary.json`` are read, not rerun, so an
    interrupted sweep resumes where it stopped.

    Args:
        plan_path: The sweep's YAML.
        encoder_dir: Archived encoder whose settings and graph are reused.
        output_dir: Root for one directory per cell and the summary table.
        device: Torch device, or None to detect.
        only: Names of cells to run; all when None.
        seed: Seed of the causal structure, shared by every cell.
        max_epochs: Epoch override, for smoke tests only.
        max_factors: Factor cap for the probe, for smoke tests only.

    Returns:
        The summary table, also written as ``summary.csv``.
    """
    plan = load_benchmark_plan(plan_path)
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    root = Path(encoder_dir)
    _, settings = read_checkpoint(root / "encoder_frozen.pt")
    tensors = torch.load(root / GRAPH_FILE, map_location="cpu", weights_only=False)
    graph = {
        "graph_pe": tensors["graph_pe"].float().to(resolved),
        "spd": tensors["spd"].to(resolved),
        "grn": tensors["grn"].float().to(resolved),
        "gene_order": [str(gene) for gene in tensors["gene_order"]],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for condition in plan.conditions:
        if only and condition.name not in only:
            continue
        summary = output / condition.name / "summary.json"
        if summary.exists():
            logger.info("%s already run; reading it", condition.name)
            records.append(json.loads(summary.read_text(encoding="utf-8")))
            continue
        logger.info("running %s on %s", condition.name, resolved)
        records.append(
            run_condition(
                condition,
                plan,
                settings,
                graph,
                output / condition.name,
                resolved,
                seed,
                max_epochs,
                max_factors,
            )
        )

    for name, encoder, sweep_csv in plan.real_cohorts:
        cached = output / f"observed_{encoder}.json"
        if cached.exists():
            records.append(json.loads(cached.read_text(encoding="utf-8")))
            continue
        logger.info("scoring the observed %s cohort", name)
        observed = score_real_cohort(
            root.parent / encoder, sweep_csv, name, plan.n_permutations, seed
        )
        save_json(observed, cached)
        records.append(observed)

    table = summarise_benchmark(records)
    table.to_csv(output / "summary.csv", index=False)
    save_json(
        {"plan": str(plan_path), "encoder_settings": settings, "seed": seed},
        output / "provenance.json",
    )
    return table
