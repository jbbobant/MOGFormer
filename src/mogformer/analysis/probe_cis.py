"""Interventional probing of a gene's own regulatory wiring.

Attention weights say what a model looked at, not what it concluded. This module
asks a causal question of the frozen encoder instead: set a gene's methylation
to a value, hide that same gene's expression, and read what expression the model
now predicts. Sweeping the injected value traces a response curve, and its slope
is the coupling the model learned.

**Sign convention, fixed throughout.** Methylation up, predicted expression
down, is silencing — so a negative slope means the model learned that promoter
methylation suppresses expression. Every statistic here inherits that sign, and
a sign-gate check on genes with known silencing behaviour should be run before
any of it is interpreted.

**The mask is deterministic.** Exactly one entry is hidden — the probed gene's
expression — using an explicitly constructed mask rather than the stochastic
masker used in training. If the training masker were used the hidden set would
vary between calls and the response curve would measure noise.

**Batching is an approximation, and it is measured.** Probing one gene per
forward pass is exact but slow. Genes are therefore grouped so that members of a
batch are far apart in the interaction graph and share no regulatory edge, which
makes cross-contamination unlikely — but not impossible, so
:func:`parity_check` quantifies the difference against single-gene probing and
reports it rather than assuming it away.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from mogformer.data.omics import MODALITY_ORDER

logger = logging.getLogger(__name__)

#: Injected values, in standard deviations of the per-gene standardised channel.
DEFAULT_GRID: np.ndarray = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.5), 3)

#: Column index of each modality inside the stacked probe input.
MODALITY_INDEX: dict[str, int] = {name: i for i, name in enumerate(MODALITY_ORDER)}

#: Minimum hop distance between two genes probed in the same forward pass.
MIN_BATCH_DISTANCE = 4


def make_gene_batches(
    spd: np.ndarray,
    gene_indices: Sequence[int],
    batch_size: int,
    min_distance: int = MIN_BATCH_DISTANCE,
    grn: np.ndarray | None = None,
) -> list[list[int]]:
    """Group genes that can be probed together without interfering.

    Two genes may share a forward pass only if they are at least
    ``min_distance`` hops apart in both directions — the unreachable bucket
    counts as far — and share no regulatory edge either way. Greedy assignment
    is enough: the goal is a large reduction in forward passes, not an optimal
    packing.

    Args:
        spd: Integer gene distances, shape ``(n_genes, n_genes)``.
        gene_indices: Genes to group.
        batch_size: Maximum genes per batch. One disables batching entirely.
        min_distance: Minimum hop distance between batch members.
        grn: Signed regulatory adjacency of the same shape, or None.

    Returns:
        Batches of gene indices, each of length at most ``batch_size``.
    """
    if batch_size <= 1:
        return [[gene] for gene in gene_indices]

    def is_far(gene: int, other: int) -> bool:
        too_close = spd[gene, other] < min_distance or spd[other, gene] < min_distance
        regulated = grn is not None and (grn[gene, other] != 0 or grn[other, gene] != 0)
        return not (too_close or regulated)

    batches: list[list[int]] = []
    for gene in gene_indices:
        for batch in batches:
            if len(batch) < batch_size and all(is_far(gene, other) for other in batch):
                batch.append(gene)
                break
        else:
            batches.append([gene])
    return batches


@torch.no_grad()
def probe_genes(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    gene_indices: Sequence[int],
    inject_modality: str,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    grid: np.ndarray = DEFAULT_GRID,
    patient_chunk: int = 256,
    cnv_diploid: torch.Tensor | None = None,
) -> dict[int, np.ndarray]:
    """Sweep an injected value and record the model's predicted expression.

    For every value on the grid the named modality of every probed gene is set
    to that value, those genes' expression is hidden, and the local
    reconstruction head's prediction for them is recorded.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)`` in
            modality order — exactly what the encoder consumed during training.
        gene_indices: Genes probed in this call, assumed mutually non-interfering.
        inject_modality: Modality to intervene on, one of
            :data:`~mogformer.data.omics.MODALITY_ORDER`.
        graph_pe: Positional encodings for the frozen gene axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device to run on.
        grid: Injected values, in standardised units.
        patient_chunk: Patients per forward pass, to bound memory.
        cnv_diploid: Standardised diploid copy number per gene. When given, the
            probed genes' copy number is pinned there, which isolates the
            methylation effect from a copy-number one.

    Returns:
        Mapping of gene index to its response, shape ``(len(grid), n_patients)``.

    Raises:
        ValueError: If ``inject_modality`` is unknown.
    """
    if inject_modality not in MODALITY_INDEX:
        raise ValueError(
            f"unknown modality {inject_modality!r}; expected one of "
            f"{list(MODALITY_INDEX)}"
        )

    n_patients, n_genes = inputs.shape[0], inputs.shape[1]
    responses = {
        gene: np.empty((len(grid), n_patients), dtype=np.float32)
        for gene in gene_indices
    }
    probed = torch.as_tensor(list(gene_indices), dtype=torch.long)
    rna_index = MODALITY_INDEX["rna"]

    for grid_position, value in enumerate(grid):
        for start in range(0, n_patients, patient_chunk):
            stop = min(start + patient_chunk, n_patients)
            chunk = inputs[start:stop].to(device)
            rna = chunk[..., MODALITY_INDEX["rna"]].clone()
            cnv = chunk[..., MODALITY_INDEX["cnv"]].clone()
            methy = chunk[..., MODALITY_INDEX["methy"]].clone()

            {"rna": rna, "cnv": cnv, "methy": methy}[inject_modality][:, probed] = (
                float(value)
            )
            if cnv_diploid is not None:
                cnv[:, probed] = cnv_diploid[probed].to(device)

            # Exactly one hidden entry per probed gene: its own expression.
            mask = torch.zeros(
                stop - start, n_genes, 3, dtype=torch.bool, device=device
            )
            mask[:, probed, rna_index] = True

            out = model(
                rna,
                cnv,
                methy,
                graph_pe,
                spd,
                grn,
                mask=True,
                mask_bool=mask,
                structural_bias=True,
            )
            predicted = out["xhat_l"][:, :, rna_index].float().cpu().numpy()
            for gene in gene_indices:
                responses[gene][grid_position, start:stop] = predicted[:, gene]

    return responses


class GridRankCorrelation:
    """Spearman correlation between the injected grid and the response.

    The grid ranks are the same for every gene and every bootstrap resample —
    each grid level always contributes the same number of patients — so they are
    computed once and reused. That is what makes the bootstrap affordable.

    Attributes:
        grid_ranks: Centred, unit-norm ranks of the repeated grid.
    """

    def __init__(self, n_patients: int, grid: np.ndarray = DEFAULT_GRID) -> None:
        """Precompute the constant grid ranks.

        Args:
            n_patients: Patients contributing at each grid level.
            grid: Injected values.
        """
        ranks = rankdata(np.repeat(grid, n_patients))
        centred = ranks - ranks.mean()
        self.grid_ranks = centred / np.sqrt((centred**2).sum())

    def __call__(self, response: np.ndarray) -> float:
        """Correlate one flattened response against the grid.

        Args:
            response: Flattened response of length ``len(grid) * n_patients``.

        Returns:
            Spearman correlation, or ``nan`` when the response is constant.
        """
        ranks = rankdata(response)
        centred = ranks - ranks.mean()
        norm = np.sqrt((centred**2).sum())
        return float(centred @ self.grid_ranks / norm) if norm > 1e-12 else float("nan")


def patient_slopes(response: np.ndarray, grid: np.ndarray = DEFAULT_GRID) -> np.ndarray:
    """Compute each patient's least-squares slope of response against the grid.

    These slopes are a sufficient statistic for the patient bootstrap: resampling
    patients and re-averaging the slopes is equivalent to refitting, so the
    intervals cost almost nothing.

    Args:
        response: Response of shape ``(len(grid), n_patients)``.
        grid: Injected values.

    Returns:
        Per-patient slope, shape ``(n_patients,)``.
    """
    centred = grid - grid.mean()
    weights = centred / (centred**2).sum()
    return weights @ response


@dataclass
class ResponseStatistics:
    """Summary of one gene's response curve.

    Attributes:
        slope: Mean per-patient slope; the primary effect size.
        slope_lo: Lower bound of the bootstrap interval.
        slope_hi: Upper bound of the bootstrap interval.
        frac_patients_negative: Fraction of patients with a negative slope.
        rho: Rank correlation between grid and response.
        rho_lo: Lower bound of the bootstrap interval on rho.
        rho_hi: Upper bound of the bootstrap interval on rho.
        delta: Mean response at the high end minus the low end of the grid.
        n: Patients contributing.
    """

    slope: float
    slope_lo: float
    slope_hi: float
    frac_patients_negative: float
    rho: float
    rho_lo: float
    rho_hi: float
    delta: float
    n: int

    @property
    def interval_excludes_zero(self) -> bool:
        """Return True when the slope interval lies wholly off zero."""
        return self.slope_lo > 0 or self.slope_hi < 0


def summarise_response(
    response: np.ndarray,
    correlator: GridRankCorrelation,
    rng: np.random.Generator,
    grid: np.ndarray = DEFAULT_GRID,
    n_boot: int = 1000,
    delta_threshold: float = 1.0,
) -> ResponseStatistics:
    """Summarise a response curve with bootstrap intervals over patients.

    The slope is the primary effect size, in predicted standard deviations of
    expression per standard deviation of injected value. The rank correlation is
    a scale-free companion, and the delta contrasts the extremes of the grid.

    Args:
        response: Response of shape ``(len(grid), n_patients)``.
        correlator: Precomputed grid ranks for this patient count.
        rng: Random generator for the bootstrap.
        grid: Injected values.
        n_boot: Bootstrap resamples over patients.
        delta_threshold: Grid magnitude defining the high and low ends.

    Returns:
        The summary.
    """
    slopes = patient_slopes(response, grid)
    n_patients = len(slopes)
    resample = rng.integers(0, n_patients, (n_boot, n_patients))

    slope_draws = slopes[resample].mean(axis=1)
    slope_lo, slope_hi = np.percentile(slope_draws, [2.5, 97.5])

    rho = correlator(response.reshape(-1))
    rho_draws = np.empty(n_boot)
    for i in range(n_boot):
        rho_draws[i] = correlator(response[:, resample[i]].reshape(-1))
    rho_lo, rho_hi = np.nanpercentile(rho_draws, [2.5, 97.5])

    return ResponseStatistics(
        slope=float(slopes.mean()),
        slope_lo=float(slope_lo),
        slope_hi=float(slope_hi),
        frac_patients_negative=float((slopes < 0).mean()),
        rho=rho,
        rho_lo=float(rho_lo),
        rho_hi=float(rho_hi),
        delta=float(
            response[grid >= delta_threshold].mean()
            - response[grid <= -delta_threshold].mean()
        ),
        n=n_patients,
    )


def observed_correlation(
    inputs: torch.Tensor, gene: int, inject_modality: str
) -> float:
    """Correlate a gene's observed modality against its observed expression.

    This is the data's own answer to the question the probe asks the model. The
    gap between them is the interesting quantity: a model merely echoing the
    correlation it was trained on has learned nothing extra, while a model whose
    learned coupling departs from the observed one is doing something the raw
    data does not.

    Args:
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        gene: Gene index.
        inject_modality: The modality the probe intervenes on.

    Returns:
        Spearman correlation, or ``nan`` when either channel is constant.
    """
    injected = inputs[:, gene, MODALITY_INDEX[inject_modality]].numpy()
    expression = inputs[:, gene, MODALITY_INDEX["rna"]].numpy()
    if np.std(injected) < 1e-12 or np.std(expression) < 1e-12:
        return float("nan")
    return float(spearmanr(injected, expression).correlation)


def run_cis_map(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    gene_order: Sequence[str],
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    inject_modality: str = "methy",
    arrow: str = "methy->rna",
    curated_genes: Sequence[str] = (),
    cnv_diploid: torch.Tensor | None = None,
    grid: np.ndarray = DEFAULT_GRID,
    batch_size: int = 8,
    n_boot: int = 1000,
    seed: int = 20240712,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probe every gene and summarise the learned coupling genome-wide.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        gene_order: Gene symbols matching the input's gene axis.
        graph_pe: Positional encodings for that axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device to run on.
        inject_modality: Modality to intervene on.
        arrow: Label recorded on every row, describing the intervention.
        curated_genes: Genes with independently known behaviour, flagged in the
            output so specificity can be tested afterwards.
        cnv_diploid: Standardised diploid copy number, to neutralise copy number.
        grid: Injected values.
        batch_size: Genes probed per forward pass.
        n_boot: Bootstrap resamples over patients.
        seed: Seed for the bootstrap.

    Returns:
        Tuple of the per-gene summary and the per-grid-point response curves.
    """
    rng = np.random.default_rng(seed)
    correlator = GridRankCorrelation(inputs.shape[0], grid)
    genes = list(range(len(gene_order)))

    spd_np = spd.cpu().numpy()
    grn_np = None if grn is None else grn.cpu().numpy()
    batches = make_gene_batches(spd_np, genes, batch_size, MIN_BATCH_DISTANCE, grn_np)
    logger.info(
        "%s: %d genes in %d batches (%.1fx fewer forward passes)",
        arrow,
        len(genes),
        len(batches),
        len(genes) / max(len(batches), 1),
    )

    responses: dict[int, np.ndarray] = {}
    for i, batch in enumerate(batches, start=1):
        responses.update(
            probe_genes(
                model,
                inputs,
                batch,
                inject_modality,
                graph_pe,
                spd,
                grn,
                device,
                grid=grid,
                cnv_diploid=cnv_diploid,
            )
        )
        if i % 10 == 0:
            logger.info("  batch %d/%d", i, len(batches))

    curated = set(curated_genes)
    summaries, curves = [], []
    for gene in genes:
        response = responses[gene]
        statistics = summarise_response(response, correlator, rng, grid, n_boot)
        name = gene_order[gene]
        observed = observed_correlation(inputs, gene, inject_modality)

        summaries.append(
            {
                "gene": name,
                "arrow": arrow,
                **statistics.__dict__,
                "rho_observed": observed,
                "rho_minus_observed": (
                    statistics.rho - observed if np.isfinite(observed) else np.nan
                ),
                "curated": int(name in curated),
            }
        )
        for position, value in enumerate(grid):
            curves.append(
                {
                    "gene": name,
                    "arrow": arrow,
                    "injected_value": float(value),
                    "mean_prediction": float(response[position].mean()),
                    "sem_prediction": float(
                        response[position].std() / np.sqrt(response.shape[1])
                    ),
                }
            )

    return pd.DataFrame(summaries), pd.DataFrame(curves)


def parity_check(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    gene_order: Sequence[str],
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    inject_modality: str = "methy",
    batch_size: int = 8,
    n_genes: int = 20,
    grid: np.ndarray = DEFAULT_GRID,
    seed: int = 20240712,
) -> pd.DataFrame:
    """Quantify what batching costs, on a random sample of genes.

    Batched probing is an approximation. This measures it directly by probing
    the same genes both ways and comparing the slopes, so the approximation is
    reported rather than assumed harmless.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs.
        gene_order: Gene symbols matching the input's gene axis.
        graph_pe: Positional encodings for that axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device to run on.
        inject_modality: Modality to intervene on.
        batch_size: Genes per forward pass in the batched arm.
        n_genes: Genes sampled for the comparison.
        grid: Injected values.
        seed: Seed for the gene sample.

    Returns:
        One row per sampled gene with both slopes and their difference. Empty
        when ``batch_size`` is one, since there is nothing to compare.
    """
    if batch_size <= 1:
        logger.info("batch_size is 1, so probing is already exact")
        return pd.DataFrame(
            columns=["gene", "slope_single", "slope_batched", "difference"]
        )

    rng = np.random.default_rng(seed)
    sampled = sorted(
        rng.choice(
            len(gene_order), min(n_genes, len(gene_order)), replace=False
        ).tolist()
    )

    single: dict[int, np.ndarray] = {}
    for gene in sampled:
        single.update(
            probe_genes(
                model,
                inputs,
                [gene],
                inject_modality,
                graph_pe,
                spd,
                grn,
                device,
                grid=grid,
            )
        )

    spd_np = spd.cpu().numpy()
    grn_np = None if grn is None else grn.cpu().numpy()
    batched: dict[int, np.ndarray] = {}
    for batch in make_gene_batches(
        spd_np, sampled, batch_size, MIN_BATCH_DISTANCE, grn_np
    ):
        batched.update(
            probe_genes(
                model,
                inputs,
                batch,
                inject_modality,
                graph_pe,
                spd,
                grn,
                device,
                grid=grid,
            )
        )

    frame = pd.DataFrame(
        [
            {
                "gene": gene_order[gene],
                "slope_single": float(patient_slopes(single[gene], grid).mean()),
                "slope_batched": float(patient_slopes(batched[gene], grid).mean()),
            }
            for gene in sampled
        ]
    )
    frame["difference"] = frame["slope_batched"] - frame["slope_single"]
    agreement = float(np.corrcoef(frame["slope_single"], frame["slope_batched"])[0, 1])
    logger.info(
        "parity: batched vs single slopes correlate %.4f, median |difference| %.5f",
        agreement,
        float(frame["difference"].abs().median()),
    )
    return frame
