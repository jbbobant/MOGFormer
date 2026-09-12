"""Interventional probing of regulation between genes.

The cis probe asks what a gene's own methylation does to its own expression.
This module asks the harder question: perturb a transcription factor and see
whether the effect reaches its targets, along the edges a reference regulatory
network says exist.

Two separate claims are testable here and they must not be conflated.

**Magnitude.** Does a perturbation move real targets more than matched
non-targets? Tested against control genes drawn from the same transcription
factor's non-neighbours, so the comparison is within-factor.

**Sign.** Does the effect carry the *direction* the network predicts — activation
raising the target, repression lowering it?

The sign test is where a naive analysis goes wrong, and this module is built
around avoiding that. Concordance must not be tested against a coin flip. The
edge set is heavily imbalanced toward activation, and the model's slopes carry
their own sign bias, so a p-value against 0.5 can be overwhelmingly significant
while the model still performs *worse* than a constant predictor that always
guesses the majority class. :func:`sign_permutation_test` shuffles edge labels
instead — preserving both the class imbalance and the slope-sign bias — and
reports the trivial majority baseline alongside the result so that failure mode
is visible rather than hidden.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from mogformer.analysis.probe_cis import (
    DEFAULT_GRID,
    MODALITY_INDEX,
    patient_slopes,
)

logger = logging.getLogger(__name__)

#: Largest fraction of the gene set that may be hidden in one forward pass.
#: Hiding too much at once starves the model of the context it needs to predict
#: anything, which would make every effect look small.
MAX_MASK_FRACTION = 0.10

#: Masking strategies for the genes being read.
MASK_MODES: tuple[str, ...] = ("whole_gene", "rna_only")


@dataclass
class SignConcordanceResult:
    """Outcome of the corrected sign-concordance test.

    Attributes:
        n_edges: Edges with a finite slope.
        concordant: Edges whose slope sign matches the network's sign.
        rate: Concordant fraction.
        null_mean_rate: Mean concordance under shuffled edge labels.
        null_sd_rate: Its standard deviation.
        perm_p: Permutation p-value against that null.
        z: Standardised distance from the null mean.
        frac_slope_positive: Fraction of slopes that are positive, which is the
            model-side bias the permutation null preserves.
        trivial_majority_baseline: Concordance from always predicting the
            majority edge sign.
        beats_trivial_baseline: Whether the observed rate exceeds it.
    """

    n_edges: int
    concordant: int
    rate: float
    null_mean_rate: float
    null_sd_rate: float
    perm_p: float
    z: float
    frac_slope_positive: float
    trivial_majority_baseline: float
    beats_trivial_baseline: bool

    @property
    def is_informative(self) -> bool:
        """Return True only if the result clears both bars.

        A result must be unlikely under shuffled labels *and* better than the
        constant majority predictor. Clearing only the first is the failure mode
        this test exists to expose.
        """
        return self.perm_p < 0.05 and self.beats_trivial_baseline


@torch.no_grad()
def probe_transcription_factor(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    factor: int,
    read_genes: Sequence[int],
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    grn: torch.Tensor | None,
    device: torch.device,
    inject_modality: str = "rna",
    mask_mode: str = "rna_only",
    grid: np.ndarray = DEFAULT_GRID,
    patient_chunk: int = 256,
) -> dict[int, np.ndarray]:
    """Perturb one transcription factor and read the response at many genes.

    All read genes are hidden together in a single forward pass, which is what
    makes probing a whole regulon affordable.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        factor: Index of the perturbed transcription factor.
        read_genes: Genes whose predicted expression is recorded.
        graph_pe: Positional encodings for the frozen gene axis.
        spd: Shortest-path matrix for that axis.
        grn: Signed regulatory matrix, or None.
        device: Device to run on.
        inject_modality: Modality of the factor to intervene on.
        mask_mode: ``"rna_only"`` hides just the read genes' expression, leaving
            their other channels visible; ``"whole_gene"`` hides all three, so a
            prediction can only come through the graph. Running both separates
            graph-routed effects from within-gene ones.
        grid: Injected values, in standardised units.
        patient_chunk: Patients per forward pass.

    Returns:
        Mapping of gene index to response, shape ``(len(grid), n_patients)``.

    Raises:
        ValueError: If ``mask_mode`` is unknown.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(
            f"unknown mask mode {mask_mode!r}; available: {list(MASK_MODES)}"
        )

    n_patients, n_genes = inputs.shape[0], inputs.shape[1]
    responses = {
        gene: np.empty((len(grid), n_patients), dtype=np.float32) for gene in read_genes
    }
    read = torch.as_tensor(list(read_genes), dtype=torch.long)
    rna_index = MODALITY_INDEX["rna"]

    for position, value in enumerate(grid):
        for start in range(0, n_patients, patient_chunk):
            stop = min(start + patient_chunk, n_patients)
            chunk = inputs[start:stop].to(device)
            rna = chunk[..., MODALITY_INDEX["rna"]].clone()
            cnv = chunk[..., MODALITY_INDEX["cnv"]].clone()
            methy = chunk[..., MODALITY_INDEX["methy"]].clone()

            {"rna": rna, "cnv": cnv, "methy": methy}[inject_modality][:, factor] = (
                float(value)
            )

            mask = torch.zeros(
                stop - start, n_genes, 3, dtype=torch.bool, device=device
            )
            if mask_mode == "whole_gene":
                mask[:, read, :] = True
            else:
                mask[:, read, rna_index] = True

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
            for gene in read_genes:
                responses[gene][position, start:stop] = predicted[:, gene]

    return responses


def slope_with_interval(
    response: np.ndarray,
    rng: np.random.Generator,
    grid: np.ndarray = DEFAULT_GRID,
    n_boot: int = 1000,
) -> dict[str, float]:
    """Summarise a response by its slope and a bootstrap interval.

    Lighter than the full cis summary: the trans sweep covers thousands of
    edges, and the rank-correlation bootstrap would dominate the runtime for no
    additional claim.

    Args:
        response: Response of shape ``(len(grid), n_patients)``.
        rng: Random generator for the bootstrap.
        grid: Injected values.
        n_boot: Bootstrap resamples over patients.

    Returns:
        Mapping with the slope, its interval, the fraction of patients with a
        negative slope, and the patient count.
    """
    slopes = patient_slopes(response, grid)
    n_patients = len(slopes)
    draws = slopes[rng.integers(0, n_patients, (n_boot, n_patients))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "slope": float(slopes.mean()),
        "slope_lo": float(low),
        "slope_hi": float(high),
        "frac_patients_negative": float((slopes < 0).mean()),
        "n": n_patients,
    }


def observed_coexpression(inputs: torch.Tensor, factor: int, target: int) -> float:
    """Correlate two genes' observed expression.

    The comparison the learned trans effect has to be judged against: an effect
    that merely tracks observed co-expression is an echo of the training data
    rather than evidence of learned routing.

    Args:
        inputs: Standardised inputs, shape ``(n_patients, n_genes, 3)``.
        factor: Index of the first gene.
        target: Index of the second gene.

    Returns:
        Spearman correlation, or ``nan`` when either is constant.
    """
    rna_index = MODALITY_INDEX["rna"]
    first = inputs[:, factor, rna_index].numpy()
    second = inputs[:, target, rna_index].numpy()
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(spearmanr(first, second).correlation)


def run_trans_sweep(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    gene_order: Sequence[str],
    grn: torch.Tensor,
    graph_pe: torch.Tensor,
    spd: torch.Tensor,
    device: torch.device,
    mask_mode: str = "rna_only",
    null_label: str = "real",
    inject_modality: str = "rna",
    n_controls_per_factor: int | None = None,
    grid: np.ndarray = DEFAULT_GRID,
    n_boot: int = 1000,
    seed: int = 20240712,
) -> pd.DataFrame:
    """Probe every regulatory edge, alongside matched non-edge controls.

    Controls are drawn per transcription factor from genes it does *not*
    regulate, so an edge is always compared against a non-edge under the same
    perturbation. A global control set would confound the comparison with
    differences between factors.

    Args:
        model: Frozen encoder.
        inputs: Standardised inputs.
        gene_order: Gene symbols matching the input's gene axis.
        grn: Signed regulatory adjacency, row target and column regulator.
        graph_pe: Positional encodings for that axis.
        spd: Shortest-path matrix for that axis.
        device: Device to run on.
        mask_mode: One of :data:`MASK_MODES`.
        null_label: Recorded on every row, distinguishing the real graph from a
            permuted one.
        inject_modality: Modality of the factor to intervene on.
        n_controls_per_factor: Controls per factor, or None to match its target
            count.
        grid: Injected values.
        n_boot: Bootstrap resamples over patients.
        seed: Seed for control sampling and the bootstrap.

    Returns:
        One row per probed pair, carrying the slope, the network's edge sign and
        the observed co-expression.
    """
    rng = np.random.default_rng(seed + 7)
    grn_np = grn.cpu().numpy()
    n_genes = len(gene_order)
    max_masked = max(2, int(MAX_MASK_FRACTION * n_genes))

    factors = sorted(np.nonzero((grn_np != 0).sum(axis=0))[0].tolist())
    logger.info(
        "%s | %s: %d factors, %d edges, at most %d genes hidden per pass",
        null_label,
        mask_mode,
        len(factors),
        int((grn_np != 0).sum()),
        max_masked,
    )

    rows = []
    for position, factor in enumerate(factors, start=1):
        targets = [t for t in np.nonzero(grn_np[:, factor])[0].tolist() if t != factor]
        pool = [g for g in range(n_genes) if grn_np[g, factor] == 0 and g != factor]
        n_controls = (
            len(targets) if n_controls_per_factor is None else n_controls_per_factor
        )
        controls = rng.choice(
            pool, size=min(n_controls, len(pool)), replace=False
        ).tolist()

        for start in range(0, len(targets + controls), max_masked):
            chunk = (targets + controls)[start : start + max_masked]
            responses = probe_transcription_factor(
                model,
                inputs,
                factor,
                chunk,
                graph_pe,
                spd,
                grn,
                device,
                inject_modality=inject_modality,
                mask_mode=mask_mode,
                grid=grid,
            )
            for gene in chunk:
                sign = int(grn_np[gene, factor])
                rows.append(
                    {
                        "tf": gene_order[factor],
                        "target": gene_order[gene],
                        "edge_sign": sign,
                        "is_edge": int(sign != 0),
                        "null": null_label,
                        "mask_mode": mask_mode,
                        "n_targets_of_tf": len(targets),
                        **slope_with_interval(responses[gene], rng, grid, n_boot),
                        "observed_coexpression": observed_coexpression(
                            inputs, factor, gene
                        ),
                    }
                )
        if position % 10 == 0:
            logger.info("  factor %d/%d", position, len(factors))

    return pd.DataFrame(rows)


def _count_concordant(edge_sign: np.ndarray, slope: np.ndarray) -> int:
    """Count edges whose slope sign matches the network's sign."""
    return int(
        (((edge_sign > 0) & (slope > 0)) | ((edge_sign < 0) & (slope < 0))).sum()
    )


def sign_permutation_test(
    edges: pd.DataFrame, n_perm: int = 10_000, seed: int = 20240712
) -> SignConcordanceResult:
    """Test sign concordance against a null that preserves both biases.

    Shuffling the edge labels — rather than the slopes, or comparing to a coin
    flip — keeps two properties the naive null destroys: the imbalance between
    activating and repressing edges, and the model's own tendency toward slopes
    of one sign. Both inflate concordance on their own.

    The trivial majority baseline is reported for the same reason. Against a
    binomial null at 0.5, a concordance of 60% over 900 edges looks
    overwhelming; if 77% of edges are activating, always guessing "activating"
    scores 77% and the model is in fact doing worse than a constant. Significant
    and useful are different questions, and only :attr:`
    SignConcordanceResult.is_informative` requires both.

    Args:
        edges: Rows for real edges, carrying ``edge_sign`` and ``slope``.
        n_perm: Label shuffles.
        seed: Seed for the shuffling.

    Returns:
        The test outcome.

    Raises:
        ValueError: If no edge has a finite slope.
    """
    rng = np.random.default_rng(seed)
    edge_sign = edges["edge_sign"].to_numpy().copy()
    slope = edges["slope"].to_numpy()

    finite = np.isfinite(slope)
    edge_sign, slope = edge_sign[finite], slope[finite]
    n_edges = len(edge_sign)
    if n_edges == 0:
        raise ValueError("no edge has a finite slope")

    observed = _count_concordant(edge_sign, slope)
    null = np.empty(n_perm)
    shuffled = edge_sign.copy()
    for i in range(n_perm):
        rng.shuffle(shuffled)
        null[i] = _count_concordant(shuffled, slope)

    null_sd = float(null.std())
    majority = float(max((edge_sign > 0).mean(), (edge_sign < 0).mean()))
    rate = observed / n_edges

    result = SignConcordanceResult(
        n_edges=n_edges,
        concordant=observed,
        rate=float(rate),
        null_mean_rate=float(null.mean() / n_edges),
        null_sd_rate=null_sd / n_edges,
        perm_p=float((1 + (null >= observed).sum()) / (n_perm + 1)),
        z=float((observed - null.mean()) / null_sd) if null_sd > 0 else float("nan"),
        frac_slope_positive=float((slope > 0).mean()),
        trivial_majority_baseline=majority,
        beats_trivial_baseline=bool(rate > majority),
    )

    logger.info(
        "sign concordance %.3f vs null %.3f (perm p %.3f) | trivial baseline "
        "%.3f | informative: %s",
        result.rate,
        result.null_mean_rate,
        result.perm_p,
        result.trivial_majority_baseline,
        result.is_informative,
    )
    if not result.beats_trivial_baseline:
        logger.warning(
            "concordance %.3f is below the trivial majority baseline %.3f; the "
            "model does not recover edge direction, whatever the p-value says",
            result.rate,
            result.trivial_majority_baseline,
        )
    return result


def magnitude_tests(sweep: pd.DataFrame, mask_mode: str) -> dict[str, object]:
    """Test whether effects are larger on real edges than on controls.

    This is the claim that survives independently of the sign question: even a
    model that cannot recover direction may still route effects preferentially
    along real edges.

    Args:
        sweep: Output of :func:`run_trans_sweep`.
        mask_mode: Masking mode to analyse.

    Returns:
        Mapping with the edge and control effect magnitudes, the comparison
        between them, the fraction of edges whose interval excludes zero, and
        the correlation with observed co-expression.
    """
    real = sweep[(sweep["null"] == "real") & (sweep["mask_mode"] == mask_mode)]
    edges = real[real["is_edge"] == 1]
    controls = real[real["is_edge"] == 0]
    if edges.empty:
        return {"mask_mode": mask_mode}

    edge_abs = edges["slope"].abs().dropna()
    control_abs = controls["slope"].abs().dropna()

    results: dict[str, object] = {
        "mask_mode": mask_mode,
        "n_edges": len(edges),
        "n_controls": len(controls),
        "edge_abs_median": float(edge_abs.median()),
        "control_abs_median": float(control_abs.median())
        if len(control_abs)
        else float("nan"),
        "frac_edges_interval_excludes_zero": float(
            ((edges["slope_lo"] > 0) | (edges["slope_hi"] < 0)).mean()
        ),
    }

    if len(control_abs) and len(edge_abs):
        results["mw_p_edge_stronger"] = float(
            mannwhitneyu(edge_abs, control_abs, alternative="greater").pvalue
        )
    if len(edge_abs) > 1:
        results["wilcoxon_p_nonzero"] = float(wilcoxon(edges["slope"].dropna()).pvalue)

    paired = edges[["slope", "observed_coexpression"]].dropna()
    if len(paired) > 2:
        results["spearman_vs_observed_coexpression"] = float(
            spearmanr(paired["slope"], paired["observed_coexpression"]).correlation
        )
    return results


def permuted_graph_retention(
    real: pd.DataFrame, permuted: pd.DataFrame
) -> dict[str, float]:
    """Measure how much of the trans effect survives shuffling the graph.

    The decisive control for whether the graph is doing the work. If effects
    persist at nearly full strength once node identities are permuted, most of
    what the probe measures is not graph-routed, and the retained fraction
    should be reported as a headline limit rather than a footnote.

    Args:
        real: Edge rows from the real graph.
        permuted: Edge rows from a graph with permuted node identities.

    Returns:
        Mapping with both effect magnitudes, the retained percentage and the
        complementary graph-attributable percentage.
    """
    real_median = float(real["slope"].abs().median())
    permuted_median = float(permuted["slope"].abs().median())
    retained = 100.0 * permuted_median / real_median if real_median else float("nan")

    logger.info(
        "permuted-graph null retains %.1f%% of the effect; %.1f%% is "
        "attributable to graph identity",
        retained,
        100.0 - retained,
    )
    return {
        "real_abs_median": real_median,
        "permuted_abs_median": permuted_median,
        "retained_pct": retained,
        "graph_attributable_pct": 100.0 - retained,
    }
