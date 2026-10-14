"""A structural causal model of tumour multi-omics on a real regulatory graph.

The trans probe asks a trained encoder what happens to a target when a
transcription factor is set to a value. On real tumours there is no ground
truth to score that answer against, so a failure cannot be attributed: it could
be the estimand, the sample size, hidden confounding, or a curated network whose
signs the cohort does not express. This simulator removes the last of those by
construction. Expression is generated *from* the CollecTRI signed edges, so every
edge's direction is true in the data, and the interventional effect of any
factor on any gene can be computed exactly by clamping the factor and solving
the system again.

Per patient, with copy number ``c``, methylation ``m``, hidden programs ``L`` and
expression ``x`` over the gene axis::

    c = sqrt(a) · A[segment] + sqrt(1 − a) · ξ              (amplicons)
    m = sqrt(κ) · Λ_m L + sqrt(1 − κ) · η
    u = β_c · c − β_m · m + sqrt(κ) · Λ_x L + sqrt(1 − κ) · ε
    x = W · tanh(x) + u                                      (equilibrium)

``W`` carries each curated edge with its curated sign, target by regulator.
``κ`` is the confounding strength: the share of each gene's exogenous variance
driven by a few hidden programs shared across genes, the bulk-tumour analogue of
subtype, proliferation and purity. Copy number is a dosage parent of its own
gene's expression and methylation a repressive one, as in the cis probe. With
the spectral radius of ``|W|`` below one and ``tanh`` 1-Lipschitz, the fixed
point exists, is unique and is reached by iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Largest spectral radius of ``|W|`` allowed; below one keeps the equilibrium
#: unique, and the margin keeps the iteration fast.
MAX_SPECTRAL_RADIUS = 0.9

#: Fixed-point iterations and the tolerance at which they may stop early.
MAX_ITERATIONS = 300
TOLERANCE = 1e-6


@dataclass(frozen=True)
class SimulationSettings:
    """Parameters of one simulated cohort.

    Attributes:
        n_patients: Patients drawn.
        confounding: Share ``κ`` of each gene's exogenous variance driven by
            hidden programs, in ``[0, 1)``.
        n_programs: Number of hidden programs.
        regulation_strength: Scale of each target's summed regulatory input;
            per-edge weights shrink as ``1/sqrt(in-degree)`` so a hub target is
            not dominated by regulation.
        cnv_effect: Dosage effect ``β_c`` of a gene's copy number on its own
            expression.
        methylation_effect: Silencing effect ``β_m`` of a gene's methylation on
            its own expression.
        amplicon_size: Genes sharing one copy-number segment.
        amplicon_share: Share ``a`` of copy-number variance common to a segment.
        seed: Seed for the structure and the draw.
    """

    n_patients: int = 949
    confounding: float = 0.0
    n_programs: int = 5
    regulation_strength: float = 1.0
    cnv_effect: float = 0.6
    methylation_effect: float = 0.6
    amplicon_size: int = 8
    amplicon_share: float = 0.5
    seed: int = 0


@dataclass
class SimulatedCohort:
    """A simulated cohort and the scale it was standardised with.

    Attributes:
        inputs: Standardised values, shape ``(n_patients, n_genes, 3)`` in
            ``(rna, cnv, methy)`` order — the layout the encoder and the probes
            read.
        exogenous: Exogenous expression input ``u``, shape
            ``(n_patients, n_genes)``, kept so interventions can be solved on
            the same patients.
        rna_mean: Per-gene mean of raw expression, shape ``(n_genes,)``.
        rna_scale: Per-gene standard deviation of raw expression.
    """

    inputs: np.ndarray
    exogenous: np.ndarray
    rna_mean: np.ndarray
    rna_scale: np.ndarray


def _standardise(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale each column, guarding against constant columns."""
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit norm, so each gene's loading has variance one."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class GeneRegulatorySCM:
    """A structural causal model whose regulatory edges are a given signed graph.

    The structure — weights, amplicon segments and program loadings — is fixed
    at construction, so cohorts of any size and interventions on them all come
    from the same causal system.

    Attributes:
        weights: Signed regulatory weights ``W``, shape ``(n_genes, n_genes)``,
            row target and column regulator — the orientation of the curated
            adjacency.
        segments: Copy-number segment of each gene, shape ``(n_genes,)``.
    """

    def __init__(
        self,
        signed_graph: np.ndarray,
        settings: SimulationSettings,
        device: torch.device | str = "cpu",
    ) -> None:
        """Draw the causal structure.

        Args:
            signed_graph: Curated adjacency in ``{-1, 0, +1}``, row target and
                column regulator.
            settings: Simulation parameters.
            device: Where the equilibrium is solved.

        Raises:
            ValueError: If the confounding share is outside ``[0, 1)`` or the
                graph is not square.
        """
        if not 0.0 <= settings.confounding < 1.0:
            raise ValueError(
                f"confounding must lie in [0, 1), got {settings.confounding}"
            )
        if signed_graph.ndim != 2 or signed_graph.shape[0] != signed_graph.shape[1]:
            raise ValueError(f"expected a square graph, got shape {signed_graph.shape}")

        self.settings = settings
        self.device = torch.device(device)
        self.n_genes = signed_graph.shape[0]
        rng = np.random.default_rng(settings.seed)

        sign = np.sign(signed_graph).astype(np.float64)
        np.fill_diagonal(sign, 0.0)
        in_degree = np.abs(sign).sum(axis=1, keepdims=True)
        magnitude = rng.uniform(0.5, 1.5, size=sign.shape)
        weights = (
            settings.regulation_strength
            * sign
            * magnitude
            / np.sqrt(np.maximum(in_degree, 1.0))
        )
        radius = float(np.abs(np.linalg.eigvals(np.abs(weights))).max())
        if radius > MAX_SPECTRAL_RADIUS:
            weights *= MAX_SPECTRAL_RADIUS / radius
            logger.info(
                "regulatory weights rescaled from spectral radius %.2f to %.2f",
                radius,
                MAX_SPECTRAL_RADIUS,
            )
        self.weights = weights

        self.segments = rng.permutation(self.n_genes) // settings.amplicon_size
        self._rna_loadings = _unit_rows(
            rng.standard_normal((self.n_genes, settings.n_programs))
        )
        self._methylation_loadings = _unit_rows(
            rng.standard_normal((self.n_genes, settings.n_programs))
        )
        self._rng = rng

    def solve(
        self,
        exogenous: np.ndarray,
        clamp_gene: int | None = None,
        clamp_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve ``x = W tanh(x) + u`` by fixed-point iteration.

        Clamping implements ``do(x_gene = value)``: the gene's own equation is
        replaced by the assigned value, and every other gene re-equilibrates
        around it.

        Args:
            exogenous: Exogenous input ``u``, shape ``(n_rows, n_genes)``.
            clamp_gene: Gene held fixed, or None for the observational solution.
            clamp_values: Its value per row, shape ``(n_rows,)``.

        Returns:
            Raw expression at equilibrium, shape ``(n_rows, n_genes)``.
        """
        weights = torch.as_tensor(
            self.weights.T, dtype=torch.float64, device=self.device
        )
        u = torch.as_tensor(exogenous, dtype=torch.float64, device=self.device)
        clamp = (
            None
            if clamp_gene is None or clamp_values is None
            else torch.as_tensor(clamp_values, dtype=torch.float64, device=self.device)
        )
        state = u.clone()
        for _ in range(MAX_ITERATIONS):
            if clamp is not None:
                state[:, clamp_gene] = clamp
            updated = torch.tanh(state) @ weights + u
            if clamp is not None:
                updated[:, clamp_gene] = clamp
            change = float((updated - state).abs().max())
            state = updated
            if change < TOLERANCE:
                break
        return state.cpu().numpy()

    def sample(self, n_patients: int | None = None) -> SimulatedCohort:
        """Draw a cohort from the model.

        Args:
            n_patients: Patients to draw; defaults to the settings' count.

        Returns:
            The standardised cohort, with what interventions on it need.
        """
        settings = self.settings
        n = settings.n_patients if n_patients is None else n_patients
        rng = self._rng
        kappa = settings.confounding

        n_segments = int(self.segments.max()) + 1
        segment_level = rng.standard_normal((n, n_segments))[:, self.segments]
        cnv = np.sqrt(settings.amplicon_share) * segment_level + np.sqrt(
            1.0 - settings.amplicon_share
        ) * rng.standard_normal((n, self.n_genes))

        programs = rng.standard_normal((n, settings.n_programs))
        methylation = np.sqrt(
            kappa
        ) * programs @ self._methylation_loadings.T + np.sqrt(
            1.0 - kappa
        ) * rng.standard_normal((n, self.n_genes))
        exogenous = (
            settings.cnv_effect * cnv
            - settings.methylation_effect * methylation
            + np.sqrt(kappa) * programs @ self._rna_loadings.T
            + np.sqrt(1.0 - kappa) * rng.standard_normal((n, self.n_genes))
        )

        rna = self.solve(exogenous)
        rna_z, rna_mean, rna_scale = _standardise(rna)
        cnv_z, _, _ = _standardise(cnv)
        methylation_z, _, _ = _standardise(methylation)
        inputs = np.stack([rna_z, cnv_z, methylation_z], axis=-1).astype(np.float32)
        return SimulatedCohort(
            inputs=inputs,
            exogenous=exogenous,
            rna_mean=rna_mean,
            rna_scale=rna_scale,
        )

    def interventional_slopes(
        self,
        cohort: SimulatedCohort,
        factor: int,
        targets: np.ndarray,
        grid: np.ndarray,
        max_patients: int = 2000,
    ) -> np.ndarray:
        """Compute the true total effect of setting a factor on each target.

        The factor is clamped at each grid value, given in the factor's
        standardised units exactly as the probe injects it, and the mean
        standardised response of each target is regressed on the grid. The
        result is in the probe's own units, so the two can be compared directly.

        Args:
            cohort: Cohort whose exogenous inputs the intervention reuses.
            factor: Gene intervened on.
            targets: Genes read, shape ``(n_targets,)``.
            grid: Assigned values, in standardised units.
            max_patients: Patients averaged over; the mean converges long
                before the full cohort is needed.

        Returns:
            Least-squares slope of each target's mean response on the grid,
            shape ``(n_targets,)``.
        """
        exogenous = cohort.exogenous[:max_patients]
        n_rows = exogenous.shape[0]
        values = cohort.rna_mean[factor] + np.asarray(grid) * cohort.rna_scale[factor]
        stacked = np.tile(exogenous, (len(grid), 1))
        clamp = np.repeat(values, n_rows)
        solved = self.solve(stacked, clamp_gene=factor, clamp_values=clamp)
        response = (
            solved[:, targets].reshape(len(grid), n_rows, len(targets)).mean(axis=1)
            - cohort.rna_mean[targets]
        ) / cohort.rna_scale[targets]
        centred_grid = np.asarray(grid) - np.mean(grid)
        return (
            centred_grid @ (response - response.mean(axis=0)) / (centred_grid**2).sum()
        )
