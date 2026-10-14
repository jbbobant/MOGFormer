"""Unit tests for the causal simulator, effect estimators and benchmark.

Every test uses a hand-built graph or a linear model with a known answer, so
none needs cohort data or a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.analysis.effect_estimators import (
    ridge_conditional_effects,
    select_ridge_penalty,
    wald_ratio_effects,
)
from mogformer.analysis.probe_cis import DEFAULT_GRID
from mogformer.experiments.simulation_benchmark import (
    BenchmarkCondition,
    load_benchmark_plan,
    score_estimates,
)
from mogformer.simulation import GeneRegulatorySCM, SimulationSettings


@pytest.fixture()
def cascade() -> np.ndarray:
    """Return a signed graph: gene 0 activates 1, gene 1 represses 2, gene 3 is free.

    Rows are targets and columns regulators, the curated orientation.
    """
    graph = np.zeros((4, 4))
    graph[1, 0] = 1.0
    graph[2, 1] = -1.0
    return graph


def test_the_equilibrium_satisfies_the_structural_equations(
    cascade: np.ndarray,
) -> None:
    """The solved expression is a fixed point of ``x = W tanh(x) + u``."""
    scm = GeneRegulatorySCM(cascade, SimulationSettings(n_patients=50, seed=1))
    exogenous = np.random.default_rng(0).standard_normal((50, 4))
    solved = scm.solve(exogenous)
    np.testing.assert_allclose(
        solved, np.tanh(solved) @ scm.weights.T + exogenous, atol=1e-5
    )


def test_interventions_recover_each_edge_sign_and_its_absence(
    cascade: np.ndarray,
) -> None:
    """Clamping a regulator moves its targets in the curated direction.

    An unconnected gene is left untouched.
    """
    scm = GeneRegulatorySCM(cascade, SimulationSettings(n_patients=400, seed=2))
    cohort = scm.sample()
    from_first = scm.interventional_slopes(cohort, 0, np.array([1, 2, 3]), DEFAULT_GRID)
    assert from_first[0] > 0
    # Activation then repression: the indirect effect is negative.
    assert from_first[1] < 0
    assert from_first[2] == pytest.approx(0.0, abs=1e-9)


def test_hidden_programs_create_correlation_without_any_edge() -> None:
    """With no regulation at all, confounding alone correlates genes."""
    empty = np.zeros((30, 30))
    spread = []
    for confounding in (0.0, 0.8):
        scm = GeneRegulatorySCM(
            empty, SimulationSettings(n_patients=2000, confounding=confounding, seed=3)
        )
        rna = scm.sample().inputs[..., 0]
        correlation = np.corrcoef(rna.T)
        spread.append(np.abs(correlation[np.triu_indices(30, 1)]).mean())
    assert spread[1] > 3 * spread[0]


def test_regulatory_weights_are_contracted_below_one() -> None:
    """A dense cyclic graph is rescaled so the equilibrium stays unique."""
    dense = np.ones((12, 12))
    scm = GeneRegulatorySCM(dense, SimulationSettings(regulation_strength=3.0))
    radius = np.abs(np.linalg.eigvals(np.abs(scm.weights))).max()
    assert radius <= 0.9 + 1e-9
    assert np.all(np.diag(scm.weights) == 0)


def test_the_simulator_rejects_total_confounding(cascade: np.ndarray) -> None:
    """A confounding share of one leaves no private variation to identify from."""
    with pytest.raises(ValueError, match="confounding"):
        GeneRegulatorySCM(cascade, SimulationSettings(confounding=1.0))


def _linear_cohort(n: int, seed: int) -> tuple[np.ndarray, float]:
    """Draw a factor and target with a hidden confounder and a CNV instrument."""
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal(n)
    copy_number = rng.standard_normal((n, 2))
    factor = 0.8 * copy_number[:, 0] + hidden + 0.5 * rng.standard_normal(n)
    effect = -0.4
    target = effect * factor + 1.5 * hidden + 0.5 * rng.standard_normal(n)
    inputs = np.zeros((n, 2, 3))
    inputs[:, 0, 0], inputs[:, 1, 0] = factor, target
    inputs[:, :, 1] = copy_number
    inputs[:, :, 2] = rng.standard_normal((n, 2))
    return inputs, effect


def test_the_instrument_recovers_a_sign_that_confounding_reverses() -> None:
    """The Wald ratio finds a negative effect that confounding hides.

    Factor and target correlate positively in the observed data.
    """
    inputs, effect = _linear_cohort(20_000, seed=4)
    assert np.corrcoef(inputs[:, 0, 0], inputs[:, 1, 0])[0, 1] > 0
    estimate, first_stage_f = wald_ratio_effects(inputs, np.array([0]), np.array([1]))
    assert first_stage_f[0] > 100
    assert estimate[0] == pytest.approx(effect, abs=0.05)


def test_a_weak_instrument_is_withheld() -> None:
    """An instrument unrelated to the exposure yields no estimate."""
    rng = np.random.default_rng(5)
    inputs = rng.standard_normal((200, 2, 3))
    estimate, _ = wald_ratio_effects(
        inputs, np.array([0]), np.array([1]), min_first_stage_f=1e6
    )
    assert np.isnan(estimate[0])


def test_ridge_reads_the_conditional_coefficient_of_the_factor() -> None:
    """With a visible confounder, ridge isolates the factor's own coefficient."""
    rng = np.random.default_rng(6)
    n = 5000
    confounder = rng.standard_normal(n)
    factor = confounder + rng.standard_normal(n)
    target = 0.5 * factor - 2.0 * confounder + 0.3 * rng.standard_normal(n)
    inputs = np.zeros((n, 3, 3))
    inputs[:, 0, 0], inputs[:, 1, 0], inputs[:, 2, 0] = factor, confounder, target
    penalty = select_ridge_penalty(inputs, n_targets=3, seed=0)
    estimate = ridge_conditional_effects(inputs, np.array([0]), np.array([2]), penalty)
    assert estimate[0] == pytest.approx(0.5, abs=0.05)


def test_condition_names_state_every_setting() -> None:
    """A condition's directory name is readable without the sweep file."""
    condition = BenchmarkCondition(n_patients=949, confounding=0.8, use_grn=False)
    assert condition.name == "n949_confounding0.8_without_grn"


def test_the_benchmark_plan_is_read_from_yaml(tmp_path) -> None:
    """Conditions, probe settings and simulation settings all come from the file."""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(
        "simulation: {n_programs: 3}\n"
        "probe: {patients: 100, n_boot: 10, n_permutations: 50}\n"
        "conditions:\n"
        "  - {n_patients: 200, confounding: 0.5, use_grn: true}\n",
        encoding="utf-8",
    )
    plan = load_benchmark_plan(plan_file)
    assert plan.simulation == {"n_programs": 3}
    assert plan.probe_patients == 100 and plan.n_boot == 10
    assert plan.conditions[0].name == "n200_confounding0.5_with_grn"


def test_estimators_are_scored_against_both_truths() -> None:
    """A perfect estimator scores one against the curated and the true sign."""
    edge_sign = np.array([1, 1, 1, -1, -1, 1])
    pairs = pd.DataFrame(
        {
            "is_edge": [1] * 6 + [0] * 6,
            "edge_sign": list(edge_sign) + [0] * 6,
            "true_effect": list(0.1 * edge_sign) + [0.0] * 6,
            "slope": list(0.2 * edge_sign) + [0.01] * 6,
            "observed_coexpression": [0.3] * 12,
            "ridge_effect": list(-edge_sign) + [0.1] * 6,
            "instrumental_effect": [np.nan] * 3 + [-1.0, -1.0, 1.0] + [0.0] * 6,
        }
    )
    scores = {row["column"]: row for row in score_estimates(pairs, 200, seed=0)}
    assert scores["slope"]["curated_sign_rate"] == 1.0
    assert scores["slope"]["true_effect_sign_rate"] == 1.0
    assert scores["slope"]["edge_to_control_magnitude"] == pytest.approx(20.0)
    assert scores["ridge_effect"]["true_effect_sign_rate"] == 0.0
    assert scores["instrumental_effect"]["n_edges"] == 3
