"""Tests for ComBat batch correction, on simulated batches."""

from __future__ import annotations

import numpy as np
import pytest

from mogformer.data.batch_correction import CombatCorrector, correct_batches
from mogformer.data.omics import OmicsData


@pytest.fixture
def batched() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """300 patients in 3 batches, 200 features; a biological group spans batches.

    Each batch adds its own shift and scale to every feature; the group shifts
    the first 20 features, equally often in every batch.
    """
    rng = np.random.default_rng(0)
    batches = np.repeat(["a", "b", "c"], 100)
    group = np.tile([0, 1], 150)
    values = rng.normal(size=(300, 200))
    values[:, :20] += 1.5 * group[:, None]
    shift = {"a": 0.0, "b": 1.0, "c": -0.8}
    scale = {"a": 1.0, "b": 1.6, "c": 0.7}
    for b in shift:
        rows = batches == b
        values[rows] = values[rows] * scale[b] + shift[b] + rng.normal(0, 0.2, 200)
    return values, batches, group


def _batch_eta(values: np.ndarray, batches: np.ndarray) -> float:
    """Median share of each feature's variance between batches."""
    grand = values.mean(axis=0)
    between = sum(
        (batches == b).sum() * (values[batches == b].mean(axis=0) - grand) ** 2
        for b in np.unique(batches)
    )
    return float(np.median(between / ((values - grand) ** 2).sum(axis=0)))


def test_combat_removes_the_batch_and_keeps_the_biology(batched) -> None:
    """Batch variance collapses; the group difference survives intact."""
    values, batches, group = batched
    corrected = CombatCorrector().fit(values, batches).transform(values, batches)

    assert _batch_eta(values, batches) > 0.2
    assert _batch_eta(corrected, batches) < 0.01
    difference = corrected[group == 1, :20].mean() - corrected[group == 0, :20].mean()
    assert difference == pytest.approx(1.5, abs=0.25)
    spread = [corrected[batches == b, 20:].std() for b in "abc"]
    assert max(spread) / min(spread) < 1.15


def test_effects_fitted_on_some_patients_apply_to_the_others(batched) -> None:
    """Held-out patients are corrected with the fitting patients' estimates."""
    values, batches, _ = batched
    fit = np.arange(300) % 4 != 0
    corrector = CombatCorrector().fit(values[fit], batches[fit])
    held_out = corrector.transform(values[~fit], batches[~fit])

    assert _batch_eta(held_out, batches[~fit]) < 0.03
    with pytest.raises(ValueError, match="not seen"):
        corrector.transform(values[:2], np.array(["a", "z"]))


def test_missing_values_stay_missing_and_other_omics_untouched(batched) -> None:
    """Only the named block changes, and its gaps are kept."""
    values, batches, _ = batched
    blocks = np.hstack([values[:, :50], values[:, 50:100], values[:, 100:150]])
    blocks[3, 5] = np.nan
    data = OmicsData(
        X=blocks,
        y=np.zeros(300, dtype=int),
        patient_ids=[f"p{i}" for i in range(300)],
        gene_names=[f"g{i}" for i in range(50)],
        label_map={"x": 0},
    )

    corrected, correctors = correct_batches(data, batches, np.arange(300), ["rna"])

    assert np.isnan(corrected.X[3, 5])
    np.testing.assert_array_equal(corrected.X[:, 50:], data.X[:, 50:])
    assert set(correctors) == {"rna"}
    assert _batch_eta(corrected.X[:, :50][:, 6:], batches) < 0.01
