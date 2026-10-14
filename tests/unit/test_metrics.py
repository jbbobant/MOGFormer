"""Unit tests for the per-fold metric suite."""

from __future__ import annotations

import numpy as np
import pytest

from mogformer.evaluation.metrics import compute_fold_metrics


def test_a_two_class_fold_is_scored_with_one_curve_per_class() -> None:
    """Binary labels get per-class threshold-free scores, like multi-class ones."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])

    scores = compute_fold_metrics(y_true, y_pred, y_proba, [0, 1], ["Neg", "Pos"])

    assert scores["roc_auc__Neg"] == pytest.approx(1.0)
    assert scores["roc_auc__Pos"] == pytest.approx(1.0)
    assert scores["roc_auc_macro"] == pytest.approx(1.0)
    assert np.isfinite(scores["pr_auc_macro"])


def test_a_class_absent_from_the_fold_scores_nan_not_zero() -> None:
    """Threshold-free scores for a class with no held-out patient are missing."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array(
        [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]]
    )

    scores = compute_fold_metrics(y_true, y_pred, y_proba, [0, 1, 2], ["A", "B", "C"])

    assert np.isnan(scores["roc_auc__C"])
    assert np.isfinite(scores["roc_auc__A"])
    assert scores["macro_f1"] >= 0.0
