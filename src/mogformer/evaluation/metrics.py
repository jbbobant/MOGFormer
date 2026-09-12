"""Per-fold metrics and their aggregation across a repeated partition.

Macro-F1 is the primary metric because the cohort is severely imbalanced and
accuracy would be dominated by the majority subtype. Per-class scores are always
reported alongside it, since a macro average can look healthy while one class
has collapsed entirely — which is exactly what happens to the rarest subtype
here.

Aggregation reports two intervals. Repeated k-fold estimates are not
independent — every pair of folds shares most of its training data — so the
naive interval across estimates is optimistic. The Nadeau–Bengio corrected
interval inflates the variance to account for that overlap. Both are reported so
the over-optimism is visible rather than quietly assumed away.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_labels: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, float]:
    """Score one fold's predictions.

    Threshold-free metrics are computed only when probabilities are supplied,
    and any class absent from this fold's held-out patients yields ``nan``
    rather than a misleading score.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_pred: Predicted labels, shape ``(n_samples,)``.
        y_proba: Class probabilities, shape ``(n_samples, n_classes)``, or None.
        class_labels: Integer codes, in the column order of ``y_proba``.
        class_names: Human-readable names, parallel to ``class_labels``.

    Returns:
        Flat mapping of metric name to value. Per-class entries are suffixed
        ``__<class name>``.

    Raises:
        ValueError: If ``class_labels`` and ``class_names`` differ in length.
    """
    if len(class_labels) != len(class_names):
        raise ValueError(
            f"class_labels has {len(class_labels)} entries but class_names has "
            f"{len(class_names)}"
        )

    labels = list(class_labels)
    scores: dict[str, float] = {
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    per_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_precision = precision_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_recall = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    for i, name in enumerate(class_names):
        scores[f"f1__{name}"] = per_f1[i]
        scores[f"precision__{name}"] = per_precision[i]
        scores[f"recall__{name}"] = per_recall[i]

    if y_proba is None:
        return scores

    binarised = label_binarize(y_true, classes=labels)
    present = binarised.sum(axis=0) > 0

    try:
        scores["roc_auc_macro"] = roc_auc_score(
            binarised[:, present],
            y_proba[:, present],
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        scores["roc_auc_macro"] = float("nan")

    scores["pr_auc_macro"] = _safe_macro_average_precision(binarised, y_proba, present)

    for i, name in enumerate(class_names):
        if not present[i]:
            scores[f"roc_auc__{name}"] = float("nan")
            scores[f"pr_auc__{name}"] = float("nan")
            continue
        try:
            scores[f"roc_auc__{name}"] = roc_auc_score(binarised[:, i], y_proba[:, i])
        except ValueError:
            scores[f"roc_auc__{name}"] = float("nan")
        scores[f"pr_auc__{name}"] = average_precision_score(
            binarised[:, i], y_proba[:, i]
        )
    return scores


def _safe_macro_average_precision(
    binarised: np.ndarray, proba: np.ndarray, present: np.ndarray
) -> float:
    """Average precision over the classes present in this fold.

    Args:
        binarised: One-hot true labels, shape ``(n_samples, n_classes)``.
        proba: Class probabilities of the same shape.
        present: Boolean mask over classes, True where the class occurs.

    Returns:
        Mean average precision over present classes, or ``nan`` if none are.
    """
    scores = [
        average_precision_score(binarised[:, i], proba[:, i])
        for i in range(binarised.shape[1])
        if present[i]
    ]
    return float(np.mean(scores)) if scores else float("nan")


def fold_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: Sequence[int]
) -> np.ndarray:
    """Return one fold's confusion matrix with a fixed class order.

    Args:
        y_true: True labels, shape ``(n_samples,)``.
        y_pred: Predicted labels, shape ``(n_samples,)``.
        class_labels: Integer codes fixing the row and column order, so
            per-fold matrices can be summed.

    Returns:
        Integer matrix of shape ``(n_classes, n_classes)``.
    """
    return confusion_matrix(y_true, y_pred, labels=list(class_labels))


def nadeau_bengio_correction(n_splits: int) -> float:
    """Return the test-to-train ratio used by the corrected-variance formula.

    For k-fold cross-validation each estimate trains on ``k - 1`` folds and
    tests on one, giving a ratio of ``1 / (k - 1)``.

    Args:
        n_splits: Folds per repeat.

    Returns:
        The ratio ``n_test / n_train``.

    Raises:
        ValueError: If ``n_splits`` is less than two.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    return 1.0 / (n_splits - 1)


def aggregate_across_folds(
    per_fold_values: Sequence[float] | np.ndarray,
    n_splits: int = 5,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Summarise one metric over the estimates of a repeated partition.

    Args:
        per_fold_values: One value per estimate. Missing values are dropped.
        n_splits: Folds per repeat, used to derive the correction ratio.
        alpha: Significance level; 0.05 gives 95% intervals.

    Returns:
        Mapping with ``mean``, ``std``, ``n``, the naive interval
        (``ci95_lo``/``ci95_hi``) and the corrected interval
        (``nb_ci95_lo``/``nb_ci95_hi``). Interval bounds are ``nan`` when fewer
        than two estimates survive.
    """
    values = np.asarray(per_fold_values, dtype=float)
    values = values[~np.isnan(values)]
    n_estimates = len(values)

    summary: dict[str, float] = {
        "mean": float(np.mean(values)) if n_estimates else float("nan"),
        "std": float(np.std(values, ddof=1)) if n_estimates > 1 else float("nan"),
        "n": float(n_estimates),
    }

    if n_estimates < 2:
        for key in ("ci95_lo", "ci95_hi", "nb_ci95_lo", "nb_ci95_hi"):
            summary[key] = float("nan")
        return summary

    critical = stats.t.ppf(1 - alpha / 2, df=n_estimates - 1)
    standard_error = summary["std"] / np.sqrt(n_estimates)
    summary["ci95_lo"] = summary["mean"] - critical * standard_error
    summary["ci95_hi"] = summary["mean"] + critical * standard_error

    inflation = (1.0 / n_estimates) + nadeau_bengio_correction(n_splits)
    corrected_sd = np.sqrt(inflation * np.var(values, ddof=1))
    summary["nb_ci95_lo"] = summary["mean"] - critical * corrected_sd
    summary["nb_ci95_hi"] = summary["mean"] + critical * corrected_sd
    return summary
