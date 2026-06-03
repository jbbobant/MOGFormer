"""
metrics.py — Per-fold metrics + aggregation (naive 95% CI and Nadeau-Bengio
corrected-resampled CI).



Nadeau & Bengio (2003): because repeated k-fold folds overlap heavily, the naive
variance of the mean across the J=25 estimates is optimistic. The corrected
variance multiplies the sample variance by (1/J + n_test/n_train). For 5-fold,
n_test/n_train = (1/5)/(4/5) = 0.25. We report both so the over-optimism is
visible rather than hidden.
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
    y_proba: Optional[np.ndarray],
    class_labels: List[int],
    class_names: List[str],
) -> Dict[str, float]:
    """All scalar metrics for one fold. Confusion matrix returned separately."""
    out: Dict[str, float] = {}
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro",
                               labels=class_labels, zero_division=0)
    out["weighted_f1"] = f1_score(y_true, y_pred, average="weighted",
                                  labels=class_labels, zero_division=0)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["mcc"] = matthews_corrcoef(y_true, y_pred)

    per_f1 = f1_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    per_p = precision_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    per_r = recall_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
    for i, name in enumerate(class_names):
        out[f"f1__{name}"] = per_f1[i]
        out[f"precision__{name}"] = per_p[i]
        out[f"recall__{name}"] = per_r[i]

    if y_proba is not None:
        Yb = label_binarize(y_true, classes=class_labels)
        # guard: a class absent from this fold's y_true breaks per-class AUC
        present = Yb.sum(axis=0) > 0
        try:
            out["roc_auc_macro"] = roc_auc_score(
                Yb[:, present], y_proba[:, present], average="macro", multi_class="ovr"
            )
        except ValueError:
            out["roc_auc_macro"] = np.nan
        out["pr_auc_macro"] = _safe_macro_ap(Yb, y_proba, present)
        for i, name in enumerate(class_names):
            if present[i]:
                try:
                    out[f"roc_auc__{name}"] = roc_auc_score(Yb[:, i], y_proba[:, i])
                except ValueError:
                    out[f"roc_auc__{name}"] = np.nan
                out[f"pr_auc__{name}"] = average_precision_score(Yb[:, i], y_proba[:, i])
            else:
                out[f"roc_auc__{name}"] = np.nan
                out[f"pr_auc__{name}"] = np.nan
    return out


def _safe_macro_ap(Yb, proba, present) -> float:
    aps = []
    for i in range(Yb.shape[1]):
        if present[i]:
            aps.append(average_precision_score(Yb[:, i], proba[:, i]))
    return float(np.mean(aps)) if aps else np.nan


def fold_confusion(y_true, y_pred, class_labels) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=class_labels)


def aggregate(
    per_fold_values: np.ndarray,
    test_frac: float = 0.2,
    train_frac: float = 0.8,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Mean + naive 95% CI + Nadeau-Bengio corrected 95% CI for one metric."""
    v = np.asarray(per_fold_values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    res = {"mean": float(np.mean(v)) if n else np.nan,
           "std": float(np.std(v, ddof=1)) if n > 1 else np.nan,
           "n": n}
    if n > 1:
        sem = res["std"] / np.sqrt(n)
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        res["ci95_lo"] = res["mean"] - tcrit * sem
        res["ci95_hi"] = res["mean"] + tcrit * sem
        # Nadeau-Bengio corrected variance of the mean
        corr = (1.0 / n) + (test_frac / train_frac)
        nb_sd = np.sqrt(corr * np.var(v, ddof=1))
        res["nb_ci95_lo"] = res["mean"] - tcrit * nb_sd
        res["nb_ci95_hi"] = res["mean"] + tcrit * nb_sd
    else:
        for k in ("ci95_lo", "ci95_hi", "nb_ci95_lo", "nb_ci95_hi"):
            res[k] = np.nan
    return res
