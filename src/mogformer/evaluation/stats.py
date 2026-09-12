"""Paired statistical comparison of models scored on identical folds.

Every model in this project is scored on the same persisted partition, which is
what makes these tests legitimate: a paired test removes the fold-to-fold
variation that would otherwise swamp a modest difference between two models.

Two tests are reported side by side. The Wilcoxon signed-rank test makes no
distributional assumption. The corrected-resampled t-test of Nadeau and Bengio
assumes normality but accounts for the overlap between repeated folds, which the
ordinary paired t-test ignores and which makes it far too liberal here. An
effect size accompanies both, because at twenty-five estimates a p-value alone
says little.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from mogformer.evaluation.metrics import nadeau_bengio_correction

#: Studentised range values over sqrt(2) at alpha = 0.05, indexed by the number
#: of models being compared. Used for the Nemenyi critical difference.
_NEMENYI_Q_ALPHA_005: dict[int, float] = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def corrected_resampled_ttest(
    differences: np.ndarray, n_splits: int = 5
) -> dict[str, float]:
    """Run the Nadeau–Bengio corrected paired t-test on per-fold differences.

    The correction inflates the variance by ``1/n + n_test/n_train`` to account
    for training sets shared between folds. Without it, repeated k-fold
    comparisons declare significance far too readily.

    Args:
        differences: Per-fold score differences between two models.
        n_splits: Folds per repeat, which sets the correction ratio.

    Returns:
        Mapping with ``t``, ``p_value`` and ``mean_diff``. Statistics are
        ``nan`` when fewer than two differences survive or all are identical.
    """
    values = np.asarray(differences, dtype=float)
    values = values[~np.isnan(values)]
    n_estimates = len(values)

    if n_estimates < 2:
        return {
            "t": float("nan"),
            "p_value": float("nan"),
            "mean_diff": float(np.mean(values)) if n_estimates else float("nan"),
        }

    mean_difference = float(values.mean())
    inflation = (1.0 / n_estimates) + nadeau_bengio_correction(n_splits)
    standard_error = np.sqrt(inflation * values.var(ddof=1))

    if standard_error <= 0:
        return {
            "t": float("nan"),
            "p_value": float("nan"),
            "mean_diff": mean_difference,
        }

    t_statistic = mean_difference / standard_error
    p_value = 2 * stats.t.sf(abs(t_statistic), df=n_estimates - 1)
    return {
        "t": float(t_statistic),
        "p_value": float(p_value),
        "mean_diff": mean_difference,
    }


def rank_biserial(differences: np.ndarray) -> float:
    """Compute the matched-pairs rank-biserial effect size.

    This is the effect-size companion to the Wilcoxon signed-rank test: it
    ranges over ``[-1, 1]`` and says how consistently one model wins, rather
    than merely whether the difference is detectable.

    Args:
        differences: Per-fold score differences between two models.

    Returns:
        Effect size, positive when the first model tends to win. Returns
        ``nan`` when every difference is zero or missing.
    """
    values = np.asarray(differences, dtype=float)
    values = values[(~np.isnan(values)) & (values != 0)]
    if len(values) == 0:
        return float("nan")

    ranks = stats.rankdata(np.abs(values))
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / ranks.sum())


def pairwise_compare(
    per_fold: Mapping[str, np.ndarray], n_splits: int = 5
) -> pd.DataFrame:
    """Compare every pair of models on their shared folds.

    Args:
        per_fold: Mapping of model name to its per-fold scores. All entries must
            be aligned to the same partition, in the same order.
        n_splits: Folds per repeat, which sets the correction ratio.

    Returns:
        One row per unordered pair, carrying both tests, the effect size and the
        number of usable pairs.

    Raises:
        ValueError: If the score vectors are not all the same length, which
            would mean the models were not scored on one partition.
    """
    lengths = {name: len(values) for name, values in per_fold.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"models were scored on different numbers of folds ({lengths}); "
            "paired comparison requires one shared partition"
        )

    rows = []
    for name_a, name_b in combinations(per_fold, 2):
        scores_a = np.asarray(per_fold[name_a], dtype=float)
        scores_b = np.asarray(per_fold[name_b], dtype=float)
        usable = ~(np.isnan(scores_a) | np.isnan(scores_b))
        differences = (scores_a - scores_b)[usable]

        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                scores_a[usable], scores_b[usable]
            )
        except ValueError:
            # Raised when every difference is zero.
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

        corrected = corrected_resampled_ttest(differences, n_splits)
        rows.append(
            {
                "model_a": name_a,
                "model_b": name_b,
                "mean_a": float(np.nanmean(scores_a)),
                "mean_b": float(np.nanmean(scores_b)),
                "mean_diff_a_minus_b": corrected["mean_diff"],
                "wilcoxon_stat": float(wilcoxon_stat),
                "wilcoxon_p": float(wilcoxon_p),
                "corrected_t": corrected["t"],
                "corrected_t_p": corrected["p_value"],
                "rank_biserial": rank_biserial(differences),
                "n_pairs": int(usable.sum()),
            }
        )
    return pd.DataFrame(rows)


def friedman_nemenyi(
    per_fold: Mapping[str, np.ndarray],
) -> tuple[float, float, pd.Series, float]:
    """Rank models across folds and compute the Nemenyi critical difference.

    The Friedman test asks whether any model differs; the critical difference
    then says which pairs of average ranks are far enough apart to distinguish.
    Two models whose ranks differ by less than it are statistically tied, and a
    critical-difference diagram draws exactly that.

    Args:
        per_fold: Mapping of model name to per-fold scores, higher being better.

    Returns:
        Tuple of the Friedman statistic, its p-value, the average ranks sorted
        best-first, and the critical difference at ``alpha = 0.05``.

    Raises:
        ValueError: If fewer than three models are supplied, or if no fold has a
            score for every model.
    """
    names = list(per_fold)
    if len(names) < 3:
        raise ValueError(
            f"the Friedman test needs at least three models, got {len(names)}"
        )

    matrix = np.vstack([np.asarray(per_fold[name], dtype=float) for name in names]).T
    complete = matrix[~np.isnan(matrix).any(axis=1)]
    if complete.shape[0] == 0:
        raise ValueError("no fold has a score for every model")

    n_folds, n_models = complete.shape
    statistic, p_value = stats.friedmanchisquare(
        *[complete[:, i] for i in range(n_models)]
    )

    # Negate first: higher score should become rank 1.
    ranks = np.apply_along_axis(stats.rankdata, 1, -complete)
    average_ranks = pd.Series(ranks.mean(axis=0), index=names).sort_values()

    q_alpha = _NEMENYI_Q_ALPHA_005.get(n_models, _NEMENYI_Q_ALPHA_005[10])
    critical_difference = q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_folds))
    return float(statistic), float(p_value), average_ranks, float(critical_difference)
