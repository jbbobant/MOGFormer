"""Confirmatory survival analysis of a frozen partition.

Survival enters the project here and nowhere earlier. The encoder was trained
outcome-blind and the partition was frozen before any outcome was read, so this
is a genuine test of the representation rather than a restatement of its own
training signal. Nothing in this module may write back to the partition.

The protocol is fixed before fitting, and this module is written to make
deviating from it awkward:

* the estimand ladder is explicit — unadjusted, then age, then age and stage —
  and the direct effect is never reported as "the" hazard ratio;
* an events-per-variable ceiling is enforced, because a Cox model with more
  covariates than the event count supports produces confident nonsense;
* proportional hazards are checked, with restricted mean survival time as the
  companion measure when the assumption fails;
* the effect estimate and its interval lead the report, not the p-value.

A confidence interval that includes one is an answer, not a failure.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Days per month, for converting the cohort's day-scaled follow-up times.
DAYS_PER_MONTH = 30.44

#: Minimum events per covariate. Below this a Cox model is not interpretable.
MIN_EVENTS_PER_VARIABLE = 10


@dataclass
class CoxResult:
    """One rung of the adjustment ladder.

    Attributes:
        model: Human-readable description of the covariates included.
        estimand: What this rung estimates — total effect, confounder-adjusted,
            or direct effect with mediated paths removed.
        hazard_ratio: Hazard ratio for the predictor of interest.
        ci_low: Lower bound of the 95% interval.
        ci_high: Upper bound of the 95% interval.
        p_value: Wald p-value, reported but never led with.
        concordance: Model concordance index.
        n: Patients contributing.
        events: Events contributing.
    """

    model: str
    estimand: str
    hazard_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    concordance: float
    n: int
    events: int

    @property
    def interval_includes_null(self) -> bool:
        """Return True when the interval spans a hazard ratio of one."""
        return self.ci_low <= 1.0 <= self.ci_high


def months_from_days(days: pd.Series | np.ndarray) -> np.ndarray:
    """Convert follow-up times from days to months.

    Args:
        days: Follow-up durations in days.

    Returns:
        Durations in months.
    """
    return np.asarray(days, dtype=float) / DAYS_PER_MONTH


def check_events_per_variable(n_events: int, n_covariates: int) -> None:
    """Warn when a model has more covariates than its event count supports.

    Args:
        n_events: Events available.
        n_covariates: Degrees of freedom the model will spend.

    Raises:
        ValueError: If there are no events at all.
    """
    if n_events <= 0:
        raise ValueError("cannot fit a survival model with zero events")

    ratio = n_events / max(n_covariates, 1)
    if ratio < MIN_EVENTS_PER_VARIABLE:
        logger.warning(
            "%d events for %d covariate(s) is %.1f events per variable, below "
            "the %d ceiling; treat this rung as exploratory",
            n_events,
            n_covariates,
            ratio,
            MIN_EVENTS_PER_VARIABLE,
        )


def fit_cox(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictor: str,
    covariates: Sequence[str] = (),
    model_label: str = "",
    estimand: str = "",
) -> CoxResult:
    """Fit one Cox proportional-hazards rung.

    Args:
        frame: One row per patient, carrying the duration, event indicator,
            predictor and covariates. Rows missing any of them are dropped.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator, 1 for an event.
        predictor: The exposure whose hazard ratio is reported.
        covariates: Additional adjustment covariates.
        model_label: Description recorded on the result.
        estimand: What this rung estimates.

    Returns:
        The fitted rung.

    Raises:
        ImportError: If lifelines is not installed.
        ValueError: If the required columns are absent or nothing survives the
            complete-case filter.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "survival analysis needs lifelines; install the 'analysis' extra"
        ) from error

    columns = [duration_col, event_col, predictor, *covariates]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing column(s): {missing}")

    complete = frame[columns].dropna()
    if complete.empty:
        raise ValueError("no complete cases remain after dropping missing values")

    n_events = int(complete[event_col].sum())
    check_events_per_variable(n_events, len(covariates) + 1)

    fitter = CoxPHFitter()
    fitter.fit(complete, duration_col=duration_col, event_col=event_col)
    summary = fitter.summary.loc[predictor]

    return CoxResult(
        model=model_label or f"{predictor} + {list(covariates)}",
        estimand=estimand,
        hazard_ratio=float(summary["exp(coef)"]),
        ci_low=float(summary["exp(coef) lower 95%"]),
        ci_high=float(summary["exp(coef) upper 95%"]),
        p_value=float(summary["p"]),
        concordance=float(fitter.concordance_index_),
        n=len(complete),
        events=n_events,
    )


def adjustment_ladder(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictor: str,
    age_col: str | None = None,
    stage_col: str | None = None,
) -> pd.DataFrame:
    """Fit the pre-registered ladder of adjustments in order.

    The three rungs answer different questions and must not be conflated. The
    unadjusted rung is the total effect. Adding age is a confounder-adjusted
    sanity check. Adding stage removes paths that run *through* stage, so it
    estimates a direct effect — and because stage is plausibly a mediator of the
    exposure rather than a confounder of it, that rung is a decomposition, never
    "the" answer.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator.
        predictor: The exposure of interest.
        age_col: Age column, or None to skip that rung and the one after it.
        stage_col: Stage column, or None to skip the final rung.

    Returns:
        One row per rung, in ladder order.
    """
    rungs: list[CoxResult] = [
        fit_cox(
            frame,
            duration_col,
            event_col,
            predictor,
            model_label="Unadjusted",
            estimand="total effect",
        )
    ]

    if age_col is not None:
        rungs.append(
            fit_cox(
                frame,
                duration_col,
                event_col,
                predictor,
                covariates=[age_col],
                model_label="+ Age",
                estimand="confounder-adjusted",
            )
        )
        if stage_col is not None:
            rungs.append(
                fit_cox(
                    frame,
                    duration_col,
                    event_col,
                    predictor,
                    covariates=[age_col, stage_col],
                    model_label="+ Age + Stage",
                    estimand="direct effect, stage-mediated paths removed",
                )
            )

    for rung in rungs:
        logger.info(
            "%s (%s): HR %.2f [%.2f, %.2f], p %.3f, %d events",
            rung.model,
            rung.estimand,
            rung.hazard_ratio,
            rung.ci_low,
            rung.ci_high,
            rung.p_value,
            rung.events,
        )
    return pd.DataFrame([rung.__dict__ for rung in rungs])


def proportional_hazards_check(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    predictor: str,
    covariates: Sequence[str] = (),
) -> pd.DataFrame:
    """Test the proportional-hazards assumption via scaled Schoenfeld residuals.

    If the assumption fails for the predictor, a single hazard ratio is not a
    meaningful summary and :func:`restricted_mean_survival_difference` becomes
    the primary effect measure.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator.
        predictor: The exposure of interest.
        covariates: Additional adjustment covariates.

    Returns:
        Frame of test statistics and p-values, one row per covariate.

    Raises:
        ImportError: If lifelines is not installed.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "survival analysis needs lifelines; install the 'analysis' extra"
        ) from error

    columns = [duration_col, event_col, predictor, *covariates]
    complete = frame[columns].dropna()
    fitter = CoxPHFitter().fit(complete, duration_col=duration_col, event_col=event_col)
    results = proportional_hazard_test(fitter, complete, time_transform="rank").summary
    return results.reset_index()


def restricted_mean_survival_difference(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    horizons: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare restricted mean survival between two groups at fixed horizons.

    Unlike a hazard ratio this needs no proportionality assumption, and it is in
    months rather than a ratio, so it says how much time the difference is worth.
    A horizon beyond the median follow-up extrapolates and must be read with
    that in mind.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time, in the horizon's units.
        event_col: Column holding the event indicator.
        group_col: Binary group column; the higher value is the exposed group.
        horizons: Truncation times to evaluate.
        n_boot: Bootstrap resamples for the percentile interval.
        seed: Seed for the bootstrap.

    Returns:
        One row per horizon with the two group means, their difference and its
        interval.

    Raises:
        ImportError: If lifelines is not installed.
        ValueError: If the group column does not hold exactly two values.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.utils import restricted_mean_survival_time
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "survival analysis needs lifelines; install the 'analysis' extra"
        ) from error

    complete = frame[[duration_col, event_col, group_col]].dropna()
    groups = sorted(complete[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"{group_col!r} must hold exactly two groups, found {groups}")
    reference, exposed = groups

    def _rmst(subset: pd.DataFrame, horizon: float) -> float:
        fitter = KaplanMeierFitter().fit(subset[duration_col], subset[event_col])
        return float(restricted_mean_survival_time(fitter, t=horizon))

    rng = np.random.default_rng(seed)
    rows = []
    for horizon in horizons:
        reference_rows = complete[complete[group_col] == reference]
        exposed_rows = complete[complete[group_col] == exposed]
        observed = _rmst(exposed_rows, horizon) - _rmst(reference_rows, horizon)

        differences = np.empty(n_boot)
        for i in range(n_boot):
            resampled_reference = reference_rows.iloc[
                rng.integers(0, len(reference_rows), len(reference_rows))
            ]
            resampled_exposed = exposed_rows.iloc[
                rng.integers(0, len(exposed_rows), len(exposed_rows))
            ]
            differences[i] = _rmst(resampled_exposed, horizon) - _rmst(
                resampled_reference, horizon
            )

        rows.append(
            {
                "horizon": horizon,
                "rmst_reference": _rmst(reference_rows, horizon),
                "rmst_exposed": _rmst(exposed_rows, horizon),
                "difference": observed,
                "ci_low": float(np.percentile(differences, 2.5)),
                "ci_high": float(np.percentile(differences, 97.5)),
                "n_boot": n_boot,
            }
        )
    return pd.DataFrame(rows)


def describe_followup(
    frame: pd.DataFrame, duration_col: str, event_col: str, group_col: str | None = None
) -> dict[str, float]:
    """Summarise follow-up and event counts before any model is fitted.

    Reporting median follow-up alongside the event count is what lets a reader
    judge whether an interval is wide because the effect is small or because the
    study has barely begun to observe events.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator.
        group_col: Optional binary group column, for per-group counts.

    Returns:
        Mapping with the cohort size, event count and median follow-up, plus
        per-group counts when a group column is given.
    """
    complete = frame[[duration_col, event_col]].dropna()
    summary: dict[str, float] = {
        "n": float(len(complete)),
        "events": float(complete[event_col].sum()),
        "median_followup": float(complete[duration_col].median()),
    }

    if group_col is not None and group_col in frame.columns:
        for value, group in frame.dropna(subset=[duration_col, event_col]).groupby(
            group_col
        ):
            summary[f"n_{value}"] = float(len(group))
            summary[f"events_{value}"] = float(group[event_col].sum())
    return summary


def logrank_p_value(
    frame: pd.DataFrame, duration_col: str, event_col: str, group_col: str
) -> float:
    """Return the log-rank p-value comparing two groups.

    Args:
        frame: One row per patient.
        duration_col: Column holding follow-up time.
        event_col: Column holding the event indicator.
        group_col: Binary group column.

    Returns:
        The p-value.

    Raises:
        ImportError: If lifelines is not installed.
        ValueError: If the group column does not hold exactly two values.
    """
    try:
        from lifelines.statistics import logrank_test
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "survival analysis needs lifelines; install the 'analysis' extra"
        ) from error

    complete = frame[[duration_col, event_col, group_col]].dropna()
    groups = sorted(complete[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"{group_col!r} must hold exactly two groups, found {groups}")

    first = complete[complete[group_col] == groups[0]]
    second = complete[complete[group_col] == groups[1]]
    result = logrank_test(
        first[duration_col],
        second[duration_col],
        first[event_col],
        second[event_col],
    )
    return float(result.p_value)
