"""Outcome analysis under the locked protocol.

``configs/patient_stratification/outcome_protocol.md`` fixes the population,
endpoints, adjustment ladder and checks before any outcome is read against the
strata or the embedding. The analysis runs once. Before it does, the same code
runs on permuted outcomes (``permute=True``): every association must vanish
there, which checks the pipeline without looking at a real result.

Two parts:

* **strata** (only if the stratification protocol reported a stable
  partition): Kaplan–Meier and log-rank, the Cox adjustment ladder,
  proportional hazards, restricted mean survival, early versus late relapse,
  and Uno's incremental C-index over the clinical model;
* **continuous embedding** (always): a ridge-penalised Cox model on the
  embedding, scored by out-of-fold C-index against PCA, MOFA+ and the
  clinical model.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from mogformer.analysis.clustering import reduce_for_clustering
from mogformer.analysis.plots import plot_hazard_forest, plot_kaplan_meier
from mogformer.analysis.survival import (
    fit_cox,
    logrank_p_value,
    proportional_hazards_check,
    restricted_mean_survival_difference,
    uno_concordance,
)
from mogformer.experiments.patient_stratification import (
    load_cohort_inputs,
    load_protocol,
)

logger = logging.getLogger(__name__)

#: IntClust groups with high late-relapse risk in ER+ disease.
INTCLUST_HIGH_RISK = ("1", "2", "6", "9")

#: Column of the processing batch that stratifies the amended models.
BATCH = "cohort"

#: Clinical covariates in the order the adjustment ladder adds them.
LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("unadjusted", ()),
    ("+ age", ("age",)),
    ("+ NPI", ("age", "npi")),
    ("+ therapies", ("age", "npi", "endocrine", "chemotherapy")),
    ("+ IntClust high-risk", ("age", "npi", "endocrine", "chemotherapy", "ic_high")),
    (
        "+ proliferation",
        ("age", "npi", "endocrine", "chemotherapy", "ic_high", "proliferation"),
    ),
)

#: The clinical model of the incremental-value test and of the comparators.
CLINICAL = ("age", "npi", "endocrine", "chemotherapy", "ic_high", "proliferation")

#: Horizons of the restricted mean survival comparison, in months.
RMST_HORIZONS = (60.0, 120.0, 180.0)

#: Landmark separating early from late relapse, in months.
LANDMARK = 60.0

#: Ridge penalties tried by the inner cross-validation.
PENALTIES = (0.01, 0.1, 1.0, 10.0)


def outcome_frame(
    clinical: pd.DataFrame, patients: Sequence[str], proliferation: pd.Series
) -> pd.DataFrame:
    """Assemble endpoints and clinical covariates for a set of patients.

    Args:
        clinical: The METABRIC clinical table, indexed by patient.
        patients: Patients, in the order of the strata or embedding.
        proliferation: Proliferation score per patient.

    Returns:
        One row per patient: ``rfs_months``, ``rfs_event``, ``dss_months``,
        ``dss_event``, the clinical covariates of :data:`CLINICAL` and the
        processing batch ``cohort``.
    """
    table = clinical.reindex(list(patients))
    return pd.DataFrame(
        {
            "rfs_months": table["RFS_MONTHS"].astype(float).to_numpy(),
            "rfs_event": table["RFS_STATUS"]
            .astype(str)
            .str.startswith("1")
            .astype(int),
            "dss_months": table["OS_MONTHS"].astype(float).to_numpy(),
            "dss_event": (table["VITAL_STATUS"] == "Died of Disease").astype(int),
            "age": table["AGE_AT_DIAGNOSIS"].astype(float).to_numpy(),
            "npi": table["NPI"].astype(float).to_numpy(),
            "endocrine": (table["HORMONE_THERAPY"] == "YES").astype(int),
            "chemotherapy": (table["CHEMOTHERAPY"] == "YES").astype(int),
            "ic_high": table["INTCLUST"]
            .astype(str)
            .isin(INTCLUST_HIGH_RISK)
            .astype(int),
            "proliferation": proliferation.reindex(list(patients)).to_numpy(),
            "cohort": table["COHORT"].astype(float).to_numpy(),
        },
        index=list(patients),
    ).reset_index(drop=True)


def permute_outcomes(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle each endpoint's (time, event) pairs across patients.

    Covariates stay with their patients, so every association between the
    strata or embedding and the outcome is destroyed while the outcome's own
    distribution is kept.

    Args:
        frame: The outcome frame.
        seed: Seed of the shuffle.

    Returns:
        A copy with permuted endpoints.
    """
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    for endpoint in ("rfs", "dss"):
        order = rng.permutation(len(frame))
        columns = [f"{endpoint}_months", f"{endpoint}_event"]
        shuffled[columns] = frame[columns].to_numpy()[order]
    return shuffled


def strata_ladder(
    frame: pd.DataFrame, endpoint: str, strata: np.ndarray
) -> pd.DataFrame:
    """Cox adjustment ladder for every stratum against the largest one.

    The six pre-registered rungs are followed by the rung of Amendment 1: the
    full clinical model with a baseline hazard per processing batch.

    Args:
        frame: Outcome frame.
        endpoint: ``"rfs"`` or ``"dss"``.
        strata: Stratum per patient.

    Returns:
        One row per (rung, stratum): hazard ratio, interval, p-value, events.
    """
    reference = int(np.bincount(strata).argmax())
    dummies = {
        f"stratum_{s}": (strata == s).astype(int)
        for s in np.unique(strata)
        if s != reference
    }
    data = frame.assign(**dummies)
    rows = []
    rungs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        (label, covariates, ()) for label, covariates in LADDER
    ]
    rungs.append(("+ stratified by batch", CLINICAL, (BATCH,)))
    for label, covariates, strata_cols in rungs:
        for name in dummies:
            others = [d for d in dummies if d != name]
            result = fit_cox(
                data,
                f"{endpoint}_months",
                f"{endpoint}_event",
                name,
                [*others, *covariates],
                model_label=label,
                strata=strata_cols,
            )
            rows.append(
                {
                    "endpoint": endpoint,
                    "rung": label,
                    "stratum": int(name.split("_")[1]),
                    "reference": reference,
                    "hazard_ratio": result.hazard_ratio,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                    "n": result.n,
                    "events": result.events,
                }
            )
    return pd.DataFrame(rows)


def landmark_split(frame: pd.DataFrame, endpoint: str) -> dict[str, pd.DataFrame]:
    """Split follow-up at the landmark into early and late relapse.

    Early: events before the landmark, censored at it. Late: patients still
    event-free and in follow-up at the landmark, timed from it.

    Args:
        frame: Outcome frame.
        endpoint: ``"rfs"`` or ``"dss"``.

    Returns:
        ``{"early": ..., "late": ...}`` frames with the same columns.
    """
    months, event = f"{endpoint}_months", f"{endpoint}_event"
    early = frame.copy()
    beyond = early[months] > LANDMARK
    early.loc[beyond, event] = 0
    early.loc[beyond, months] = LANDMARK
    late = frame[frame[months] > LANDMARK].copy()
    late[months] = late[months] - LANDMARK
    return {"early": early, "late": late}


def incremental_value(
    frame: pd.DataFrame,
    endpoint: str,
    strata: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    batch_stratified: bool = False,
) -> dict[str, float]:
    """Uno ΔC-index and likelihood-ratio test of adding strata to the clinical model.

    Args:
        frame: Outcome frame.
        endpoint: ``"rfs"`` or ``"dss"``.
        strata: Stratum per patient.
        n_boot: Bootstrap resamples for the ΔC interval.
        seed: Seed of the bootstrap.
        batch_stratified: Give both models a baseline hazard per processing
            batch (Amendment 1).

    Returns:
        The two C-indices, ΔC with its percentile interval, and the
        likelihood-ratio statistic and p-value.
    """
    months, event = f"{endpoint}_months", f"{endpoint}_event"
    reference = int(np.bincount(strata).argmax())
    dummies = {
        f"stratum_{s}": (strata == s).astype(int)
        for s in np.unique(strata)
        if s != reference
    }
    batch = [BATCH] if batch_stratified else []
    data = frame.assign(**dummies)[[months, event, *CLINICAL, *dummies, *batch]]
    data = data.dropna()
    strata_cols = batch or None
    base = CoxPHFitter().fit(
        data[[months, event, *CLINICAL, *batch]], months, event, strata=strata_cols
    )
    full = CoxPHFitter().fit(data, months, event, strata=strata_cols)
    base_risk = base.predict_partial_hazard(data).to_numpy()
    full_risk = full.predict_partial_hazard(data).to_numpy()
    time, observed = data[months].to_numpy(), data[event].to_numpy()
    c_base = uno_concordance(time, observed, base_risk)
    c_full = uno_concordance(time, observed, full_risk)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(len(data), len(data), replace=True)
        deltas.append(
            uno_concordance(time[drawn], observed[drawn], full_risk[drawn])
            - uno_concordance(time[drawn], observed[drawn], base_risk[drawn])
        )
    statistic = 2.0 * (full.log_likelihood_ - base.log_likelihood_)
    return {
        "c_clinical": c_base,
        "c_clinical_plus_strata": c_full,
        "delta_c": c_full - c_base,
        "delta_c_low": float(np.percentile(deltas, 2.5)),
        "delta_c_high": float(np.percentile(deltas, 97.5)),
        "lr_statistic": float(statistic),
        "lr_p_value": float(stats.chi2.sf(statistic, len(dummies))),
        "batch_stratified": batch_stratified,
        "n": len(data),
        "events": int(observed.sum()),
    }


def ridge_cox_out_of_fold(
    features: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> list[float]:
    """Out-of-fold Harrell C of a ridge Cox model, penalty chosen per fold.

    Args:
        features: Standardised features, one row per patient.
        time: Follow-up per patient.
        event: Event indicator per patient.
        n_splits: Outer folds, stratified by event.
        n_repeats: Repeats of the outer split.
        seed: Seed of the splits.

    Returns:
        One C-index per outer fold and repeat.
    """
    scores = []
    for repeat in range(n_repeats):
        outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed + repeat)
        for train, test in outer.split(features, event):
            penalty = choose_ridge_penalty(
                features.iloc[train], time[train], event[train], seed
            )
            model = fit_ridge_cox(
                features.iloc[train], time[train], event[train], penalty
            )
            risk = model.predict_partial_hazard(features.iloc[test]).to_numpy()
            scores.append(concordance_index(time[test], -risk, event[test]))
    return scores


def fit_ridge_cox(
    features: pd.DataFrame, time: np.ndarray, event: np.ndarray, penalty: float
) -> CoxPHFitter:
    """Fit a ridge-penalised Cox model on standardised features.

    Args:
        features: Standardised features, one row per patient.
        time: Follow-up per patient.
        event: Event indicator per patient.
        penalty: Ridge penalty.

    Returns:
        The fitted model.
    """
    data = features.assign(_time=time, _event=event)
    return CoxPHFitter(penalizer=penalty, l1_ratio=0.0).fit(data, "_time", "_event")


def choose_ridge_penalty(
    features: pd.DataFrame, time: np.ndarray, event: np.ndarray, seed: int
) -> float:
    """Pick the ridge penalty by 3-fold inner cross-validation.

    Args:
        features: Standardised features, one row per patient.
        time: Follow-up per patient.
        event: Event indicator per patient.
        seed: Seed of the folds.

    Returns:
        The penalty among :data:`PENALTIES` with the best mean inner C-index.
    """
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_score = PENALTIES[0], -np.inf
    for penalty in PENALTIES:
        fold_scores = []
        for train, test in inner.split(features, event):
            model = fit_ridge_cox(
                features.iloc[train], time[train], event[train], penalty
            )
            risk = model.predict_partial_hazard(features.iloc[test]).to_numpy()
            fold_scores.append(concordance_index(time[test], -risk, event[test]))
        if np.mean(fold_scores) > best_score:
            best, best_score = penalty, float(np.mean(fold_scores))
    return best


def continuous_cindex(
    frame: pd.DataFrame,
    endpoint: str,
    representations: Mapping[str, np.ndarray],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
    batch_indicators: bool = False,
) -> pd.DataFrame:
    """Out-of-fold C-index of ridge Cox models on each representation.

    Args:
        frame: Outcome frame.
        endpoint: ``"rfs"`` or ``"dss"``.
        representations: Name to reduced features (patients x components).
            ``"clinical"`` and ``"clinical + <name>"`` combinations are added
            here.
        n_splits: Outer folds.
        n_repeats: Repeats of the outer split.
        seed: Seed of the splits.
        batch_indicators: Add processing-batch indicators to the clinical
            block (Amendment 1); the clinical sets are then named
            ``clinical + batch``.

    Returns:
        One row per representation: mean and SD of the out-of-fold C-index.
    """
    months, event = f"{endpoint}_months", f"{endpoint}_event"
    clinical = frame[list(CLINICAL)]
    complete = (
        clinical.notna().all(axis=1).to_numpy() & frame[months].notna().to_numpy()
    )
    time = frame[months].to_numpy()[complete]
    observed = frame[event].to_numpy()[complete]
    clinical_z = (clinical[complete] - clinical[complete].mean()) / clinical[
        complete
    ].std()
    base = clinical_z.reset_index(drop=True)
    base_name = "clinical"
    if batch_indicators:
        batches = pd.get_dummies(
            frame.loc[complete, BATCH].astype(int), prefix="batch", drop_first=True
        ).astype(float)
        base = pd.concat([base, batches.reset_index(drop=True)], axis=1)
        base_name = "clinical + batch"
    sets = {base_name: base}
    for name, values in representations.items():
        block = pd.DataFrame(
            values[complete], columns=[f"{name}_{i}" for i in range(values.shape[1])]
        )
        if not batch_indicators:
            sets[name] = block
        sets[f"{base_name} + {name}"] = pd.concat([base, block], axis=1)
    rows = []
    for name, features in sets.items():
        scores = ridge_cox_out_of_fold(
            features, time, observed, n_splits, n_repeats, seed
        )
        rows.append(
            {
                "endpoint": endpoint,
                "representation": name,
                "dimensions": features.shape[1],
                "c_index_mean": float(np.mean(scores)),
                "c_index_sd": float(np.std(scores, ddof=1)),
                "n": int(complete.sum()),
                "events": int(observed.sum()),
            }
        )
        logger.info(
            "%s %s: out-of-fold C %.3f ± %.3f",
            endpoint,
            name,
            rows[-1]["c_index_mean"],
            rows[-1]["c_index_sd"],
        )
    return pd.DataFrame(rows)


def run_outcome_analysis(
    protocol_path: str | Path,
    results_root: str | Path,
    stratification_dir: str | Path,
    output_dir: str | Path,
    permute: bool = False,
    post_hoc: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    """Carry out the locked outcome protocol on METABRIC ER+/HER2−.

    Args:
        protocol_path: The stratification ``protocol.yaml``.
        results_root: Directory holding the pretraining runs.
        stratification_dir: Output of the stratification protocol.
        output_dir: Where outputs are written.
        permute: Run on permuted outcomes, the pipeline check that precedes
            the real analysis.
        post_hoc: Analyse the strata even if they did not meet the
            stratification bar. Every output is then a post hoc analysis,
            outside the locked protocol, and is labelled so.
        seed: Seed of every resampling step.

    Returns:
        The record of both parts.
    """
    protocol = load_protocol(protocol_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    population = protocol.discovery_population
    cohort_name = next(
        name for name, s in protocol.cohorts.items() if population in s["populations"]
    )
    settings = protocol.cohorts[cohort_name]
    cohort = load_cohort_inputs(cohort_name, settings, Path(results_root), seed)
    rows = cohort.context.populations[population]
    patients = list(np.asarray(cohort.patients)[rows])

    clinical = pd.read_csv(Path(settings["bundle"]) / "clinical.csv", index_col=0)
    frame = outcome_frame(
        clinical, patients, cohort.context.covariates["proliferation"]
    )
    if permute:
        frame = permute_outcomes(frame, seed)
    label = "PERMUTED OUTCOMES (pipeline check)" if permute else population
    if post_hoc:
        label = f"{label} (post hoc)"

    stratification = json.loads(
        (Path(stratification_dir) / "record.json").read_text(encoding="utf-8")
    )[population]
    record: dict[str, object] = {
        "population": population,
        "permuted": permute,
        "post_hoc": post_hoc,
        "strata_meet_bar": bool(stratification["meets_bar"]),
        "n": len(frame),
        "rfs_events": int(frame["rfs_event"].sum()),
        "dss_events": int(frame["dss_event"].sum()),
    }

    if stratification["meets_bar"] or post_hoc:
        strata_table = pd.read_csv(Path(stratification_dir) / "strata.csv")
        strata_table = strata_table[strata_table["population"] == population]
        strata = (
            strata_table.set_index("patient").loc[patients, "stratum"].to_numpy(int)
        )
        frame["stratum"] = strata
        ladder = pd.concat(
            [strata_ladder(frame, e, strata) for e in ("rfs", "dss")], ignore_index=True
        )
        ladder.to_csv(output / "adjustment_ladder.csv", index=False)
        checks: dict[str, object] = {
            "logrank_p": {
                e: logrank_p_value(frame, f"{e}_months", f"{e}_event", "stratum")
                for e in ("rfs", "dss")
            },
            "proportional_hazards": proportional_hazards_check(
                frame, "rfs_months", "rfs_event", "stratum", list(CLINICAL)
            ).to_dict(orient="records"),
        }
        reference = int(np.bincount(strata).argmax())
        rmst = []
        for s in np.unique(strata):
            if s == reference:
                continue
            pair = frame[np.isin(strata, [reference, s])].copy()
            pair["exposed"] = (pair["stratum"] == s).astype(int)
            table = restricted_mean_survival_difference(
                pair, "rfs_months", "rfs_event", "exposed", RMST_HORIZONS, seed=seed
            )
            rmst.append(table.assign(stratum=s, reference=reference))
        pd.concat(rmst, ignore_index=True).to_csv(output / "rmst.csv", index=False)
        landmarks = []
        for period, part in landmark_split(frame, "rfs").items():
            part_strata = part["stratum"].to_numpy(int)
            landmarks.append(
                strata_ladder(part, "rfs", part_strata)
                .query("rung == 'unadjusted'")
                .assign(period=period)
            )
        pd.concat(landmarks, ignore_index=True).to_csv(
            output / "early_versus_late_relapse.csv", index=False
        )
        checks["incremental_value"] = {
            "batch_stratified": incremental_value(
                frame, "rfs", strata, seed=seed, batch_stratified=True
            ),
            "unstratified": incremental_value(frame, "rfs", strata, seed=seed),
        }
        record["strata"] = checks
        plot_kaplan_meier(
            frame,
            {
                "Relapse-free survival": ("rfs_months", "rfs_event"),
                "Disease-specific survival": ("dss_months", "dss_event"),
            },
            "stratum",
            output,
            name="relapse_by_stratum",
            title=label,
        )
        plot_hazard_forest(
            ladder[ladder["endpoint"] == "rfs"],
            output,
            name="adjustment_forest",
            title=f"{label}: relapse hazard by stratum",
        )
    else:
        record["strata"] = "skipped: no stratification met the pre-registered bar"

    representations = {
        "encoder": reduce_for_clustering(
            cohort.embeddings[0][rows],
            protocol.variance_explained,
            protocol.max_components,
        ),
        "pca_inputs": reduce_for_clustering(
            cohort.comparators["pca_inputs"][rows],
            protocol.variance_explained,
            protocol.max_components,
        ),
        "mofa": reduce_for_clustering(
            cohort.comparators["mofa"][rows],
            protocol.variance_explained,
            protocol.max_components,
        ),
    }
    continuous = pd.concat(
        [
            continuous_cindex(
                frame, e, representations, seed=seed, batch_indicators=batch
            )
            for e in ("rfs", "dss")
            for batch in (False, True)
        ],
        ignore_index=True,
    )
    continuous.to_csv(output / "continuous_embedding_cindex.csv", index=False)
    record["continuous"] = continuous.to_dict(orient="records")
    (output / "record.json").write_text(
        json.dumps(record, indent=2, default=float), encoding="utf-8"
    )
    return record
