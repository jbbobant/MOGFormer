# Outcome analysis: locked protocol

Locked on 2026-10-13, before the stratification analysis has run, and so before
any outcome has been read by stratum or against the embedding. The only outcome
data seen so far are cohort-wide event counts (METABRIC: 580 relapses among
1,416 patients; 467 deaths from the disease).

## Population and endpoints

- **Population:** METABRIC ER+/HER2− (887 patients).
- **Primary endpoint:** relapse-free survival (`RFS_MONTHS`, `RFS_STATUS`).
- **Secondary endpoint:** disease-specific survival: `OS_MONTHS`, with death
  counted as an event when `VITAL_STATUS` is "Died of Disease" and censored
  otherwise.
- **TCGA:** no outcome analysis. Follow-up is too short, and the plan restricts
  TCGA to structural replication.

## Analyses

### A. Strata

This part runs only if the stratification protocol reports a stable
stratification. Otherwise it is skipped and recorded as skipped.

1. **Kaplan–Meier curves** per stratum and a log-rank test.
2. **Cox adjustment ladder** for the stratum (reference: the largest stratum):
   1. unadjusted;
   2. + age;
   3. + NPI;
   4. + endocrine therapy and chemotherapy (`HORMONE_THERAPY`, `CHEMOTHERAPY`);
   5. + the IntClust high-risk group (IC1, IC2, IC6, IC9 against the rest);
   6. + the proliferation score.

   Each step reports the HR with its 95% CI, and the number of events per
   variable is checked.
3. **Checks:**
   - proportional hazards (Schoenfeld residual test);
   - RMST differences at 5, 10 and 15 years, with bootstrap CIs (2,000
     resamples);
   - a landmark analysis of early (< 5 years) and late (≥ 5 years) relapse.
4. **Incremental value:** Uno's C-index of the clinical model (age, NPI,
   therapies, IntClust group, proliferation) with and without the strata.
   Reported as ΔC with a bootstrap CI (1,000 resamples) and a likelihood-ratio
   test.

### B. Continuous embedding (runs whether or not strata exist)

- **Model:** ridge-penalised Cox on the population's embedding, reduced as in
  the stratification protocol (top PCs explaining 90% of variance, capped at
  32). The penalty is chosen by inner cross-validation.
- **Score:** out-of-fold Harrell's C-index, from 5-fold cross-validation
  repeated 5 times, stratified by event.
- **Comparators,** under the same procedure at equal dimension:
  - PCA of the standardised inputs;
  - MOFA+;
  - the clinical model alone;
  - the clinical model plus the embedding.

## Predictions

- **PO1:** the strata are associated with RFS before adjustment (log-rank
  p < 0.05).
- **PO2:** the incremental value over NPI, proliferation and IntClust is small
  (ΔC ≤ 0.02).
- **PO3:** the embedding's out-of-fold C-index does not exceed MOFA+'s by more
  than 0.02.

## Rules

- The analysis runs once. Its code is reviewed and tested on permuted outcomes
  before the real outcomes are loaded.
- No encoder, population, K or adjustment set is changed after outcomes have
  been seen. Any further analysis is labelled post hoc.

## Outputs

Written to `results/experiments/patient_stratification/`:
- `relapse_by_stratum.png`;
- `adjustment_forest.png`;
- `early_versus_late_relapse.png`;
- `incremental_value.md`;
- `continuous_embedding_cindex.csv`.

## Amendment 1 (2026-10-13, before any outcome was read)

**Trigger.** The stratification protocol found a stable K = 3 partition of
METABRIC ER+/HER2− (PAC 0.058, bootstrap ARI 0.93, seed ARI 0.82). One of its
strata, stratum 2 (295 patients), is 77% processing batch (`COHORT`) 3. That
batch's methylation is shifted as a whole: its mean standardised methylation
input is +0.14, against −0.08 to −0.18 for batches 1, 2 and 4. RNA and copy
number show no such shift. Strata 0 and 1 differ biologically: stratum 0 is LumA
with IC3/IC8; stratum 1 is LumB with IC1/IC9/IC6/IC10 and higher proliferation
and burden. The amendment was decided with the project lead after these
structural results, and before any survival column was loaded by stratum or
against the embedding.

**Changes.**
1. **Primary contrast.** Stratum 1 against stratum 0 (the largest, the
   reference) is the main comparison. Stratum 2 is reported against the same
   reference, labelled as mostly a processing batch, and is not interpreted
   biologically.
2. **Batch-stratified model.** A seventh rung adds a Cox model stratified by
   `COHORT` (a separate baseline hazard per batch) on top of rung 6. It is the
   primary adjusted estimate. Rungs 1–6 are kept as written.
3. **Incremental value.** Both the clinical model and the clinical model plus
   strata are stratified by `COHORT`. The unstratified ΔC is also reported.
4. **Continuous embedding.** Every model is repeated with batch indicators
   added to the clinical covariates (`clinical + batch`, and `clinical + batch
   + <representation>`). The unadjusted comparison is kept as written.

The predictions are unchanged.
