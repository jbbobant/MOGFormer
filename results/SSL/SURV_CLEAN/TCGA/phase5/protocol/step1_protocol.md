# Phase 5 — Step 1 protocol (locked before fitting)
 
**Question.** Is the outcome-blind K=2 CNV/aneuploidy-burden split prognostic for PFI?
 
**Predictor.** `cluster` = Cluster B (high burden) vs A (low burden), reference = A.
 
**Cohort.** 335 labeled LumA (`step8_confirmatory_partition.parquet`, A=219/B=116,
core=334/1). Drop `TCGA-OL-A66H` (missing PFI/OS/DSS time) -> n=334 for PFI. The single
non-core patient `TCGA-PE-A5DE` (Cluster B, consensus 0.64) keeps its pre-registered
label in the primary analysis; a core-only sensitivity is reported.
 
**Endpoint.** PFI (non-cancer deaths censored per TCGA-CDR). Time in months (days/30.44).
 
**Covariates.**
- `age` = age_at_index, continuous, per year (conventional Cox coding).
- `stage` run both ways in parallel (comparison, not a choice):
  - ordinal I=1, II=2, III=3, IV=4 (sub-letters folded to parent; Stage 0->1);
  - binary early (0/I/II)=0 vs late (III/IV)=1.
  - Stage X and NaN (n=4) dropped for the stage rungs only -> stage-adjusted n=330.
 
**Adjustment ladder (estimands fixed in advance).**
1. unadjusted -> total effect;
2. +age -> confounder-adjusted sanity;
3. +age+stage -> direct (non-stage-mediated) effect — never reported as "the" HR.
 
**EPV discipline.** 35 PFI events -> <=3 covariates per model; the +age+stage rung
(3 df) sits at the 10-events-per-variable ceiling. No 4+ covariate confirmatory model.
 
**Diagnostics / fallback.** PH checked via scaled Schoenfeld residuals
(`proportional_hazard_test`, rank transform). If cluster PH is violated, RMST difference
(B-A) at 60 and 120 months is the primary effect measure. RMST is reported as a companion
regardless (bootstrap 2000x percentile CI). The 120-month RMST extrapolates beyond most
follow-up (median FU ~32 mo) and is interpreted cautiously.
 
**Reporting philosophy.** Lead on HR/CI, not p<0.05. A CI including 1 at 35 events is
expected and acceptable; overclaiming significance is not.
 
**Firewall.** Survival enters here for the first time. The frozen partition and the
`C_LumA` embedding carry no survival provenance (encoder trained outcome-blind, Phase 2).
 
**Seed.** Bootstrap seed = 42.
