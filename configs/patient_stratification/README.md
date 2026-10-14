# Patient stratification: pre-registered protocol

Written on 2026-10-13, before any clustering of these populations with the
encoders below, and before any outcome is read. Outcomes are analysed only
under `outcome_protocol.md`, which is locked before that analysis runs.

## Encoders, chosen on outcome-blind criteria only

One encoder per cohort, chosen by the effective-dimension plan
(`configs/embedding_spread/README.md`). The criteria were the probe panel,
participation ratio, seed stability and the nuisance probes. No clustering of
the populations below and no outcome entered the choice.

| Cohort | Encoder (config) | Runs | Read-out |
|---|---|---|---|
| TCGA-BRCA | `configs/ssl/tcga_default_encoder.yaml` | `tcga_programs_fused_k16_seed42/43/44` | `summary`: 16 program blocks of 32, 512-d |
| METABRIC | `configs/ssl/metabric_default_encoder.yaml` | `metabric_programs_fused_k16_seed42/43/44` | the same |

Seed 42 gives the primary partition. Seeds 43 and 44 are used only for seed
agreement.

The suite's clustering checkpoint has already been run on two of these
populations (TCGA Luminal A and ER+/HER2−; METABRIC Luminal A and ER+/HER2−),
using 200 resamples of k-means on the raw embedding, structure only. Its PAC
values are known. This protocol does not reuse them; it reruns everything below.

## Populations

| Population | Cohort | n | Role |
|---|---|---|---|
| ER+/HER2− (`ER_IHC` positive, `HER2_SNP6` not gained) | METABRIC | 887 | **Primary discovery**; the outcome test |
| Luminal A (`CLAUDIN_SUBTYPE` LumA) | METABRIC | 500 | Secondary |
| ER+/HER2− (`receptors.csv`, `er_positive_her2_negative`) | TCGA | ≈ 439 | Structural replication only; no outcome claim |
| Luminal A (PAM50) | TCGA | ≈ 334 | Structure only; re-test of the old ploidy split |

## Procedure

1. **Input.** The encoder's `summary` read-out, restricted to the population.
   Each dimension is standardised within the population. Clustering uses the
   top principal components that explain 90% of the variance, capped at 32.
   The same reduction applies to every representation of the ladder below. As a
   sensitivity analysis, k-means is also run on the raw embedding, which is the
   suite checkpoint's convention.
2. **Consensus clustering** for K = 2…6: 500 resamples of 80% of patients, with
   two algorithms:
   - k-means (5 initialisations);
   - a full-covariance Gaussian mixture.
3. **Bar for "a stratification exists"** at a given K and algorithm. All three
   must hold:
   - PAC ≤ 0.10, with the consensus bounds (0.1, 0.9);
   - bootstrap ARI ≥ 0.80: the mean ARI between the full-population labels and
     the labels refitted on 100 bootstrap resamples, on the patients drawn;
   - seed ARI ≥ 0.70: the ARI between the seed-42 partition and the seed-43
     and seed-44 partitions (both pairs).
   Among the (K, algorithm) pairs that pass, the reported stratification is the
   one with the lowest PAC; ties go to the smaller K, then to k-means. If none
   passes, the finding is "no stable stratification", reported as such. The
   candidate with the lowest PAC is then still audited and described (steps
   5–7) as the best available partition. It is labelled as not meeting the bar
   and does not enter the outcome analysis of strata.
4. **Representation ladder.** The same procedure (steps 1–3, without the seed
   condition) is run on:
   - PCA of the standardised inputs (all omics concatenated);
   - MOFA+;
   - PCA of each omics alone;
   - the copy-number burden score alone (and ploidy in TCGA).
   If a burden or ploidy split reproduces the encoder's partition (ARI ≥ 0.8),
   it is reported as that split, not as a new one.
5. **Novelty audit.** AMI and ARI of the reported partition against every known
   axis, with `screen_covariates` (Holm-corrected).
   - **METABRIC:** PAM50, IntClust, proliferation score, ESR1 expression,
     burden, NPI, age, and `COHORT`.
   - **TCGA:** PAM50, proliferation, burden, ploidy, tissue source site, and
     age.

   The METABRIC table carries no histological grade, and the TCGA 2012
   copy-number, methylation and integrated clusters are not in the bundle. Both
   are recorded as unavailable. A re-clustering after regressing the known axes
   out of the embedding is labelled exploratory.
6. **Biology, per stratum:**
   - differential features per omics (Mann–Whitney, Benjamini–Hochberg);
   - which program blocks separate the strata (per-block variance between
     strata);
   - the strata's position on the known axes.

   Arm-level copy-number profiles need the gene coordinates, which are not yet
   downloaded, and are deferred.
7. **TCGA replication.** A classifier trained in METABRIC on the harmonised gene
   axis (genes shared by both encoders' inputs) assigns TCGA ER+/HER2− patients
   to the METABRIC strata. The agreement with TCGA's own clustering is reported.
   Structure only.

## Predictions (from the research plan)

- **PS5:** the TCGA Luminal A split does not coincide with ploidy (ARI with
  "median copy number ≥ 3" below 0.5).
- **PS6:** if a stable ER+/HER2− stratification exists, it is largely explained
  by known axes (AMI with the best known axis ≥ 0.3; proliferation or IntClust
  most likely).
- **PS7:** no directional prediction on whether a stratification exists.

## Outputs

Written to `results/experiments/patient_stratification/`:
- `stability_by_k.png` and `.csv`: PAC, bootstrap ARI and seed ARI per K,
  algorithm and representation;
- `strata_versus_known_axes.png` and the audit table;
- `strata_composition.png`;
- `differential_features.csv` and the per-block separation;
- `tcga_replication.png`.

No outcome column is loaded by the stratification experiment.

## Results (2026-10-13)

Run: `results/experiments/patient_stratification/` (`record.json`,
`stability_by_k.csv`, `strata_versus_known_axes.csv`).

- **METABRIC ER+/HER2−:** K = 3 k-means; PAC 0.058, bootstrap ARI 0.93, seed ARI
  0.82; the bar is met.
  - Stratum 0 is LumA, IC3/IC8.
  - Stratum 1 is LumB, IC1/IC9/IC6/IC10, with high proliferation and burden.
  - Stratum 2 is 77% processing batch 3, whose methylation is shifted as a
    whole.
  - No simpler representation reproduces the partition (ARI ≤ 0.24).
- **METABRIC Luminal A:** K = 2; the bar is met, but the split is processing
  batch 3 (AMI 0.32).
- **TCGA ER+/HER2− and Luminal A:** no stable stratification (seed ARI 0.64 and
  0.59).
- **Replication** of the METABRIC strata in TCGA ER+/HER2−: ARI 0.12, on 283
  shared genes.
- **Predictions:** PS5 held (ARI with ploidy 0.007). PS6 held (burden η² 0.38,
  proliferation 0.30).

The outcome results are in `outcome_analysis/`; the protocol was amended once
before any outcome was read (`outcome_protocol.md`). The write-up is in the
report "MOGFormer report: patient stratification and outcome".

## Rerun on the batch-corrected METABRIC encoder (2026-10-14)

`protocol_combat.yaml` reruns the protocol unchanged, except that the METABRIC
runs are `metabric_combat_encoder_seed42–44` (expression and methylation
ComBat-corrected for processing batch; `configs/batch_correction/`). Results are
in `results/experiments/patient_stratification_combat/`. Structure only.

- **METABRIC ER+/HER2−:** bar not met. K = 3 gives PAC 0.148 and seed ARI 0.56.
  The batch stratum is gone; the high-risk LumB/IC1/IC9/IC6/IC10 group remains.
- **METABRIC Luminal A:** bar not met. The batch split is gone (AMI 0.01).
- **TCGA:** unchanged, since its encoder is the same.

The outcome analysis is not rerun.

## Post hoc survival on the batch-corrected strata (2026-10-14)

**Post hoc**, outside the locked protocol: these strata did not meet the bar.
Run with `outcome_analysis --post-hoc` on `patient_stratification_combat`;
outputs are in `outcome_analysis_post_hoc/`. The same code on permuted outcomes
first gave nulls throughout (HR 0.88–1.19, log-rank p 0.59, C ≈ 0.5).

METABRIC ER+/HER2−, 887 patients, 335 relapses. Strata 0 (196) and 1 (422) are
mostly LumA; stratum 2 (269) is the high-risk group: LumB, IC1/IC9/IC6/IC10,
8q gain, 17p loss, high proliferation. The reference is stratum 1, the
largest.

Relapse hazard of stratum 2 against stratum 1:

| Rung | Relapse-free | Disease-specific |
|---|---|---|
| Unadjusted | 1.78 (1.41–2.25) | 2.25 (1.71–2.94) |
| + age, NPI, therapies | 1.51 (1.19–1.92) | 1.82 (1.39–2.40) |
| + IntClust high-risk | 1.22 (0.92–1.62) | 1.47 (1.06–2.03) |
| + proliferation | 1.03 (0.76–1.39) | 1.19 (0.84–1.67) |
| + stratified by batch | 1.08 (0.80–1.47) | 1.25 (0.88–1.76) |

- **Stratum 2's excess risk is the known biology.** It is absorbed by the
  IntClust high-risk group and proliferation. Most of it is early relapse
  (HR 2.17 before 5 years, 1.47 after). Restricted mean relapse-free time is
  5.0 months shorter at 5 years and 24.5 months shorter at 15 years.
- **Stratum 0 against stratum 1:** HR 0.79 (0.58–1.08). After proliferation it
  is 0.72 (0.53–0.99), and for late relapse 0.67 (0.43–1.03). This is a weak,
  exploratory hint of lower late relapse in stratum 0, which is defined mainly
  by 1p copy-number gains and hypomethylation.
- **No incremental value.** ΔC over the clinical model is 0.001
  (−0.039 to 0.029) when batch-stratified; likelihood-ratio p = 0.16.
- **Continuous embedding:** out-of-fold relapse C is 0.608, against 0.618 for
  PCA and 0.610 for MOFA+. With the clinical model it reaches 0.645 (PCA
  0.650, MOFA+ 0.651). None differs beyond noise.
- **The uncorrected batch stratum's residual hazard (1.39) does not reappear,**
  which supports reading it as batch.

## Interpretability of the batch-corrected METABRIC encoder (2026-10-14)

Run with `interpretability`; outputs are in
`patient_stratification_combat/interpretability/`. Encoder:
`metabric_combat_encoder_seed42`, on METABRIC ER+/HER2−.

**Method.** Integrated gradients attribute a linear read-out of the embedding to
every input (gene × omics). The baseline is the average patient (all-zero
standardised input). The path is log-spaced, because the value tokenizer is
steep near zero, which makes the encoder behave almost like a soft threshold
around the average patient. Completeness holds: the median gap is 0.0008
against a median score of 0.10.

- **What drives the relapse risk score** (ridge Cox on the embedding; post hoc):
  - High expression of proliferation genes raises it: GMPS, CCNE2, MCM4, MCM6,
    CDC25C, CCNB1, RFC4, CDCA7.
  - High MYH11 (myoepithelial / stromal) and the ER-pathway genes CREBBP and
    SIAH2 lower it.
  - RASGRF2 promoter methylation raises it.
  - Copy-number gains at 16p13 (CHTF18, METRN) lower it.
  - Shares of the absolute attribution: copy number 40%, expression 32%,
    methylation 27%.
- **What defines the high-risk stratum** (outcome-free):
  - an 8q copy-number gain (TMEM65, EXT1, RRM2B, TP53INP1, THEM6, TRPS1,
    EXOSC4, SYBU, MTDH) with 8p loss (CCAR2);
  - 17p loss (TP53, VAMP2, TNFSF12), visible in the differential features;
  - high proliferation (MCM4, CCNE2, CDC25C, AURKA, PLK1);
  - promoter hypermethylation of developmental genes (WNT2, PTF1A, GDF6).

  This is the 8p-loss/8q-gain, 17p-loss, proliferative profile of IC1 and IC9.
  One dosage paradox stands out: TRPS1 and SYBU are gained in copy number, yet
  low expression of them points towards the stratum.
- **Omics reliance.** The fusion gate puts 59% of its weight on methylation, 24%
  on copy number and 17% on expression, similarly in every stratum (the
  high-risk stratum leans a little more on copy number, 0.28). The gate weights
  how token content is mixed, not how much each omics contributes:
  attribution is spread across all three.
- **Gene programs.** Every block alone carries some relapse information
  (out-of-fold C 0.56–0.64):
  - the best is P4 (SLC39A6/LIV-1, BCL2: the luminal, ER-driven program),
    0.64;
  - then P2 (BRCA1, MYBL2, UBE2C: proliferation), 0.63;
  - and P15 (the 8q genes), 0.62.
