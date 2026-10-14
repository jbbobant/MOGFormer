# Processing-batch correction (METABRIC)

## Why

The stratification protocol (`configs/patient_stratification/`) found that one of
the three stable METABRIC ER+/HER2− strata is 77% processing batch (`COHORT`) 3,
and that the stable Luminal A split is that batch. A closer look at the inputs:

| Omics | Batch effect | First principal component |
|---|---|---|
| Expression | 19% of genes with batch η² > 0.10; stronger inside Luminal A alone, so not subtype mix | 10% of variance, batch η² 0.36 |
| Methylation | a global level shift (batch 3 about 0.3 units higher), weak per gene | 9% of variance; it is the patient's mean methylation (r = 0.98), batch η² 0.18 |
| Copy number | none (median η² 0.003) | — |

Batches differ only modestly in subtype mix: LumA ranges 26–40% of each batch,
and ER-positive 72–83%.

## Method

ComBat, the parametric empirical-Bayes location-and-scale form (Johnson, Li &
Rabinovic 2007), implemented in `mogformer.data.batch_correction`:
- it is applied to expression and methylation, before gene selection and
  scaling;
- it is fitted on the fitting patients (train plus monitor) and applied to all;
- it uses no biological covariates, which would carry labels into an encoder
  trained without them.

The configuration is `metabric_combat_encoder.yaml`: the METABRIC default
encoder plus `preprocess.batch_correction: combat`.

## Check before training (PCA of the inputs, 32 components, linear probes)

| | Expression, raw | Expression, ComBat | Methylation, raw | Methylation, ComBat |
|---|---|---|---|---|
| Batch (5 classes, chance ≈ 0.2) | 0.78 | 0.12 | 0.44 | 0.20 |
| First component's batch η² | 0.36 | 0.00 | 0.18 | 0.00 |
| ER | 0.885 | 0.881 | 0.861 | 0.856 |
| Integrative cluster | 0.637 | 0.631 | 0.465 | 0.460 |
| PAM50 | 0.632 | 0.599 | 0.476 | 0.464 |

The batch falls to chance, and biology is kept. The small loss in PAM50 is
probably the part of the subtype mix that differs between batches.

## Predictions, written before training

1. The batch probe on the encoder's summary falls from 0.62 to below 0.35.
2. Effective dimension stays at or above 15. It may fall somewhat, since the
   batch directions are gone.
3. Primary probes pass the amended rule against the uncorrected encoder: ER,
   HER2 and integrative cluster unchanged; PAM50 may lose up to 0.03.
4. Rerun on this encoder, the stratification of METABRIC ER+/HER2− loses its
   batch stratum. The LumA / LumB-high-risk division remains, and the Luminal A
   batch split disappears.

## Outcomes (2026-10-14)

### Encoder (three seeds; evaluation `metabric_combat_encoder`)

| | Uncorrected default (`metabric_programs_fused_k16`) | ComBat |
|---|---|---|
| Batch probe (5 classes, chance ≈ 0.2) | 0.617 | **0.251** |
| Effective dimension | 16.03 ± 0.60 | 15.29 ± 0.79 (14.9, 14.8, 16.2) |
| ER / HER2 / integrative cluster | 0.850 / 0.699 / 0.563 | 0.842 / 0.709 / 0.565 |
| PAM50 | 0.533 | 0.517 |
| Proliferation | 0.906 | 0.913 |
| Seed-to-seed CKA | 0.975–0.983 | 0.985–0.987 |
| Held-out masked loss, global / local head | 0.636 / 0.552 | 0.669 / 0.574 |

- **Prediction 1 held.** The batch probe fell to 0.25.
- **Prediction 2 held on average only.** The mean dimension is 15.3, but two
  seeds are just under 15.
- **Prediction 3 held.** Every primary probe passes the amended rule, and PAM50
  loses 0.016.
- **Reconstruction is slightly worse**, as expected once the batch can no
  longer be used to predict values.
- **Caveat: a small residual batch signal.** PCA of the corrected inputs scores
  0.08 on the batch probe, so the encoder keeps a little batch signal beyond
  what its inputs carry linearly. It is near chance, but on the relative
  nuisance rule it does not pass.

### Stratification rerun (`results/experiments/patient_stratification_combat/`)

| Population | Uncorrected | ComBat |
|---|---|---|
| METABRIC ER+/HER2− | K = 3; PAC 0.058, seed ARI 0.82; **bar met**; one stratum 77% batch 3 | K = 3; PAC 0.148, bootstrap 0.88, seed ARI 0.56; **bar not met** |
| METABRIC Luminal A | K = 2; PAC 0.019; bar met; the batch-3 split (AMI 0.32) | K = 2; PAC 0.070, seed ARI 0.40; bar not met; batch AMI 0.01 |

The METABRIC ER+/HER2− strata after correction:
- **Stratum 2 (269):** LumB 45%, Her2 11%, and IC1, IC9, IC6, IC10. It is the
  same high-risk group as before, now without batch.
- **Strata 0 (196) and 1 (422):** mostly LumA, IC3/IC8, with no batch
  alignment. Their division is not reproducible across seeds.
- **Known axes:** burden (η² 0.39) and proliferation (0.24) remain the main
  axes; batch is no longer among them.

Prediction 4 held: the batch stratum and the batch split of Luminal A are gone,
and the high-risk LumB/IC division remains. But the remaining partition does
not meet the stratification bar. Much of the earlier stability came from the
batch.

### Conclusion

- **The ComBat encoder replaces the uncorrected one** as the METABRIC encoder
  for any further analysis. Its embedding no longer encodes the processing
  batch, at a cost of 0.7 dimensions and 0.016 in PAM50.
- **On it, METABRIC ER+/HER2− has no stratification that meets the
  pre-registered bar.** Its most robust structure is the high-risk
  LumB/IC1/IC9/IC6/IC10 group.
- **The outcome analysis is not rerun.** It was locked to run once, so any
  survival analysis on these strata would be post hoc.
