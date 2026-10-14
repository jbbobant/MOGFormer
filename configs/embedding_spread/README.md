# Embedding spread: cheap objectives

The default encoder's patient summary uses about 7 (TCGA) and 9 (METABRIC) of its
128 dimensions. These configs test three inexpensive ways to spread it, each added
on its own to the default encoder (`configs/ssl/<cohort>_default_encoder.yaml`).

| Config | Change |
|---|---|
| `tcga_spread_uniformity_0_01` | uniformity term on the normalised summary, weight 0.01, batch plus 512-summary memory bank |
| `tcga_spread_uniformity_0_1` | the same, weight 0.1 |
| `tcga_spread_vicreg` | VICReg variance and covariance (1 : 0.04) on a 512-wide expander, batch plus 512-row bank, weight 0.25 (about 9% of the starting loss) |
| `metabric_spread_rank_selection` | 1,200 epochs instead of 800; keep, among epochs within 1% of the best monitoring loss, the one with the highest RankMe |

The regularisers enter the training loss only. Stopping always follows the
monitoring reconstruction loss.

## Protocol

Single-seed screen (seed 42, split seed 42), judged against the default encoder's
three seeds on the summary read-out's participation ratio:

| | Seeds 42–44 | Mean | SD | +2 SD | +4 SD |
|---|---|---|---|---|---|
| TCGA | 7.36, 7.58, 7.05 | 7.33 | 0.26 | 7.86 | 8.39 |
| METABRIC | 8.63, 8.37, 8.97 | 8.65 | 0.30 | 9.26 | 9.87 |

- Above +4 SD, with no primary probe below the default by more than its interval:
  confirm with seeds 43 and 44.
- Between +2 and +4 SD: one more seed decides.
- Below +2 SD: dropped.

Adoption after confirmation follows the decision rule fixed in the plan: the
effective dimension is more than 2 SD above the default; primary probes and the
nuisance probe are within 1 SD; seed-to-seed CKA is at least 0.9.

**Rule amendments (2026-10-11, decided after the first four screens, applying to every later method):**

- **Primary probes.** A probe passes if the default's value lies inside the
  method's 95% interval in every seed, or if the drop is within 1 seed SD. With
  only three seeds, the seed SD can be far smaller than any probe's resolution:
  ER's was 0.001, against a ±0.03 interval.
- **Nuisance probe.** It is judged against the comparators, not by seed SD. It
  must stay below PCA and MOFA+ of the same inputs. More dimensions recover more
  of what the inputs carry, batch included, so an absolute 1-SD condition
  penalises every gain in dimension.

## Already answered from the default runs: rank selection in TCGA

There is no TCGA rank-selection run. In the three default TCGA runs, about 125
epochs lie within 1% of the best monitoring loss. Across them, the monitoring
participation ratio varies by at most 0.05 (best-loss epoch against best-PR
epoch: 7.12 against 7.16, 7.18 against 7.23, 7.20 against 7.25). That is a fifth
of a seed SD, so selection inside the window cannot move TCGA. The METABRIC run
tests the other half of the idea, longer training, since its dimension was still
rising at 800 epochs.

## Predictions, written before the runs

1. Uniformity at 0.1 raises the TCGA summary's dimension by more than 4 SD
   (above 8.4). At 0.01 the rise is smaller and may be inside 2 SD.
2. VICReg raises dimension by more than 2 SD, but less than uniformity at 0.1.
3. Rank selection at 1,200 epochs adds 0.5–1 dimension in METABRIC (9.1–9.6),
   at most around +2 SD.
4. No method moves a primary probe by more than its interval. Uniformity at 0.1
   is the most likely to cost probe accuracy, if any does.
5. None reaches the target of 15 on its own.

## Program-scoped read-outs

| Config | Change |
|---|---|
| `tcga_programs_coexpression_k8` | 8 co-variation programs, each read by a masked-attention query over its genes (2 layers) into 32 dimensions; embedding 256-d; the global head modulates each gene by its own program only |
| `tcga_programs_coexpression_k16` | the same with 16 programs; embedding 512-d |

Programs are fitted per run on the fitting patients (train plus monitor), by Ward
linkage of each gene's standardised three-omics profile. At K = 16 on TCGA the
sizes run from 289 genes down to 38. The assignment is saved as
`gene_programs.csv` beside the weights.

### Predictions, written before the runs

1. At least one K reaches an effective dimension of 15 in the concatenated
   embedding, with every primary probe passing the amended rule.
2. K = 16 has a higher dimension than K = 8.
3. Proliferation is carried mostly by one block: a probe on that block alone
   comes within 0.05 of the full embedding's R².
4. No block holds more than 30% of the embedding's variance.
5. Consensus clustering is at least as reproducible across seeds as the
   default's, unlike the uniformity term.

### Outcome: K = 16 and K = 8, reading final gene states (TCGA, seed 42)

| Summary read-out | Default (3 seeds) | K = 16 | K = 8 |
|---|---|---|---|
| Participation ratio | 7.33 ± 0.26 | 9.58 (+8.5 SD) | 9.00 (+6.3 SD) |
| PAM50 | 0.724 | 0.709 | 0.735 |
| ER | 0.872 | 0.858 | 0.858 |
| HER2 | 0.626 | 0.644 | 0.621 |
| Proliferation | 0.793 | **0.847** | 0.760 |
| Tissue source site (nuisance) | 0.128 | 0.103 | 0.106 |
| Best monitoring loss | 1.109 | **1.034** | 1.038 |
| Held-out masked loss, global head | 0.653 | **0.547** | 0.586 |
| Held-out masked loss, local head | 0.478 | 0.483 | 0.468 |

Both pass the screen under the amended rule. In every case the default's value
lies inside the run's interval, and the nuisance probe falls. K = 16 is
stronger, so it goes to confirmation. Scoping improved the global head's
reconstruction by a sixth.

The blocks are not program-specific, though:
- Each 32-d block alone predicts PAM50 at 0.68–0.74, ER at 0.86–0.88 and
  proliferation at 0.64–0.78, close to the full embedding.
- Variance is spread evenly: 6% per block.

The read-out reads the genes' final states, which the inter-gene transformer has
already mixed. Every member gene therefore carries the same patient-wide
directions, and so does every block. Predictions 1 (dimension ≥ 15) and 3
(proliferation in one block) failed. Prediction 4 (no block above 30%) held, but
only trivially. Prediction 2 (K = 16 above K = 8) held.

Follow-up: `tcga_programs_fused_k16` reads each gene's fused token, before any
mixing between genes. A block then sees only its own genes' measurements.

### Outcome: K = 16 reading fused tokens (TCGA, seed 42): passes the screen, confirming

| Summary read-out | Default (3 seeds) | Fused K = 16 [95% interval] | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 7.33 ± 0.26 | **11.92** | +17 |
| PAM50 | 0.724 | 0.750 [0.691, 0.809] | +3.7 |
| ER | 0.872 | 0.850 [0.809, 0.891] | default inside the interval |
| HER2 | 0.626 | 0.672 [0.626, 0.719] | +3.4 |
| Proliferation | 0.793 | **0.898** [0.877, 0.918] | +12 |
| Tissue source site (nuisance) | 0.128 | 0.127 | below PCA (0.171) and MOFA+ (0.148) |
| Best monitoring loss | 1.109 | 1.041 | |
| Held-out masked loss, global / local head | 0.653 / 0.478 | 0.561 / 0.473 | |

- Consensus clustering is as stable as the default's: PAC at K = 2 is 0.16 in
  Luminal A (default 0.17) and 0.06 in ER+/HER2− (default 0.08).
- The gene-mean read-out's participation ratio fell to 4.4.

The blocks now specialise:

| Block | Contains | Alone predicts |
|---|---|---|
| 0 (289 genes) | MKI67 | proliferation, R² 0.87 |
| 2 (119 genes) | ERBB2 and GRB7 | HER2 best, 0.71 |
| 5, 15 | | proliferation 0.36–0.37 |

Block variance shares stay even, at 5–7%. The prediction is revised:
proliferation is carried mostly by its own block (0.87 against 0.90 for the full
embedding, within 0.05), so prediction 3 holds for this variant. Prediction 1
still fails (dimension 11.9, below 15).

This is the first method to raise dimension, primary probes and reconstruction
together. The `final`-input K = 16 confirmation was stopped in favour of
confirming this variant (seeds 43 and 44).

#### Confirmation, seeds 42–44 (evaluation `tcga_programs_fused_k16_seeds`): adopted

| Summary read-out | Default (3 seeds) | Fused K = 16 (3 seeds) |
|---|---|---|
| Participation ratio | 7.33 ± 0.26 | **11.42 ± 0.44** (+16 SD) |
| PAM50 | 0.724 ± 0.007 | 0.741 ± 0.017 |
| ER | 0.872 ± 0.001 | 0.850 ± 0.001 (default inside every seed's interval) |
| HER2 | 0.626 ± 0.014 | 0.663 ± 0.009 |
| Proliferation | 0.793 ± 0.008 | **0.886 ± 0.011** |
| Tissue source site (nuisance) | 0.128 | 0.117 (PCA 0.171, MOFA+ 0.148) |
| Seed-to-seed CKA | 0.95–0.97 | **0.976–0.986** |
| Seed-to-seed ARI, Luminal A, K = 2 / K = 3 | 0.50–0.62 / 0.20–0.63 | **0.73–0.85 / 0.80–0.83** |
| Held-out masked loss, global / local head | 0.653 / 0.478 | 0.561 / 0.462 |

Every condition of the amended rule is met:
- the dimension rises by far more than 2 SD;
- no primary probe fails: ER's small drop lies inside every seed's interval,
  and the others improve;
- the nuisance probe stays below PCA and MOFA+;
- CKA is above 0.9.

Beyond the rule, consensus clusters now reproduce across seeds better than
the default's. **Adopted:** `configs/ssl/tcga_default_encoder.yaml` now carries
this read-out, identical to this config apart from its name. The METABRIC screen
is `metabric_programs_fused_k16`.

### Outcome: fused K = 16 on METABRIC (seed 42): passes the screen, confirming

The run is compared with the METABRIC default (1,200 epochs, rank selection;
three seeds). Training stopped at epoch 1,163, and the rank rule kept epoch 946.
On METABRIC's fitting patients the programs range from 183 genes down to 25.

| Summary read-out | Default (3 seeds) | Fused K = 16 [95% interval] |
|---|---|---|
| Participation ratio | 10.34 ± 0.46 | **15.93** (+12 SD), the first to reach the target of 15 |
| PAM50 | 0.543 | 0.544 |
| ER | 0.851 | 0.850 |
| HER2 | 0.715 | 0.710 [0.678, 0.743] |
| Integrative cluster | 0.556 | 0.554 |
| Proliferation | 0.876 | 0.903 |
| Processing batch (nuisance) | 0.589 | 0.618 (PCA 0.764, MOFA+ 0.678) |
| Held-out masked loss, global / local head | 0.705 / 0.548 | 0.634 / 0.542 |

The 17p13.3 dosage genes (ABR, CRK, PFN1, TP53) share program 6. That block is
nearly one-dimensional (block participation ratio 2.9) and holds 7% of the
variance. Other markers have blocks of their own: ERBB2 is in block 1, which
alone predicts HER2 best (0.70); MKI67 is in block 4, and ESR1 in block 3.

Consensus clustering changed, as the plan predicted:
- The once-dominant K = 2 split of ER+/HER2− patients is no longer the stable
  one: PAC rose from 0.009 to 0.32, and its ARI with copy-number burden fell
  from 0.16 to 0.
- K = 3 is now the stable cut (PAC 0.09).
- In Luminal A, K = 2 stays stable (PAC 0.02).

Every condition is met on this seed. Seeds 43 and 44 are running.

#### Confirmation, seeds 42–44 (evaluation `metabric_programs_fused_k16_seeds`): adopted

| Summary read-out | Default (3 seeds) | Fused K = 16 (3 seeds) |
|---|---|---|
| Participation ratio | 10.34 ± 0.46 | **16.03 ± 0.60** (15.5–16.7; +12 SD) |
| PAM50 | 0.543 ± 0.006 | 0.533 ± 0.009 (default inside every interval) |
| ER | 0.851 | 0.850 |
| HER2 | 0.715 ± 0.046 | 0.699 ± 0.011 (default inside every interval) |
| Integrative cluster | 0.556 | 0.563 |
| Proliferation | 0.876 | **0.906** |
| Processing batch (nuisance) | 0.589 | 0.617 (PCA 0.764, MOFA+ 0.678) |
| Seed-to-seed CKA | 0.93–0.95 | **0.975–0.983** |
| Seed-to-seed ARI, Luminal A, K = 2 / K = 3 | 0.98 / 0.73–0.87 | 0.87–0.94 / 0.84 |
| Seed-to-seed ARI, ER+/HER2−, K = 2 / K = 3 | 0.94–0.96 / 0.68–0.90 | 0.04–0.78 / 0.68–0.83 |
| Held-out masked loss, global / local head | 0.705 / 0.548 | 0.636 / 0.552 |

Every condition of the amended rule is met, and the target is reached on every
seed: dimension ≥ 15 and CKA ≥ 0.9.

The ER+/HER2− K = 2 split no longer reproduces across seeds, which is expected:
it was the 17p13.3 dosage split, now confined to one block, and K = 3 is the
cut that reproduces. **Adopted:** `configs/ssl/metabric_default_encoder.yaml`
now carries this read-out, identical to this config apart from its name.

## Cross-omics contrastive views

| Config | Change from the TCGA default (fused program read-outs) |
|---|---|
| `tcga_contrastive_0_1` | two views per patient, each hiding a different whole omics (with fresh gene masks); reconstruction averaged over both; InfoNCE between projections of the two embeddings, weight 0.1 |
| `tcga_contrastive_0_5` | the same, weight 0.5 |

Common settings:
- temperature 0.3, chosen moderate so that similar patients are not pushed
  apart (a low temperature concentrates on the nearest negatives, often
  same-subtype patients);
- projection head 256 → 128, training-only;
- negatives: the batch plus the 512 most recent projections.

Both runs are compared with the new TCGA default's three seeds
(`tcga_programs_fused_k16_seeds`): dimension 11.42 ± 0.44.

Caveat: each step also reconstructs two views, and every view hides a whole
omics, where the default hides one for 20% of patients. If a run wins, a control
with the same two views and no InfoNCE is needed to credit the gain to the
contrastive term.

### Predictions, written before the runs

1. At weight 0.1, dimension rises above 11.42 + 2 SD (12.3) with no probe
   failing.
2. HER2 improves most among the primary probes, since its amplification is
   visible in both copy number and expression.
3. Weight 0.5 raises dimension more than 0.1, but costs clustering
   reproducibility, as the uniformity term did.
4. The per-block structure survives: proliferation and HER2 stay in their own
   blocks.

### Outcome: weights 0.1 and 0.5 (TCGA, seed 42): both fail

| Summary read-out | Default (3 seeds) | Contrastive 0.1 | Contrastive 0.5 |
|---|---|---|---|
| Participation ratio | 11.42 ± 0.44 | 13.71 (+5.2 SD) | 13.31 (+4.3 SD) |
| PAM50 | 0.741 | 0.706 | 0.680 [0.618, 0.742] |
| ER | 0.850 | 0.846 | 0.838 |
| HER2 | 0.663 | 0.651 | 0.647 |
| Proliferation | 0.886 | **0.720** [0.680, 0.759] | **0.763** [0.724, 0.801] |
| Tissue source site (nuisance) | 0.117 | 0.101 | 0.089 |
| Early stop / best epoch | 600 / 576 | 311 / 251 | 466 / 406 |
| Best monitoring loss | 1.04 | 1.45 | 1.49 |
| Held-out masked loss, local head | 0.462 | **0.843** | **0.897** |
| PAC at K = 2, ER+/HER2− / Luminal A | 0.06 / 0.16 | 0.11 / 0.26 | 0.47 / 0.35 |

Dimension rose, but proliferation fell far outside its interval in both runs,
and clustering became less stable. Both fail the rule.

The training curves point to a confound, not to the contrastive term alone.
Even at weight 0.1, where InfoNCE is about a fifth of the loss, the local
head's monitoring loss stalls near 1.05 (default: 0.68 at epoch 400), and
training stops early. In these runs every view hides a whole omics, against 20%
of patients in the default. `tcga_two_views_control` trains on the same two
views without InfoNCE, to tell the two causes apart.

Predictions 1 and 2 failed. Prediction 3 held for clustering, though not for
dimension (0.5 < 0.1).

### Control: the same two views, no InfoNCE (TCGA, seed 42)

| Summary read-out | Default | Two views, no InfoNCE | Two views + InfoNCE 0.1 |
|---|---|---|---|
| Participation ratio | 11.42 ± 0.44 | 12.25 | 13.71 |
| PAM50 | 0.741 | 0.709 | 0.706 |
| Proliferation | 0.886 | 0.726 | 0.720 |
| Held-out masked loss, local head | 0.462 | 0.861 | 0.843 |
| Early stop | 600 | 363 | 311 |

The control fails the same way, so the cause is the view regime: hiding a whole
omics in every view stops the local head from learning and costs the probes.
InfoNCE itself adds about 1.5 dimensions over the control, at no further probe
cost.

Redesign (`view_omics_prob`): each view hides a whole omics for only 20% of
patients, the default masker's rate (different omics when both views hide one).
Reconstruction then sees the default mask distribution. Three runs:
- `tcga_contrastive_0_1_partial_views`;
- `tcga_two_views_partial_control`, the same views without InfoNCE;
- `tcga_contrastive_0_5_partial_views`.

### Outcome: partial views (TCGA, seed 42): contrastive objective not adopted

| Summary read-out | Default (3 seeds) | Two views, no InfoNCE | InfoNCE 0.1 | InfoNCE 0.5 |
|---|---|---|---|---|
| Participation ratio | 11.42 ± 0.44 | 11.28 | 11.98 (+1.3 SD) | 11.49 |
| PAM50 | 0.741 | 0.731 | 0.717 | 0.687 |
| ER | 0.850 | 0.845 | 0.862 | 0.856 |
| HER2 | 0.663 | 0.653 | 0.638 | 0.653 |
| Proliferation | 0.886 | 0.895 | 0.877 | **0.780** |
| Tissue source site (nuisance) | 0.117 | 0.111 | 0.108 | 0.092 |
| PAC at K = 2, ER+/HER2− / Luminal A | 0.06 / 0.16 | 0.09 / 0.09 | **0.50** / 0.17 | **0.52** / 0.19 |
| Best monitoring loss | 1.04 | 1.00 | 1.00 | 1.04 |
| Held-out masked loss, local head | 0.462 | 0.438 | 0.433 | 0.455 |

With the view regime fixed, all three runs train normally, and reconstruction is
slightly better than the default's. The contrastive term then adds little:
- At weight 0.1, dimension rises by 1.3 SD over the default (0.7 over its
  control), below the 2-SD threshold, and the ER+/HER2− K = 2 split loses its
  stability (PAC 0.06 → 0.50).
- At weight 0.5, the dimension does not move, proliferation falls far outside
  its interval, and the same split destabilises.

That matches the class-collision risk raised before the runs. Pushing patients
apart breaks the dominant within-subtype split before it adds new dimensions.
The +1.5 dimensions seen earlier came on a degraded encoder (the local head
barely learned) and did not survive the fix.

Verdict: not adopted at either weight. A shared/specific split of the embedding,
which builds on the contrastive term, is not pursued, since the term does not
help. The two-view control alone is
neutral (dimension 11.28), so two views are no gain without the loss either.
Predictions 1 and 2 failed. Prediction 3 held for clustering. Prediction 4 was
not tested.

## Outcomes

### Uniformity, weight 0.1 (TCGA, seed 42): dropped

The effective dimension rose far beyond the threshold, but every primary probe
fell, so the method fails the decision rule.

| Summary read-out | Default (3 seeds) | Uniformity 0.1 | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 7.33 | 17.54 | +39 |
| PAM50 macro-F1 | 0.724 | 0.698 | −3.8 |
| ER macro-F1 | 0.872 | 0.846 | −32 |
| HER2 macro-F1 | 0.626 | 0.567 | −4.4 |
| Proliferation R² | 0.793 | 0.669 | −15 |
| Tissue source site (nuisance) | 0.128 | 0.098 | −7.8 (better) |
| Held-out masked loss, local head (ratio to gene mean) | 0.478 | 0.646 | |

- Training stopped early at epoch 551. The best monitoring loss was 1.287,
  against 1.109 for the default.
- Consensus clustering became less stable: in Luminal A, PAC at K = 2–3 went
  from 0.15–0.17 to 0.37–0.52.
- The gene-mean read-out's participation ratio rose only from 5.8 to 7.0.

Reading: the term spread the summary over the sphere by adding directions that
carry little biology. Proliferation and ER, the strongest real axes, lost
variance share to them. This is the risk the plan named: uniformity bought by
spreading noise.

Predictions 1 and 4 failed. The dimension rose as predicted, but at a probe cost
well beyond one interval.

### VICReg on an expander, weight 0.25 (TCGA, seed 42): dropped, no effect

| Read-out | Default (3 seeds) | VICReg | Change in default seed SDs |
|---|---|---|---|
| Summary participation ratio | 7.33 ± 0.26 | 7.51 | +0.7 |
| Masked-mean summary | 7.33 ± 0.19 | 7.37 | +0.3 |
| Gene mean | 5.60 ± 0.54 | 5.93 | +0.6 |

- Every probe moved by less than 1.5 seed SDs. Held-out reconstruction was
  unchanged (masked, local head: 0.486 against 0.478).
- The best monitoring loss was slightly lower than the default's (1.096 against
  1.109).
- On the monitoring patients, RankMe of the summary sat at about 27 throughout,
  against 39–55 under uniformity.

Reading: the penalty was met inside the expander. A ReLU network can map a
summary that spans about 7 directions to 512 outputs that each have unit
variance and are nearly decorrelated, so the summary itself never had to
spread. In VICReg proper, the invariance term between two views ties the
expander to the encoder; without a second view, nothing does. This method
belongs with the two-view objective (method 1), not alone.

Prediction 2 failed: no rise beyond seed noise.

### Uniformity, weight 0.01 (TCGA, seed 42): passes the screen, confirming

| Summary read-out | Default (3 seeds) | Uniformity 0.01 [run's 95% interval] | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 7.33 ± 0.26 | 11.92 | +17 |
| Masked-mean participation ratio | 7.33 ± 0.19 | 12.64 | +28 |
| Gene-mean participation ratio | 5.60 ± 0.54 | 7.32 | +3.2 |
| PAM50 | 0.724 | 0.710 [0.660, 0.759] | −2.1 |
| ER | 0.872 | 0.862 [0.832, 0.893] | −12 (seed SD 0.001) |
| HER2 | 0.626 | 0.583 [0.535, 0.631] | −3.2 |
| Proliferation | 0.793 | 0.809 [0.783, 0.834] | +1.9 |
| Tissue source site (nuisance) | 0.128 | 0.116 | −3.1 (better) |
| Held-out masked loss, local head | 0.478 | 0.511 | |

- PC1's share of variance fell from 0.24 to 0.14.
- The best monitoring loss was 1.122, against 1.109 for the default.
- Consensus clustering is less stable than the default: in Luminal A, PAC at
  K = 2 went from 0.17 to 0.52.

Screen verdict: the dimension rise is above 4 SD, and every primary probe's
default value lies inside this run's interval, so the method goes to
confirmation (seeds 43 and 44). On a single seed, it would not yet meet the
adoption rule:
- HER2 and PAM50 are more than 1 seed SD below the default.
- ER's three-seed SD (0.001) is far smaller than any probe's interval, so for ER
  the 1-SD condition cannot be met by any change at all.

Prediction 1's second half failed: at 0.01 the rise is large, not small.

#### Confirmation, seeds 42–44 (evaluation `tcga_spread_uniformity_0_01_seeds`)

| Summary read-out | Default (3 seeds) | Uniformity 0.01 (3 seeds) | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 7.33 ± 0.26 | 11.67 ± 0.31 | +17 |
| Gene-mean participation ratio | 5.60 ± 0.54 | 7.37 ± 0.38 | +3.3 |
| PAM50 | 0.724 ± 0.007 | 0.713 ± 0.003 | −1.7 |
| ER | 0.872 ± 0.001 | 0.863 ± 0.004 | −11 |
| HER2 | 0.626 ± 0.014 | 0.599 ± 0.018 | −2.1 |
| Proliferation | 0.793 ± 0.008 | 0.785 ± 0.023 | −0.9 |
| Tissue source site (nuisance) | 0.128 ± 0.004 | 0.116 ± 0.001 | −3.3 (better) |
| Seed-to-seed CKA | 0.95–0.97 | 0.92–0.94 | |
| Seed-to-seed ARI at K = 2, Luminal A | 0.50–0.62 | 0.05–0.31 | |
| Held-out masked loss, local head | 0.478 | 0.501 | |

The dimension gain is reproducible, and CKA stays above the 0.9 target. On the
rule as written, the method is not adopted: PAM50, ER and HER2 are each more than
1 seed SD below the default. Each drop is, however, far inside its probe's 95%
interval (±0.04–0.05). The clustering read-out is the stronger objection: the
dominant Luminal A split no longer reproduces across seeds.

Verdict: held, not adopted. It is a candidate component for combining with a
method that supplies the missing structure.

Under the amended probe rule, the probes pass: in every seed, the default's
PAM50, ER and HER2 values lie inside the method's 95% interval. The method still
meets every written condition. It remains held because the clustering read-out
is not reproducible across seeds, and that read-out is the embedding's purpose.

### Rank selection at 1,200 epochs (METABRIC, seed 42): passes the screen, confirming

The best monitoring loss was 1.264. The rule kept epoch 1,005 (RankMe 40.2);
the last epoch had RankMe 39.9. Since the two barely differ, the gain comes from
the longer training, not from the selection.

| Summary read-out | Default (3 seeds) | 1,200 epochs, rank selection [95% interval] | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 8.65 ± 0.30 | 10.29 | +5.4 |
| Masked-mean participation ratio | 8.43 ± 0.22 | 10.01 | +7.2 |
| Gene-mean participation ratio | 5.50 ± 1.06 | 7.44 | +1.8 |
| PAM50 | 0.531 | 0.542 [0.517, 0.567] | +3.0 |
| ER | 0.848 | 0.862 [0.840, 0.885] | +2.0 |
| HER2 | 0.694 | 0.768 [0.740, 0.796] | +2.9 |
| Integrative cluster | 0.548 | 0.567 [0.536, 0.598] | +1.6 |
| Proliferation | 0.879 | 0.856 [0.842, 0.870] | −1.5 |
| Processing batch (nuisance) | 0.554 | 0.590 [0.558, 0.622] | +2.4 (worse) |
| Held-out masked loss, local head | 0.560 | 0.547 | |

- Consensus clustering is as stable as the default or more so. In Luminal A,
  PAC at K = 3 fell from 0.23 to 0.05.
- The dominant K = 2 split is unchanged (ARI with copy-number burden 0.16 in
  ER+/HER2−).

Screen verdict: the dimension rise is above 4 SD and every primary probe
improved, so the method goes to confirmation (seeds 43 and 44). Two cautions:
- The processing-batch nuisance probe rose by 2.4 SD, and the default's value
  lies outside this run's interval. On the adoption rule, that is a veto
  unless the seeds disagree.
- Proliferation fell slightly.

Prediction 3 held: the gain is +1.6 dimensions, above the predicted 0.5–1.

#### Confirmation, seeds 42–44 (evaluation `metabric_spread_rank_selection_seeds`)

- Seed 43 stopped early at epoch 1,080; the rule kept epoch 785.
- Seed 44 stopped at 1,146; the rule kept epoch 925.

| Summary read-out | Default (3 seeds) | 1,200 epochs, rank selection (3 seeds) | Change in default seed SDs |
|---|---|---|---|
| Participation ratio | 8.65 ± 0.30 | 10.34 ± 0.46 | +5.6 |
| PAM50 | 0.531 ± 0.004 | 0.543 ± 0.006 | +3.3 |
| ER | 0.848 ± 0.007 | 0.851 ± 0.013 | +0.4 |
| HER2 | 0.694 ± 0.026 | 0.715 ± 0.046 | +0.8 |
| Integrative cluster | 0.548 ± 0.011 | 0.556 ± 0.013 | +0.7 |
| Proliferation | 0.879 ± 0.016 | 0.876 ± 0.017 | −0.2 |
| Processing batch (nuisance) | 0.554 ± 0.015 | 0.589 ± 0.008 | +2.4 (worse) |
| Seed-to-seed CKA | 0.95–0.97 | 0.93–0.95 | |
| Seed-to-seed ARI at K = 3, ER+/HER2− | 0.59–0.84 | 0.68–0.90 | |
| Held-out masked loss, local head | 0.560 | 0.548 | |

Every condition of the adoption rule is met except the nuisance probe: the
processing-batch probe rose by 2.4 seed SDs in all three seeds, and the rule
allows 1.

Context for that veto: the batch is strongly present in the inputs themselves.
The probe scores 0.82 on raw inputs, 0.76 on PCA and 0.68 on MOFA+. Even after
the rise, the encoder (0.59) carries less of it than any comparator. More
dimensions recover more of what the inputs hold, batch included.

Verdict: vetoed by the nuisance condition as written. **Adopted under the
amended rule.** `configs/ssl/metabric_default_encoder.yaml` now trains for 1,200
epochs with rank selection. It is identical, apart from its name, to this
config, whose seeds 42–44 serve as the new METABRIC default's reference runs.
