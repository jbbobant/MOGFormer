# Encoder ablation — leave one group out, pre-registered

Four groups of changes may plausibly enrich the self-supervised encoder, but
none is guaranteed to. Rather than one run per change, the encoder is trained
once with every group on (`*_all_groups.yaml`), and once with each group
switched back to the reference encoder's setting (`*_without_<group>.yaml`).
The reference encoders are `configs/ssl/tcga.yaml` and `configs/ssl/metabric.yaml`,
three seeds each, already evaluated.

| Group | Switched on | Reference setting |
| --- | --- | --- |
| `masking` | 35% of genes hidden whole, half of them as a gene plus its interaction neighbours; one whole omics hidden for 20% of patients | 15% of genes hidden whole, gene by gene; no omics-level masking |
| `identity_modulation` | gene identity scales and shifts each value token (zero-initialised FiLM) | identity added beside the value |
| `summary_tokens` | 4 summary tokens (embedding 512 wide), read by a cross-attention global head | 1 summary token, scale-only (FiLM) global head |
| `fusion_residual` | a zero-initialised linear residual of the three unfused omics tokens | the gate's convex combination alone |
| copy-number scale (TCGA only) | `log2(CN / 2)`, which keeps absolute dosage and ploidy | `log2(CN / ploidy)` |

The copy-number scale swaps a setting rather than adding one, so it is run as
`tcga_all_groups_log_ratio_copy_number.yaml` against `tcga_all_groups.yaml`.
The other runs keep the ploidy-relative scale decided earlier, so the ploidy
axis cannot return and confound the other four comparisons.

Runs: TCGA 6, METABRIC 5, one seed (42) each, with the reference encoders'
epoch budgets. Every run is scored by the evaluation suite against the same
comparators, MOFA+ included (at most 128 factors).

## Readouts and decision rule, fixed before the runs

Primary readouts:
- the mean macro-F1 of the classification probes (PAM50, ER, HER2, and in
  METABRIC the integrative clusters), on the embedding's best read-out;
- held-out reconstruction under the reference mask (E5, mean loss / gene-mean
  baseline over the three omics);
- the participation ratio (E3).

Seed noise for each readout is the standard deviation across the reference
encoder's three seeds.

- A group stays in the default if removing it worsens at least one primary
  readout by more than one seed SD, and improves none by more than one SD.
- A group that makes no difference stays only if it is cheap and needed later.
  That applies to omics-level masking, which the shared/specific latents of
  the next step require.
- If the all-groups encoder's participation ratio stays below 15, a sixth
  group is tested afterwards: a covariance penalty on the embedding, and the
  global-head loss weight at 0.5, 1 and 2.
- The copy-number scale: `log_ratio` is adopted only if it improves a primary
  readout beyond seed noise without the clustering checkpoint vetoing on the
  ploidy split (ARI ≥ 0.8).
- The clustering checkpoint can veto a configuration, never select one.

## The sixth group: triggered

The TCGA all-groups encoder's participation ratio is 5.1 on a 512-wide
embedding (first component: 39% of its variance), below 15. The sixth group is
therefore tested on TCGA with three runs, each the all-groups encoder plus one
setting:
- a decorrelation penalty on the summary embedding, weight 1;
- the global-head loss weight at 0.5;
- the global-head loss weight at 2.

The plan's covariance penalty is computed on correlations, not raw
covariances. A raw covariance can be driven to zero by shrinking the
embedding, which costs nothing because the cross-attention head
layer-normalises the summary tokens. VICReg blocks this with a variance term;
correlations are immune to scale. The weight was set from the penalty's size,
not from any outcome: about 1.8 per batch on the all-groups encoder, against a
reconstruction loss of about 1.6. Batches of 32 patients estimate a 512-wide
correlation matrix only at low rank, so the penalty is a weak regulariser.

METABRIC runs of the sixth group follow only if one of these settings passes
the decision rule on TCGA.

Prediction, written before these runs: the decorrelation penalty raises the
participation ratio beyond seed noise without lowering the probes; neither
global-head weight moves any primary readout beyond seed noise.

## Outcome, TCGA (one seed per run; `results/experiments/encoder_ablation/encoder_card_tcga.md`)

Effect of having each group (all groups minus without it), in reference-seed
standard deviations; positive means the group helps:

| Group | Probe macro-F1 | Held-out loss (reference mask) | Participation ratio | Verdict by the rule |
| --- | --- | --- | --- | --- |
| masking | −3.8 | −10.1 | −1.3 | drop |
| summary tokens | −1.2 | +0.5 | −4.2 | drop |
| identity modulation | +1.0 | +10.1 | −1.8 | mixed |
| fusion residual | −1.0 | +8.7 | −0.7 | keep (probes at the threshold) |

- Masking is the only group that makes cross-omics reconstruction beat the
  gene mean: 0.77 with it, 1.28 without. Its held-out penalty is partly
  built in: the encoder without it trains on exactly the reference mask it is
  scored on.
- The log-ratio copy-number scale brings ploidy back. Ploidy explains 0.28–0.36
  of the embedding's variance (PC1 R² 0.84–0.91), and the checkpoint vetoes on
  the ploidy split (ARI 0.90–0.94 in both populations). Rejected.
- Every encoder with the tested groups has a participation ratio of 5.0–6.1,
  below the reference's 7.8–8.3.

Predictions: 1 failed (PR stays near 5; masking and summary tokens do not
carry gains); 2 held (cross-omics beats the gene mean only with masking);
3 failed (identity modulation and the residual improve reconstruction far
beyond noise); 4 held; 5 held (nuisance probe 0.10–0.13 against 0.10–0.11).

## Outcome, METABRIC (one seed per run; `results/experiments/encoder_ablation/encoder_card_metabric.md`)

| Group | Probe macro-F1 | Held-out loss (reference mask) | Participation ratio | Verdict by the rule |
| --- | --- | --- | --- | --- |
| masking | −0.9 | −28.3 | −1.6 | drop |
| summary tokens | −0.3 | −11.2 | −1.7 | drop |
| identity modulation | +2.6 | +6.3 | +0.2 | keep |
| fusion residual | +2.0 | +5.8 | −0.3 | keep |

The verdicts agree with TCGA's. The held-out SD units are inflated, because
the three reference seeds agree almost exactly on held-out loss (SD 0.001):
read them as direction only. Cross-omics reconstruction again needs masking
(0.84 with it, 1.02 without). The all-groups participation ratio is 8.2,
within the reference's 7.3–9.5: METABRIC does not show TCGA's collapse.

### Decisions taken after the TCGA results (not pre-registered)

- **Identity modulation** ("mixed"): kept. Reconstruction improves by 10 SD,
  and probes by 1 SD; the participation ratio falls by 1.8 SD (5.0 against 5.5).
- **Masking**: the omics-level part is needed downstream and is the only source
  of cross-omics reconstruction. One extra run separates it from the harder
  whole-gene masks:
  `tcga_one_summary_token_omics_level_masking_only.yaml`, compared with
  `tcga_without_summary_tokens.yaml`, which differs only by the 35% whole-gene
  masks with neighbourhoods. The harder masks stay in the default only if the
  decision rule keeps them on that comparison.
- **Sixth group**: run on the provisional default instead of the all-groups
  encoder: one summary token with the scale-only head, identity modulation,
  the fusion residual, and the masking the run above selects. The summary
  tokens it was first built on are dropped. Its configurations are written
  once the masking is settled.

### Masking split (TCGA, one summary token)

| Encoder | Probe macro-F1 | Held-out loss | Participation ratio | Cross-omics loss |
| --- | --- | --- | --- | --- |
| omics-level masking only | 0.729 | 0.658 | 7.1 | 0.76 |
| omics-level + 35% whole-gene masks with neighbourhoods | 0.741 | 0.556 | 6.1 | 0.77 |

Omics-level masking alone yields cross-omics reconstruction. The harder
whole-gene masks help the probes (+1.7 SD) and held-out reconstruction (far
beyond noise) but lower the participation ratio (−3.7 SD): "mixed". This
follows the precedent set for identity modulation (a large reconstruction gain,
no probe loss, a moderate fall in participation ratio: keep), so the harder
masks are kept.

**Candidate default**: `tcga_without_summary_tokens.yaml` (all masking,
identity modulation, fusion residual, one summary token with the scale-only
global head, ploidy-relative copy number). The sixth group runs on it:
`tcga_default_candidate_{decorrelation,global_head_weight_0_5,global_head_weight_2}.yaml`,
with that run (decorrelation 0, global-head weight 1) as the baseline.

### Sixth group, TCGA (on the candidate default)

| Encoder | Probe macro-F1 | Held-out loss | Participation ratio | Cross-omics loss |
| --- | --- | --- | --- | --- |
| candidate default (global-head weight 1) | 0.741 | 0.556 | 6.1 | 0.77 |
| + decorrelation penalty (weight 1) | 0.697 | 0.936 | 2.4 | 0.94 |
| global-head weight 0.5 | 0.742 | 0.495 | 7.3 | 0.73 |
| global-head weight 2 | 0.732 | 0.664 | 7.7 | 0.80 |

- The decorrelation penalty fails on every readout: the monitoring loss stalls
  at 2.23 (baseline 1.62).
- A global-head weight of 0.5 improves held-out reconstruction (about 7.6 SD)
  and the participation ratio, leaves the probes unchanged and worsens
  nothing: kept.
- The prediction for this group failed: the penalty did not raise the
  participation ratio, and a global-head weight did move the readouts.

As pre-registered, a setting passed on TCGA, so it is checked on METABRIC
(`metabric_default_candidate_global_head_weight_0_5.yaml` against
`metabric_without_summary_tokens.yaml`). The TCGA default
(`configs/ssl/tcga_default_encoder.yaml`) is trained with three seeds.

### Sixth group, METABRIC check

| Encoder | Probe macro-F1 | Held-out loss | Participation ratio | Cross-omics loss |
| --- | --- | --- | --- | --- |
| candidate default (global-head weight 1) | 0.662 | 0.571 | 10.3 | 0.83 |
| global-head weight 0.5 | 0.666 | 0.567 | 8.6 | 0.82 |

The result is "mixed": probes about +1 SD, held-out reconstruction better, the
participation ratio about 1.4 SD lower. By the same precedent the setting is
kept, so both cohorts share one default (`configs/ssl/tcga_default_encoder.yaml`,
`configs/ssl/metabric_default_encoder.yaml`), each trained with three seeds.

## The default encoder, three seeds per cohort

Suite outputs: `results/experiments/representation_evaluation/{tcga,metabric}_default_encoder/`.

| Readout (best read-out per seed) | TCGA reference | TCGA default | METABRIC reference | METABRIC default |
| --- | --- | --- | --- | --- |
| Probe macro-F1 | 0.722–0.736 | 0.742–0.746 | 0.650–0.657 | 0.646–0.666 |
| Held-out loss / gene mean | 0.64–0.66 | 0.47–0.50 | 0.556–0.558 | 0.555–0.567 |
| Cross-omics loss / gene mean | 1.06–1.14 | 0.73–0.74 | 0.99–1.04 | 0.81–0.82 |
| Participation ratio | 7.8–8.3 | 7.1–7.5 | 7.3–9.5 | 8.2–9.0 |
| Nuisance probe (lower is better) | 0.10–0.11 | 0.12–0.13 | 0.56–0.58 | 0.54–0.57 |
| Linear CKA between seeds (summary) | 0.94–0.95 | 0.95–0.97 | 0.87–0.95 | 0.95–0.97 |

The clustering checkpoint does not veto in either cohort (ploidy share 0.013 in
TCGA; ARI with the burden split ≤ 0.16). In METABRIC, the default encoder
splits both ER+/HER2− (370 / 517) and Luminal A in two with very high
stability (PAC 0.002–0.008). The split is copy number at 17p13.3: ABR, CRK,
PFN1, MYO1C, PITPNA, TIMM22 and 24 more, each AUC 0.98–0.99. It is not the
processing cohort, integrative cluster, PAM50, HER2, cellularity or
menopausal state (ARI ≤ 0.06). A single regional event now dominates the
METABRIC structure, the case the genomic prior's window masking is meant to
handle.

## Predictions, written before the runs

1. Masking and summary tokens carry most of the gain. With all groups on,
   the participation ratio reaches at least 15. Removing either group lowers it
   by more than seed noise.
2. Cross-omics reconstruction (E6) beats the gene mean only when masking is on
   (omics-level masks are its training signal).
3. Identity modulation and the fusion residual are within seed noise on every
   primary readout.
4. With `log_ratio` copy number, the ploidy share of the embedding returns
   above 0.05, and the checkpoint vetoes on the ploidy split in TCGA Luminal A.
5. No group changes the nuisance probes beyond seed noise.
