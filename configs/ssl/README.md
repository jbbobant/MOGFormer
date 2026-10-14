# Self-supervised encoder — configuration and pre-registered predictions

`tcga.yaml` and `metabric.yaml` pretrain the masked multi-omics encoder
outcome-blind, one run per seed:

```bash
uv run python -m mogformer.cli.experiments ssl_pretraining --config configs/ssl/tcga.yaml --bundle data/bundle/tcga --seeds 42 43 44
uv run python -m mogformer.cli.experiments ssl_pretraining --config configs/ssl/metabric.yaml --bundle data/bundle/metabric/ssl --seeds 42 43 44
```

and each configuration's seeds are then evaluated together:

```bash
uv run python -m mogformer.cli.experiments representation_evaluation --cohort tcga --encoder-runs results/experiments/ssl_pretraining/tcga_ssl_encoder_seed42 results/experiments/ssl_pretraining/tcga_ssl_encoder_seed43 results/experiments/ssl_pretraining/tcga_ssl_encoder_seed44 --name tcga_ssl_encoder
```

## What differs from the archived encoders, and why

None of these choices needs an ablation: each removes a defect or a wasted
resource, or aligns the objective with the data. Their joint effect is
measured once, by the evaluation suite.

| Setting | Archived encoder | This configuration | Reason |
| --- | --- | --- | --- |
| Copy number (TCGA) | absolute copies, standardised | `log2(CN / ploidy)`, clipped to ±2 | a uniform genome doubling dominated the inputs' cluster structure |
| Copy number (METABRIC) | ordinal codes 0–4, standardised | codes − 2 (−2…2) | a bounded log-ratio surrogate on the same footing as TCGA's |
| Expression (METABRIC) | z-scores log-transformed as counts (clipped below zero) | z-scores standardised as they are | the archived run saw every below-average value as zero |
| Standardised values | unbounded | clipped to \|z\| ≤ 5 | rare amplifications reached z ≈ 25 as inputs and as targets |
| Variance ranking of expression | raw counts | log expression | the model reads log expression |
| Gene pool | 250 variance-ranked + curated (433) | 1000 or 2000 variance-ranked + curated | less steering by a hand-picked list |
| Identity table | one row per gene of the universe (17,136 rows, 2.19 M parameters, 433 used) | one row per selected gene | rows that are never selected never learn |
| Interaction prior | STRING combined score ≥ 400, binary | experimental + database channels ≥ 400, confidence-weighted random walk | the co-expression channel made the prior a restatement of the data |
| Genes without an edge | one shared, all-zero positional code | a learned code of their own | they were indistinguishable by position |
| Imputed entries | scored as reconstruction targets | never scored | the model was rewarded for predicting the imputation median |
| Loss pooling | over all masked entries | mean within each omics, then summed | expression was masked twice as often and so weighed twice as much |
| Global decoder head | scale and offset by the summary | scale only | the offset is a per-patient constant added to every gene |
| Bias-gate learning rate | 1000 × base | 10 × base | the gates had become switches (9 of 24 heads off) |
| Schedule | constant learning rate | 20 warm-up epochs, cosine decay | standard for small transformers |
| Patient splits | train / monitor | train (78%) / monitor (12%) / held out (10%), fixed across seeds | the first measure of generalisation |
| Monitoring masks | redrawn each epoch | one fixed draw | stopping on a noisy criterion |
| Attention | explicit score tensor | fused kernel, bias as additive mask | same mathematics; memory scales to 2,000+ genes |
| Read-outs | summary token | summary token; its mean over 10 masks; mean of gene states | the suite chooses |

## Predictions, written before the runs

For the TCGA encoder (three seeds), against the archived encoder's suite
baseline (`results/experiments/representation_evaluation/E600_p60_gated_meta`):

1. Held-out reconstruction beats the gene-mean baseline in every omics.
2. The participation ratio rises above the archived 8.3, and no probe target
   loses more than 0.02.
3. The ploidy share of the embedding's variance falls below 0.05 (from 0.18),
   and the clustering checkpoint no longer vetoes on the ploidy split
   (ARI < 0.8 with it in both populations).
4. The burden share may become the largest single axis; it is watched.

### Outcome, TCGA (three seeds, 1,225 genes, 748 patients; suite under `results/experiments/representation_evaluation/tcga_ssl_encoder/`)

Every run trained the full 600 epochs (68–80 min each on an RTX 5060 Laptop
GPU; 0.79 M parameters against the archived 2.86 M).

| Prediction | Result | Held |
| --- | --- | --- |
| 1. Held-out masked reconstruction beats the gene mean in every omics | loss / baseline 0.62 (expression), 0.70 (copy number), 0.61 (methylation) | yes |
| 2a. Participation ratio above 8.3 | 7.0–8.3 over seeds and read-outs | no |
| 2b. No probe loses more than 0.02 | ER 0.880 → 0.87; PAM50 0.797 → 0.72; HER2 0.632 → 0.58; proliferation 0.957 → 0.76 | no |
| 3. Ploidy share < 0.05, no veto on the ploidy split | share 0.012–0.014; ARI with the ploidy split ≤ 0.07 | yes |
| 4. Burden as largest axis | burden share 0.03; not dominant | — |

Further results:
- Cross-omics reconstruction (one omics hidden entirely) is at or worse than
  the gene mean (ratios 1.00–1.15). That is expected for an encoder never
  trained with omics-level masking, and it is the baseline for that masking.
- Seed stability: linear CKA between seeds is 0.90–0.96 (criterion ≥ 0.8). The
  consensus partitions of Luminal A do not agree across seeds (ARI ≤ 0.25 at
  K = 2).
- The tissue-source-site nuisance probe falls from 0.136 to 0.10–0.12.
- Against PCA of the same inputs, the encoder is better on ER (0.87 vs 0.82),
  equal on PAM50 (0.72), and worse on HER2 (0.58 vs 0.68), proliferation
  (0.76 vs 0.96) and age (0.08 vs 0.32).
- The read-outs differ little. The mean of the summary token over masks is the
  most consistent: the highest effective dimension and CKA, and the best or
  equal-best probes.

Reading: removing ploidy from the inputs removed it from the embedding as
intended, and reconstruction now generalises to unseen patients. However, the
embedding did not gain effective dimensions to replace the ploidy axis, and
the archived encoder's advantage over PCA on PAM50 and ER was not reproduced
(only ER remains). The new encoder is no richer than PCA of its own inputs.
The summary token is used: the decoder head that reads only the summary
reconstructs held-out values as well as the head that also reads the gene's
own state (loss 0.30 vs 0.31 on copy number). It is low-rank, however. These
runs alone cannot say which of the changes made at once is responsible. The
masking, read-out and regularisation groups of the next step are the tests.

### Outcome, METABRIC (three seeds, 1,182 genes, 1,416 patients; suite under `results/experiments/representation_evaluation/metabric_ssl_encoder/`)

All runs trained the full 800 epochs (162–240 min each).

- Held-out masked reconstruction beats the gene mean in every omics. Loss /
  baseline: expression 0.67, methylation 0.86, copy number 0.14 (local head)
  and 0.57 (global head). The gap between the two heads on copy number shows
  how much of a hidden call the gene's own state recovers.
- Cross-omics reconstruction is at the gene mean (ratios 0.97–1.15), as in
  TCGA.
- Participation ratio: 9.3–10.6 (summary read-outs) and 6.8–7.4 (mean of gene
  states).
- Copy-number burden is the largest covariate axis. Burden share is 0.09–0.12
  (PCA 0.05), and the burden probe reaches R² 0.98 (PCA 0.76). The clustering
  checkpoint does not veto on it: ARI with the burden split is ≤ 0.17.
- Against PCA of the same inputs, the encoder ties on ER (0.85–0.86). It is
  worse on HER2 (0.66–0.68 vs 0.82), integrative cluster (0.54 vs 0.58),
  PAM50 (0.54 vs 0.56), proliferation (0.87–0.89 vs 0.94) and age (0.17 vs
  0.32).
- The processing-cohort nuisance probe is much lower than PCA's (0.55–0.58 vs
  0.76).
- Seed stability: linear CKA 0.87–0.95.

For METABRIC the same three seeds are trained; its suite (probes against
PAM50, ER, HER2, integrative cluster, proliferation, burden, age, and the
processing cohort as nuisance) has no archived baseline, because the archived
METABRIC encoder saw clipped expression.
