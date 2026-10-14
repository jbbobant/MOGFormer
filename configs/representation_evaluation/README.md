# Representation evaluation — pre-registered predictions

The evaluation suite scores a frozen patient representation (an encoder's
embedding, or a comparator built from the same inputs) on linear probes,
effective dimension, the share of it one covariate explains, and a
structure-only clustering checkpoint. No survival information is read.

Each prediction below was written before the run it concerns.

## Archived TCGA encoder against its comparators (748 patients)

Prediction, written before the run:

- The archived encoder's embedding does not beat PCA of the concatenated
  omics at equal dimension on the probe panel (mean |Δ| ≤ 0.02).
- Its first principal component explains ≥ 50% of the embedding's variance
  and correlates with ploidy at |ρ| ≥ 0.8.

Outcome: both parts failed (see
`results/experiments/representation_evaluation/E600_p60_gated_meta/`).

## Copy number relative to ploidy, comparators only (748 TCGA patients)

Inputs are rebuilt twice from the same cohort archive with the same
preprocessing (250 variance-ranked genes plus the curated list, per-gene
standardisation): once with absolute copy number, once with
`log2(CN / ploidy)` clipped to ±2. PCA of all omics, PCA per omics and the raw
inputs are scored on both. Results:
`results/experiments/representation_evaluation/ploidy_relative_inputs/`.

Predictions, written before the run:

1. The share of variance explained by ploidy in PCA of all omics falls from
   0.18 to below 0.05.
2. PCA's first component is no longer ploidy (R² < 0.2).
3. The Luminal A and ER+/HER2− two-cluster splits of PCA no longer match the
   ploidy split (ARI < 0.3).
4. PAM50 and ER probes on PCA do not fall by more than 0.02.
5. Per-gene Spearman ρ between expression and copy number rises with relative
   copy number for the majority of genes (paired Wilcoxon over genes).

If (3) fails, the dominant split is not only ploidy: that is reported as such,
and arm-level burden becomes the next suspect in the checkpoint.

Outcome (rebuilt inputs: 475 genes, 250 variance-ranked plus 225 curated; the
absolute arm reproduces the archived inputs' axes — ploidy share 0.18,
participation ratio 16.1):

| Prediction | Absolute | Ploidy-relative | Held |
| --- | --- | --- | --- |
| 1. Ploidy share of PCA variance | 0.184 | 0.008 | yes |
| 2. PC1 vs ploidy, R² | 0.965 | 0.000 | yes |
| 3. K = 2 split vs ploidy split, ARI (Luminal A / ER+/HER2−) | 0.98 / 0.95 | −0.003 / 0.08 | yes |
| 4. PAM50 / ER probe change on PCA | — | +0.002 / −0.008 | yes |
| 5. Genes whose expression–CN ρ rises (median ρ) | (0.11) | 87% (0.19), Wilcoxon p < 1e−300 | yes |

The burden probe falls with ploidy (R² 0.78 → 0.24): burden as defined here
(fraction of genes whose absolute copy number is not two) is mostly ploidy.
Without the ploidy axis, PCA's effective dimension rises from 16.1 to 26.9 and
no population has a stable two-cluster split any longer (PAC 0.12 in Luminal A,
0.63 in ER+/HER2−).
