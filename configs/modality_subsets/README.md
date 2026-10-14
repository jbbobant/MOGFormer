# Modality-subset configurations

Five-class PAM50 subtyping on TCGA-BRCA from a subset of the three omics. Each
file is `configs/tcga.yaml` with a different `active_modalities`. That setting
drives the variance-ranked gene selection as well as the features a model sees,
so a subset run is not the trimodal run with columns dropped: it selects the
genes that vary most *in the modalities it uses*. All five configurations share
one folds file, so every per-fold score is paired with every other.

| config | modalities | question |
| --- | --- | --- |
| `configs/tcga.yaml` | expression, copy number, methylation | reference |
| `rna_only_5class.yaml` | expression | is the label recoverable from expression alone? |
| `cnv_only_5class.yaml` | copy number | how much of the label is visible in copy number? |
| `methy_only_5class.yaml` | methylation | how much is visible in methylation? |
| `cnv_methy_5class.yaml` | copy number, methylation | what do the two non-expression omics carry together? |

The transformer's expression-only twin is
`configs/transformer_variants/pretrained_embedding_rna_only_5class.yaml`.

## Predictions, written before any subset was run

PAM50 labels are assigned by a nearest-centroid rule on the expression of 50
genes. The expected pattern follows from that construction, not from data:

1. **Expression alone matches the trimodal model.** For the best classical
   model (XGBoost or random forest), the paired difference in macro-F1 between
   `rna_only` and the trimodal reference has a 95% Nadeau–Bengio interval that
   contains zero.
2. **The non-expression omics carry the label only partially.**
   `cnv_methy` scores well above the stratified dummy floor, because copy
   number and methylation are correlated with subtype biology (basal-like
   genomic instability, luminal methylation programmes), and clearly below
   `rna_only`.
3. **Copy number beats methylation for Basal and HER2-enriched**, the two
   subtypes defined by copy-number-driven programmes (genome-wide instability;
   ERBB2 amplification). Read from per-class F1.
4. **The transformer shows the same pattern as the baselines**: its
   expression-only twin is within noise of its trimodal version.

Prediction 1 failing — the trimodal model reliably beating expression alone —
would contradict the circular-label argument and would be reported as such.

## Running them

Classical baselines, CPU:

```bash
uv run python -m mogformer.cli.experiments baselines --configs configs/modality_subsets/rna_only_5class.yaml configs/modality_subsets/cnv_only_5class.yaml configs/modality_subsets/methy_only_5class.yaml configs/modality_subsets/cnv_methy_5class.yaml
```

## Outcome

Scored on 27 Sep 2026; per-fold scores under `results/experiments/baselines/`
and `results/experiments/transformer_variants/`, paired tests under
`results/experiments/paired_statistics/modality_subsets/`. Differences are
trimodal minus subset; p-values are Nadeau–Bengio corrected, 25 paired
estimates.

| prediction | result | verdict |
| --- | --- | --- |
| 1. Expression alone matches the trimodal model | XGBoost −0.007 (p = 0.73); elastic-net −0.031 (p = 0.12) | holds |
| 2. Copy number + methylation carry the label partially | XGBoost 0.719 vs dummy floor 0.193, and 0.137 below expression only (p = 3.6e-6) | holds |
| 3. Copy number beats methylation for Basal and HER2 | Basal 0.86 vs 0.96, HER2 0.61 vs 0.76 (XGBoost F1) | **fails**: methylation is stronger on every subtype |
| 4. The transformer shows the same pattern | trimodal − expression only = −0.013 (p = 0.43) | holds |

Methylation alone reaches the same macro-F1 as copy number and methylation
together (0.719 for both, XGBoost), so in this cohort copy number adds nothing
over methylation for PAM50 either.
