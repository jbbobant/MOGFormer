# METABRIC targets that are not derived from expression

On TCGA-BRCA, PAM50 subtypes are predicted as well from expression alone as
from all three omics. PAM50 is *called* from expression centroids, so that
result may say more about the label than about the omics. These
configurations test that directly. METABRIC carries labels derived from
different measurements, and each is predicted from each omics subset:

| Target (`label_col`) | Classes | Derived from |
| --- | --- | --- |
| `SUBTYPE` (PAM50) | 5 | expression centroids |
| `INTCLUST` | 11 (cluster 4 split by ER) | copy number jointly with expression (Curtis et al. 2012) |
| `HER2_SNP6` | 3 (gain / neutral / loss; 2 `UNDEF` excluded) | ERBB2 copy number from SNP6 arrays |
| `ER_IHC` | 2 | protein, by immunohistochemistry |

Each target has its own bundle directory (`data/bundle/metabric/<target>/`) and
its own 5×5 stratified partition, shared by the four subsets
(`all_omics`, `rna_only`, `cnv_only`, `methy_only`). Scores within a target are
therefore paired. Every other setting is copied from `configs/tcga.yaml`, and
ERBB2 and ESR1 are in the forced-in curated gene list, so every subset sees
the genes these labels are defined on.

Methylation is read as M-values, `log2(β / (1 − β))` with β clipped to
[0.001, 0.999]. That is the transform behind TCGA's M-values, applied to
METABRIC's gene-level promoter beta values by `utils.build_cohort_bundle`.
Classical models and MOGFormer both read this representation.

## Predictions, written before any run

Macro-F1 over 25 estimates; "beats" means a positive paired difference with
Nadeau–Bengio corrected p < 0.05, for the best classical model on each target.

1. **PAM50 (circular control).** All omics does not beat expression only, as
   on TCGA. Copy number only and methylation only are clearly below both.
2. **HER2_SNP6 (copy-number positive control).** Copy number only beats
   expression only. All omics does not beat copy number only.
3. **INTCLUST (partly copy-number-defined).** All omics beats expression only.
   Copy number only beats methylation only.
4. **ER_IHC (protein).** Expression only is strong (ESR1), and all omics does
   not beat it. Copy number only is the weakest subset.

If prediction 2 or 3 fails, the argument that fusion is uninformative on
TCGA only because PAM50 is expression-derived is weakened, and the report
must say so.

## Outcome

Paired differences in macro-F1 over 25 estimates, Nadeau–Bengio corrected p
(`results/experiments/paired_statistics/metabric_<target>/`). Classical models:
linear SVM and random forest. METABRIC expression is read as distributed
(z-scores, `preprocess.rna_scale: log`); earlier runs that log-transformed it
as counts are archived and not used. MOGFormer (gated fusion, final protocol:
at least 100 epochs, epoch chosen on a 5-epoch moving average of inner
macro-F1) was run on HER2 status and integrative clusters only.

| Prediction | Linear SVM | Random forest | MOGFormer | Verdict |
| --- | --- | --- | --- | --- |
| 1. PAM50: all omics does not beat expression | −0.045 (0.047) | −0.019 (0.23) | — | **holds** (all omics is worse for the SVM) |
| 1. PAM50: copy number and methylation clearly below | expression − copy number +0.276 (2e−11) | +0.228 (3e−11) | — | **holds** |
| 2. HER2: copy number beats expression | +0.041 (0.024) | +0.108 (0.0005) | +0.090 (0.0003) | **holds** |
| 2. HER2: all omics does not beat copy number | −0.031 (0.27) | −0.090 (0.0005) | −0.015 (0.56) | **holds** |
| 3. Integrative cluster: all omics beats expression | +0.007 (0.69) | +0.057 (0.004) | +0.044 (0.085) | **partly**: random forest only |
| 3. Integrative cluster: copy number beats methylation | +0.269 (9e−11) | +0.268 (2e−12) | — | **holds** |
| 4. ER: all omics does not beat expression | −0.015 (0.16) | −0.006 (0.31) | — | **holds** |
| 4. ER: copy number is the weakest subset | copy number − methylation +0.082 (0.001) | +0.070 (0.003) | — | **fails**: methylation is weakest |

Not predicted: copy number alone is below expression on integrative clusters
(SVM −0.075, p = 0.004; random forest −0.058, p = 0.023). Under the original
stopping rule (best single epoch, no minimum), MOGFormer collapsed to the
majority class on 4 of 25 HER2 folds from copy number; the rule was changed
for that reason, before the integrative-cluster runs, and changes no mean
significantly.
