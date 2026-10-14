# MOGFormer

A multi-omics graph transformer over RNA-seq, copy number and DNA methylation,
used to ask what such a model actually learns about breast cancer.

The model reads every gene as one token built from its three measurements, lets
genes attend to one another along a protein-interaction graph (STRING) and a
signed regulatory graph (CollecTRI), and is trained either on a label or
self-supervised, by reconstructing hidden measurements. The frozen
self-supervised encoder is then used to stratify patients, to test those
strata against relapse, and to read back what drives them.

> **Status: research code, experiments in progress.** Everything runs through
> one tested package, `src/mogformer/`. [ARCHITECTURE.md](ARCHITECTURE.md) holds
> the layout, the coding conventions and a generated registry of every symbol.

---

## Pipeline

```
cleaned matrices ──► cohort bundles ──► self-supervised encoder ──► representation evaluation
 (data/clean)        (data/bundle)      (3 seeds per cohort)         (probes, PCA / MOFA+ baselines)
                                               │
                                               ├──► patient stratification ──► survival analysis
                                               │    (consensus clustering)      (locked protocol)
                                               │
                                               └──► interpretability
                                                    (attribution, omics reliance, gene programs)
```

| Stage | Script | Configuration | Writes to `results/experiments/` |
| --- | --- | --- | --- |
| Cohort bundles | `scripts/build_cohort_bundles.py` | paths in `utils/build_cohort_bundle.py` | `data/bundle/` |
| Self-supervised pretraining and its evaluation | `scripts/pretrain_encoder.py` | `configs/ssl/<cohort>_default_encoder.yaml` | `ssl_pretraining/`, `representation_evaluation/` |
| Stratification and survival | `scripts/stratify_and_analyse_survival.py` | `configs/patient_stratification/` | `patient_stratification_combat/` |
| Interpretability | `scripts/interpret_encoder.py` | `configs/patient_stratification/protocol_combat.yaml` | `patient_stratification_combat/interpretability/` |

Each script is a documented wrapper around one or more experiments of the
package's command line (`uv run python -m mogformer.cli.experiments list`),
which can also be called directly, one experiment at a time.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). PyTorch is taken
from the CUDA 12.8 index; it also runs on CPU, but pretraining is meant for a
GPU (about 1.2 h per TCGA run and 4 h per METABRIC run on one consumer GPU).

```bash
uv sync --extra analysis --extra baselines
```

```bash
uv run pytest
```

The tests use synthetic data only; they need no cohort.

---

## Data

The cohorts are not distributed with the repository. [docs/data.md](docs/data.md)
lists every file the pipeline reads, its source and where it goes:

- TCGA-BRCA and METABRIC expression, copy number, methylation and clinical
  tables (cBioPortal);
- the STRING v12 interaction network and the CollecTRI signed regulatory
  network;
- small reference gene lists (PAM50, the curated gene list).

Once the cleaned matrices are in place:

```bash
uv run python -m scripts.build_cohort_bundles
```

---

## Running the pipeline

Run every command from the repository root.

```bash
uv run python -m scripts.pretrain_encoder --cohort tcga
```

```bash
uv run python -m scripts.pretrain_encoder --cohort metabric
```

```bash
uv run python -m scripts.stratify_and_analyse_survival
```

```bash
uv run python -m scripts.interpret_encoder
```

Before a long run, check the whole chain on a three-epoch smoke test:

```bash
uv run python -m scripts.pretrain_encoder --cohort tcga --seeds 42 --max-epochs 3
```

Every run writes its resolved configuration, a provenance record (inputs,
versions, seed) and its outputs next to each other, so a result can always be
traced to the exact settings that produced it.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `src/mogformer/data/` | Loading, alignment, copy-number scales, per-run preprocessing, batch correction, gene programs |
| `src/mogformer/graph/` | STRING and CollecTRI graphs, shortest paths, positional encodings |
| `src/mogformer/models/` | The transformer, its layers, the self-supervised encoder and decoder |
| `src/mogformer/training/` | Losses, regularisers, the training loops, seeding |
| `src/mogformer/evaluation/` | The shared cross-validation harness, metrics, statistics, model-comparison figures |
| `src/mogformer/analysis/` | Embedding extraction, representation evaluation, clustering, survival, attribution, analysis figures |
| `src/mogformer/experiments/` | One module per experiment, composing the above |
| `src/mogformer/cli/` | Command lines: `mogformer` (supervised) and `mogformer.cli.experiments` |
| `scripts/` | Entry points of the pipeline stages |
| `configs/` | Every experiment's settings, with a README per directory stating its predictions and outcomes |
| `utils/` | Repository tooling: cohort bundles, the symbol registry, the run index |
| `tests/` | Unit and integration tests on synthetic data |
| `docs/` | Data provenance |

Data (`data/`), generated results (`results/`, `figures/`) and private notes
are not tracked.

---

## Reproducibility and leakage controls

- **Seeds.** Every entry point seeds Python, NumPy and PyTorch through one
  function and records the seed; the patient splits have their own seed, shared
  by all encoder seeds, so seeds differ only in initialisation.
- **Outcome-blind encoder.** Pretraining never reads a survival column, and no
  label enters the encoder's inputs. Batch correction uses no biological
  covariates for the same reason.
- **Fitted on the fitting patients only.** Gene selection, imputation, scaling,
  ComBat batch correction and the gene programs are fitted on the training and
  monitoring patients of each run, then applied to every patient. Held-out
  patients are used only to measure reconstruction.
- **Probes inside folds.** Every linear probe fits its scaler and model on the
  training folds of a repeated cross-validation; the supervised harness refits
  its whole preprocessing inside each fold of one shared partition.
- **Locked protocols.** Clustering is defined before it is run
  (`configs/patient_stratification/README.md`) and reads no outcome. The
  survival analysis is locked before any outcome is read
  (`outcome_protocol.md`), runs once, and is preceded by the same code on
  permuted outcomes, where every association must vanish. Anything outside the
  locked protocol is labelled post hoc in its outputs.
- **Unsupervised reductions.** The standardise-then-PCA reduction applied
  before clustering and survival models is fitted on the population's
  embeddings without labels or outcomes; survival models choose their ridge
  penalty by inner cross-validation and are scored out of fold.

---

## Findings so far

Paths under `results/` are written by the experiments named next to them and
are not part of the repository. The predictions behind each finding, written
before the runs, are in the README of the matching `configs/` directory.


PAM50 breast cancer subtypes are assigned by a 50-gene **RNA** centroid
classifier. Training a trimodal model to recover them therefore asks copy number
and methylation to contribute to a label that, by construction, holds no
information they uniquely carry. That framing turned out to be the project's
most useful result, and the work reorganised around it.

The findings below are each backed by a pre-registered protocol or an
explicit negative control.

### 1. On a circular label, fusion buys nothing

Every model is scored on one shared partition (5 repeats of 5-fold CV, 25
paired estimates), with Nadeau–Bengio corrected intervals and tests. The best
transformer ties gradient-boosted trees, and neither gains anything from copy
number or methylation — the expected outcome when the target is an
expression-derived label.

| Model | Five subtypes (949) | Four subtypes (914) |
| --- | --- | --- |
| XGBoost, all three omics | **0.849** [0.808, 0.891] | **0.926** [0.906, 0.947] |
| MOGFormer, pretrained gene vectors | 0.844 [0.800, 0.889] | 0.910 [0.878, 0.942] |
| MOGFormer, gated fusion | 0.833 [0.791, 0.876] | 0.874 [0.841, 0.906] |
| MOGFormer, learned gene identity | 0.816 [0.763, 0.870] | 0.864 [0.843, 0.886] |
| Elastic-net logistic | 0.819 [0.786, 0.853] | 0.901 [0.875, 0.926] |
| PAM50 nearest centroid (expression) | 0.742 [0.716, 0.767] | 0.864 [0.834, 0.895] |

XGBoost minus the best MOGFormer: +0.005 on five subtypes (corrected p = 0.84).
The gated and learned-identity variants read methylation as M-values, the
pretrained one as beta values. Reading it either way changes no variant's
score measurably (all paired differences ≤ 0.02, corrected p ≥ 0.32;
details (`results/experiments/methylation_representation/README.md`)).

| Five subtypes, all three omics minus expression only | Δ macro-F1 | corrected p |
| --- | --- | --- |
| XGBoost | −0.007 | 0.73 |
| Elastic-net logistic | −0.031 | 0.12 |
| MOGFormer, pretrained gene vectors | −0.013 | 0.43 |

Copy number and methylation without expression reach 0.719 (XGBoost), well
above the 0.193 floor but 0.137 below expression alone. Removing the
gene-identity embedding drops the transformer to 0.579: most selected genes
are then indistinguishable to it, because their interaction-graph encodings
nearly coincide. Tables, tests and figures are under
results/experiments/ (`results/experiments/`); the predictions were written
down before these runs in
[configs/modality_subsets/README.md](configs/modality_subsets/README.md).

### 1b. Where the label is not derived from expression, copy number matters

If fusion buys nothing on PAM50 only because PAM50 is called from expression,
copy number should matter for labels defined by copy number. METABRIC carries
such labels. Each was predicted from each omics subset on its own 5×5
partition, with predictions written down before the runs
([configs/metabric/README.md](configs/metabric/README.md)).

| METABRIC target (n) | Derived from | All omics | Expression | Copy number | Methylation |
| --- | --- | --- | --- | --- | --- |
| PAM50 (1,416) | expression | 0.810 | **0.855** | 0.587 | 0.465 |
| ER status (1,392) | protein | 0.906 | **0.912** | 0.825 | 0.743 |
| Integrative cluster (1,416) | copy number + expression | **0.651** | 0.618 | 0.543 | 0.274 |
| HER2 status (1,414) | ERBB2 copy number | 0.607 | 0.597 | **0.676** | 0.414 |

Best classical model per cell (linear SVM or random forest), macro-F1 over 25
estimates. METABRIC expression is distributed as per-gene z-scores and is
standardised as it is (`preprocess.rna_scale: log`). Runs before 2 Oct 2026
applied the count transform, log(1 + x) after clipping at zero, which erased
every below-average value; they are archived under
`results/archive/metabric_clipped_expression/` and are not used. Paired
differences, corrected p:

- **HER2 status: copy number beats expression**, +0.041 (SVM, p = 0.024) and
  +0.108 (random forest, p = 0.0005). Adding the other omics does not help and,
  for the random forest, hurts (−0.090, p = 0.0005).
- **Integrative cluster: all omics beats expression** for the random forest
  (+0.057, p = 0.004), not for the linear SVM (+0.007, p = 0.69). Copy number
  alone is *below* expression (SVM −0.075, p = 0.004; random forest −0.058,
  p = 0.023): the clusters were defined from copy number acting on expression,
  which expression already carries.
- **PAM50 and ER status: all omics does not beat expression**, reproducing
  TCGA. For the linear SVM on PAM50, adding the other omics is worse than
  expression alone (−0.045, p = 0.047).

Which omics helps therefore follows what the label is derived from. Fusion
gains, where they exist, are about 0.06 of macro-F1. Two outcomes went against
the predictions: methylation, not copy number, is the weakest input for ER
status, and the integrative-cluster gain does not hold for the linear model.

MOGFormer with gated fusion was run on the two copy-number-linked targets.
Under the original stopping rule it collapsed on 4 of 25 HER2 folds from copy
number (macro-F1 0.34–0.38, against ≈ 0.69 on the rest): the inner-validation
macro-F1, read at a single epoch, peaked before the rare loss class
(78 patients) was learnt. The final protocol therefore trains at least 100
epochs and selects the epoch on a 5-epoch moving average of inner macro-F1
(`min_epochs`, `selection_window`). The collapsed folds recover (lowest score
0.58), and no mean changes significantly (HER2 +0.035, p = 0.62; integrative
cluster −0.005, p = 0.74;
figure (`results/experiments/report_figures/stopping_rule.png`)). Under the final protocol:

| MOGFormer | All omics | Expression | Copy number |
| --- | --- | --- | --- |
| HER2 status | 0.654 | 0.580 | 0.669 |
| Integrative cluster | 0.627 | 0.583 | 0.526 |

Copy number beats expression for HER2 status (+0.090, p = 0.0003), and all
omics does not beat copy number (−0.015, p = 0.56). On integrative clusters the
all-omics gain over expression is +0.044 (p = 0.085). Against the classical
models on the same input MOGFormer is indistinguishable everywhere except all
omics for HER2 status, where it beats the random forest (+0.068, p = 0.003) but
not the linear SVM (+0.047, p = 0.15).

### 2. The earlier encoder's unsupervised split was stable, redundant, and not prognostic

Outcome-blind self-supervised pretraining yields a near-perfect two-way split of
335 Luminal A patients — which a single CNV-burden score reproduces exactly, and
which does not predict outcome.

| Quantity | Value |
| --- | --- |
| Consensus PAC at K=2 | 0.012 |
| k-means vs GMM agreement | ARI 1.00 |
| CNV-only vs frozen partition | ARI 1.00 |
| RNA-only vs frozen partition | ARI −0.01 |
| PFI hazard ratio (pre-registered) | 1.28 [0.64, 2.55], 35 events |
| Regularised Cox out-of-fold C-index | 0.45 [0.31, 0.58] |

The survival protocol (`results/SSL/SURV_CLEAN/TCGA/luma_survival/protocol/protocol.md`)
was locked before any survival data was touched.

### 3. In-silico perturbation reads back co-expression, not regulation

Intervening on a gene's methylation and reading its reconstructed expression
recovers the silencing direction genome-wide (cis median ρ −0.206, Wilcoxon
p = 1.1e−72). Perturbing a transcription factor is a different matter. The
trans probe was run on the frozen TCGA encoder and, independently, on the
frozen METABRIC encoder, each against the CollecTRI signed edges among its
genes; "targets fully hidden" masks every channel of the read-out genes.

| Quantity | TCGA-BRCA | METABRIC |
| --- | --- | --- |
| edges tested | 903 | 663 |
| median \|slope\|, curated edges / matched non-edges | 1.54×, p = 2e−17 | 1.08×, p = 0.01 |
| effect lost when only the regulatory graph is shuffled | 0% (−8%) | 8% |
| effect lost when interaction and regulatory graphs are shuffled | 35% | not run |
| median \|ρ\|, curated edges / matched non-edges, in the data | 1.23× | 0.96× |
| data sign concordance with CollecTRI, permutation p | 0.629, p = 0.13 | 0.548, p = 0.21 |
| sign concordance with CollecTRI | 0.597 | 0.517 |
| label-permutation null (p) | 0.581 (0.14) | 0.538 (0.91) |
| always answering "activating" | **0.772** | **0.760** |
| model sign = observed co-expression sign | **0.81** | **0.77** |
| observed co-expression sign = CollecTRI sign | 0.63 | 0.55 |

Three things follow.

- **The direction of regulation is not recovered in either cohort.** Neither
  cohort beats the permutation null that preserves the edge set's 3:1
  activating imbalance, and both lose to a constant predictor. Against a coin
  flip the TCGA figure would look decisive (binomial p = 9e−10); that null is
  wrong twice over.
- **The effect sign is the co-expression sign.** On edges with strong observed
  co-expression (|ρ| > 0.2) the model agrees with the data's sign 91% (TCGA) and
  95% (METABRIC) of the time, and matches CollecTRI exactly as often as the raw
  correlation does (0.68 vs 0.69 in TCGA). The probe differentiates a masked
  reconstruction, i.e. a conditional association learned from observational
  data; it has no access to interventional information that could separate a
  regulatory direction from a correlation.
- **The regulatory graph carries almost none of the effect.** Shuffling which
  genes the CollecTRI edges connect leaves the effect intact (108% retained in
  TCGA, 92% in METABRIC); the learned regulatory attention biases are ≤0.3
  logits against a distance bias of up to ±1. What routing exists in TCGA comes
  from the STRING interaction graph (35% lost when both graphs are shuffled).
- **The curated signs are not visible in the data either.** Without any model,
  observed co-expression matches CollecTRI signs no better than the same
  permutation null (TCGA 0.629 vs 0.613; METABRIC 0.548 vs 0.534), and 67% of
  TCGA's repressive edges are *positively* correlated. In METABRIC curated edges
  are not even more co-expressed than non-edges. A learner of conditional
  associations cannot recover a sign the observational data does not carry.

Tables and figures:
TCGA (`results/experiments/sign_concordance/TCGA/sign_tests.md`),
METABRIC (`results/experiments/external_trans_sweep/metabric/sign_tests.md`),
figures (`results/experiments/report_figures/`). The gene-level cis specificity
test does not separate curated silenced genes from controls (p = 0.49), so the
cis claim is a correct *global* direction, not per-gene discrimination.

### 4. Where the signs are true, the same probe recovers them

To separate the estimand from the data, cohorts were simulated from a
structural causal model built on the same 903 CollecTRI edges of the TCGA gene
axis, with every curated sign true by construction. Copy number acts as a
dosage parent of its own gene (with shared amplicon segments), methylation as a
repressor, and five hidden programs confound expression. Their strength (0.8)
was calibrated so that the simulated co-expression matches TCGA's (edge
co-expression sign agreement 0.608 vs 0.629). The encoder was retrained with
the archived TCGA settings, probed exactly as before and scored with the same
test. Every condition was trained for the same number of optimiser steps.

| Sign agreement with the curated edge | probe | co-expression | ridge | CNV instrument | graph share of probe effect |
| --- | --- | --- | --- | --- | --- |
| simulated, 949 patients, graph given | **0.794** | 0.608 | 0.891 | 0.818 | 70% |
| simulated, 949 patients, graph withheld | 0.635 (null 0.629) | 0.608 | 0.891 | 0.818 | — |
| simulated, 3,000 patients, graph given | **0.944** | 0.605 | 0.888 | 0.824 | 13% |
| simulated, 10,000 patients, graph given | **0.936** | 0.602 | 0.901 | 0.801 | 26% |
| TCGA-BRCA, observed (748) | 0.597 | 0.629 | 0.602 | 0.507 | 0% (−8%) |
| METABRIC, observed (1,416) | 0.517 | 0.548 | 0.496 | 0.544 | 8% |

Always answering "activating" scores 0.772 (0.760 in METABRIC); every bold
entry clears both it and the label-permutation null (p < 0.001).

- **The estimand is not the obstacle.** At TCGA's size and confounding, the
  probe recovers curated signs above the constant baseline, and the regulatory
  graph carries most of its effect. From 3,000 patients it beats a ridge
  regression on the same visible inputs.
- **The graph is what makes it work.** Withholding the regulatory bias drops
  the probe to its null (0.635 vs 0.629), where it sits in the real cohorts.
- **The observed cohorts do not express the curated signs.** Model-free
  estimators that recover 80–90% of signs in the simulation all fall short of
  the constant baseline in both real cohorts. The best of them, ridge in TCGA
  (0.602), beats its permutation null (0.546, p < 0.001) but is 17 points below
  always answering "activating". That includes a copy-number instrument, which
  is robust to hidden confounding. Co-expression stays near 0.60 at any
  sample size, so more patients alone would not change this.
- Without hidden programs (confounding 0.0), the reconstruction objective
  stalls near the per-gene mean (monitoring loss 0.81 vs 0.44) and early stopping
  ends training there. Those two encoders learned nothing to probe and are
  reported in the tables but not interpreted.

Tables and figure:
summary (`results/experiments/simulation_benchmark/summary.csv`),
figure (`results/experiments/report_figures/simulation_sign_recovery.png`).

### 5. The self-supervised encoder organises known biology, not new prognostic strata

The current encoder reads each gene program (16 groups of co-varying genes)
from the genes' own measurements into its own block of the embedding. It uses
11.4 (TCGA) and 15.3 (METABRIC) effective dimensions, against 7–9 for the
earlier summary-token encoder, with better probes and reconstruction
(`configs/embedding_spread/README.md`). METABRIC expression and methylation are
ComBat-corrected for processing batch (`configs/batch_correction/README.md`):
uncorrected, the encoder's most stable cluster was a processing batch.

| METABRIC ER+/HER2− (887 patients, 335 relapses) | Result |
| --- | --- |
| Pre-registered stratification bar (PAC ≤ 0.10, bootstrap ARI ≥ 0.80, seed ARI ≥ 0.70) | not met after batch correction (K = 3: PAC 0.148, seed ARI 0.56) |
| High-risk stratum (LumB, IC1/IC9/IC6/IC10, 8q gain, 17p loss) vs the largest, post hoc | relapse HR 1.78 [1.41, 2.25] unadjusted; 1.03 [0.76, 1.39] after proliferation |
| Strata over the clinical model, post hoc | Uno ΔC 0.001 [−0.039, 0.029] |
| Embedding alone, out-of-fold relapse C-index | 0.608, against PCA 0.618 and MOFA+ 0.610 |

Attribution of the relapse risk score to the inputs (integrated gradients)
puts proliferation genes first (GMPS, CCNE2, MCM4, CDC25C); the high-risk
stratum is defined by an 8q copy-number gain with 8p and 17p loss and high
proliferation, the profile of integrative clusters 1 and 9. Details:
`configs/patient_stratification/README.md`.

---

## Development

```bash
uv run ruff check src utils tests scripts
```

```bash
uv run mypy
```

```bash
uv run python -m utils.build_registry --check
```

```bash
uv run pre-commit install
```

The registry check is the repository's guard against duplicate
implementations: it fails when the symbol registry in `ARCHITECTURE.md` is
stale, or when any name is defined in more than one module. Code follows
Google-style docstrings, full type annotations, snake_case names and the
conventions in [ARCHITECTURE.md](ARCHITECTURE.md), including the figure style.

The supervised pipeline runs through the `mogformer` command, for example:

```bash
uv run mogformer folds --config configs/tcga.yaml
```

```bash
uv run mogformer train --config configs/tcga.yaml --models B5_xgboost B1_pam50_centroid
```

---

## License

MIT — see [LICENSE](LICENSE).
