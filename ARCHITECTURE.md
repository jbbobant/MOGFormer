# MOGFormer — architecture and conventions

This document is the contract for the codebase. It states where code lives, how
it is written, and what every symbol is for. The symbol registry at the end is
generated from the source, so it cannot drift.

Read the [Ground rules](#ground-rules) before adding a module, and search the
[registry](#symbol-registry) before adding a function.

---

## Why this document exists

The project was developed as Kaggle notebooks and then partially modularised
twice, in parallel, without a shared definition of any concept. That produced
two independent implementations of the same data loader — and the consequence
was not untidy code, it was a scientific defect:

```python
# MOGFORMER BL/harness/data.py, inside load_omics
filtered_clin = filtered_clin[filtered_clin[label_col] != "BRCA_Normal"]
```

That hardcoded line dropped the 35 Normal-like patients inside the loader. The
baseline harness therefore built its folds over 914 patients while the MOGFormer
runs built theirs over 949. `make_folds` is byte-identical between the two
codebases; only the cohort feeding it differed. The result is that baselines and
MOGFormer were never scored on the same partitions, so the paired Wilcoxon test
and critical-difference diagram the evaluation protocol promises cannot be
computed from the stored results.

One canonical implementation per concept is therefore a correctness requirement
in this repository, not a matter of taste. The registry and its duplicate check
exist to enforce it.

### The two trees are branches, not versions

`src/` is not an older copy of `Recent_Kaggle.py`. They are two lineages forked
from a common ancestor:

* **`src/` — the supervised lineage.** Holds `MultiOmicsGraphClassifier`,
  `FoldTrainer`, `MultiClassFocalLoss`, `PLRModality`, `GeneEmbeddingAdapter`
  and the metrics and plotting suite. Nineteen symbols here exist nowhere else.
* **`Recent_Kaggle.py` — the self-supervised lineage.** Holds `MOGFormerSSL`,
  `DualHeadDecoder`, `MMMMasker`, `GatedFusion`, the value-level mask tokens and
  both interventional probes.

Where they overlap they have drifted: `ModalityLifting` is 31% textually
similar between them and `StructuralGraphAttention` 29%. Migration therefore
means **unifying** those layers, not picking a winner. The union is recorded in
[Contested symbols](#contested-symbols).

`src/` also cannot be imported at all — `src/models/layers/modality_lifting.py`
uses `Optional`, `Sequence` and `math` without importing any of them. It only
ever worked as pasted notebook cells, which is worth knowing before treating any
of it as a reference implementation.

### A superseded statistic the migration surfaced

Migrating the trans probe turned up a second defect, this one in the analysis
rather than the code. Sign concordance between the model's trans effects and
CollecTRI edge directions was reported as 60.0% with a binomial p of 9.3e-10 —
against a null of 0.5.

That null is wrong. The edge set is 77% activating and the model's slopes carry
their own sign bias; a label-permutation null preserving both sits at 0.586 and
gives p = 0.16. Worse, always predicting "activating" scores 0.772, so the model
loses to a constant predictor.

`sign_permutation_test` in `analysis/probe_trans.py` is the corrected test, and
`SignConcordanceResult.is_informative` requires a result to clear *both* the
permutation null and the trivial baseline. A test pins the failing case so the
superseded statistic cannot come back.

### A defect the unification fixed

The two lineages disagreed about the tumor summary token's distance bucket. The
self-supervised branch pads the summary row and column of the shortest-path
matrix with `max_distance + 2`, a bucket of its own, and sizes the bias table
`max_distance + 3` to match. The supervised branch padded with `0`, which places
the summary token in the same bias bucket as every gene's own diagonal —
conflating "this is me" with "this is the whole-tumor summary".

It does not crash, so it silently affected every supervised classifier run. The
unified layer takes the self-supervised treatment, which means **new classifier
runs are not bit-comparable with the archived ones**. Whether it explains any of
the gap to the classical baselines is untested and should not be asserted; it is
a reason to re-run, not a conclusion.

---

## Ground rules

1. **One concept, one implementation.** Before writing a function, search the
   [registry](#symbol-registry). If something close exists, extend it or call
   it. Two functions with the same name in different modules is a build failure.
2. **Configuration is data, never a literal.** Cohort filters, file names, label
   columns, thresholds and paths arrive through a config object. The bug above
   is what a hardcoded filter costs.
3. **One harness.** Every model — classical baseline or MOGFormer — is scored
   through `mogformer.evaluation` against one `folds.json`, one metrics module
   and one stats module. Nothing computes its own metrics.
4. **Fit on train, apply to test.** Gene selection, imputation, scaling and
   graph construction are refit inside each fold. No exceptions, including for
   the graph.
5. **Determinism is a feature.** Every entry point seeds explicitly and writes a
   provenance record naming inputs, config and versions.
6. **The legacy trees are gone.** `Recent_Kaggle.py`, `MOGFORMER BL/harness/`,
   the old `src/`, `P9.py`, `Probe.py` and `save/` were folded into
   `src/mogformer/` and then deleted. What remains under `scripts/` is
   unmigrated data acquisition and figure code; fold it in rather than
   extending it in place.

---

## Coding conventions

| Concern | Rule |
|---|---|
| Classes | `PascalCase` noun phrases: `SquareGridMesh`, `FuelProperties` |
| Functions | `snake_case` verb phrases: `compute_rate_of_spread_balbi_2009`, `build_simulation_context` |
| Variables | `snake_case`, physical quantities carry unit suffix: `wind_speed_m_per_s`, `cell_spacing_m` |
| Arrays | Carry shape meaning: `source_positions_xyz`, `gain_matrix_source_by_receiver` |
| Booleans | Read as assertions: `is_burning`, `has_ignited`, `uses_diagonal_neighbors` |
| Constants | `UPPER_SNAKE_CASE` with unit suffix: `STEFAN_BOLTZMANN_CONSTANT_W_PER_M2_K4` |
| Abbreviations | Never. `rate_of_spread`, not `ros`. `digital_elevation_model`, not `dem` |
| Paper notation | Allowed **only** inside the body of a function whose name states which equation it implements |
| Comments | None |
| Docstrings | Google style: one-line summary, then `Args:` / `Returns:` / `Raises:`. Concise — the signature carries the types |
| Type annotations | Mandatory on every function signature and dataclass field |
| Array operations | Vectorized numpy. No Python loops over cells |
| File scope | One primary class or one cohesive group of pure functions per file |
| Protocol files | Contain only the Protocol definition, no implementation |

## Layout

```
src/mogformer/
  config.py              Typed configuration objects, loaded from YAML
  data/
    omics.py             Raw matrix loading, patient alignment, label encoding
    preprocess.py        MultiOmicsTransformer — per-fold selection and scaling
    folds.py             Fold construction, persistence, leakage assertions
  graph/
    string_graph.py      STRING PPI edges to adjacency over selected genes
    grn.py               CollecTRI signed regulatory edges
    spd.py               Shortest-path distance matrices
    positional_encoding.py  Random-walk and Laplacian positional encodings
  models/
    classifier.py        MultiOmicsGraphClassifier — supervised subtyping
    ssl.py               MOGFormerSSL — masked multi-modal pretraining
    layers/
      modality_lifting.py     Scalar to d-dimensional token projection
      mini_transformer.py     Intra-gene cross-modality attention
      gated_fusion.py         Gated alternative to the mini transformer
      structural_attention.py SPD-biased attention block
      global_transformer.py   Inter-gene transformer over the gene sequence
      decoder.py              Dual-head reconstruction decoder
      masking.py              Masked multi-modal masker
  training/
    losses.py            Focal loss, masked Huber, participation ratio
    trainer.py           MaskedReconstructionTrainer — the pretraining loop
    seed.py              Global seeding
  evaluation/
    registry.py          Model registry: baselines and MOGFormer
    estimator.py         Scikit-learn wrapper exposing MOGFormer to the harness
    metrics.py           Metric suite, naive and Nadeau–Bengio intervals
    stats.py             Paired Wilcoxon, corrected t, Friedman–Nemenyi
    runner.py            Cross-validation orchestration
    plots.py             Evaluation figures
  analysis/
    embedding.py         Outcome-blind pretraining and embedding extraction
    clustering.py        Consensus clustering, PAC, confound screen, ablation ladder
    survival.py          Cox adjustment ladder, PH check, RMST
    probe_cis.py         Interventional cis probe (methylation to expression)
    probe_trans.py       Trans propagation and the corrected sign test
    plots.py             Analysis figures
  cli/                   One entry point per phase
tests/
  unit/                  Fast, no data dependency
  integration/           Synthetic end-to-end runs
utils/
  build_registry.py      Regenerates the registry in this file
scripts/
  GRN download.py        CollecTRI acquisition (not yet folded in)
  meth_prepro.ipynb      Methylation preprocessing (not yet folded in)
  figures/               Cohort and network figures (not yet folded in)
results/
  baselines/             Classical harness outputs, kept as evidence
```

The package sits under `src/` so that tests import the installed distribution
rather than the working directory — a layout that catches a missing
`__init__.py` or an unpackaged module before publication rather than after.

### Where the legacy code went

| Legacy location | Destination |
| --- | --- |
| `Recent_Kaggle.py` §`src.models.layers.*` | `src/mogformer/models/layers/` |
| `Recent_Kaggle.py` §`src.graph.*` | `src/mogformer/graph/` |
| `Recent_Kaggle.py` §`src.evaluation.fold_preprocess` | `src/mogformer/data/preprocess.py` |
| `Recent_Kaggle.py` §`src.evaluation.fold_data` | `src/mogformer/data/omics.py` |
| `Recent_Kaggle.py` §`src.evaluation.cv` | `src/mogformer/data/folds.py` |
| `Recent_Kaggle.py` §`src.models.ssl`, §`DHDecoder`, §`MMMMasker` | `src/mogformer/models/` |
| `Recent_Kaggle.py` §`EMBEDDING EXTRACTION` | `src/mogformer/analysis/embedding.py` |
| `Recent_Kaggle.py` §`EMBEDDING ANALYSIS AND CLUSTERING` | `src/mogformer/analysis/clustering.py` |
| `Recent_Kaggle.py` Phase 8 probe | `src/mogformer/analysis/probe_cis.py` |
| `Recent_Kaggle.py` Phase 9 cis/trans | `src/mogformer/analysis/probe_cis.py`, `probe_trans.py` |
| `Recent_Kaggle.py` MMM trainer | `src/mogformer/training/trainer.py` |
| `MOGFORMER BL/harness/{metrics,stats,models,plots,run_phase0}.py` | `src/mogformer/evaluation/` |
| `MOGFORMER BL/harness/{data,preprocess,cv}.py` | `src/mogformer/data/` |
| `MOGFORMER BL/test_*.py` | `tests/unit/` (four acceptance claims, all covered) |
| `MOGFORMER BL/results/` | `results/baselines/` (evidence, not code) |
| `src/models/classifier.py`, `utils/loss.py`, `utils/seed.py` | `src/mogformer/models/`, `training/` |

Where a symbol exists in both trees, the reconciliation is recorded in
[Contested symbols](#contested-symbols).

---


### Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Module | `lower_snake_case` | `structural_attention.py` |
| Package | `lower_snake_case`, singular | `mogformer/graph/` |
| Class | `CapWords` | `MultiOmicsTransformer` |
| Function, method, variable | `lower_snake_case` | `compute_shortest_paths` |
| Constant | `UPPER_SNAKE_CASE` at module level | `MODALITY_ORDER` |
| Internal symbol | single leading underscore | `_load_omics_raw` |
| Type variable | `CapWords` with `T` suffix | `ArrayT` |

No abbreviations that are not already domain terms. `rna`, `cnv`, `methy`,
`spd`, `pe`, `grn`, `hvg` and `mad` are domain terms and are allowed bare.
`Structural_attention_block.py` and `DHDecoder.py` are the existing violations;
both are renamed on migration.

### Docstrings

Every module, public class and public function carries a Google-style
docstring. Internal helpers carry at least a one-line summary.

```python
def compute_shortest_path_matrix(
    adjacency: torch.Tensor,
    max_distance: int = 5,
) -> torch.Tensor:
    """Compute the truncated all-pairs shortest-path matrix of a gene graph.

    Runs Floyd-Warshall over the binarised adjacency. Distances greater than
    ``max_distance`` -- including pairs in different connected components -- are
    collapsed onto the single bucket ``max_distance + 1``.

    Args:
        adjacency: Square adjacency matrix, shape ``(n_genes, n_genes)``. Any
            strictly positive entry is treated as an edge; weights are ignored.
        max_distance: Largest hop count represented exactly. Must be positive.

    Returns:
        Integer distance matrix of shape ``(n_genes, n_genes)`` on the same
        device as ``adjacency``, with values in ``[0, max_distance + 1]`` and a
        zero diagonal.

    Raises:
        ValueError: If ``adjacency`` is not a square 2-D tensor, or if
            ``max_distance`` is not positive.
    """
```

That is the real docstring of `mogformer.graph.spd.compute_shortest_path_matrix`.
The first line is an imperative sentence ending in a period. Shape and units
belong in the docstring: `(n_patients, n_genes)`, `z-units`, `months`.

### Typing

Public functions are fully annotated. `from __future__ import annotations` at
the top of every module. Prefer `np.ndarray` and `torch.Tensor` with the shape
documented rather than inventing shape-typed aliases.

### Imports

Absolute, never implicit-relative. The legacy `from preprocess import ...` in
`MOGFORMER BL/harness/models.py` only resolves by working-directory accident and
is the reason that tree cannot be imported as a package.

```python
from mogformer.data.preprocess import MultiOmicsTransformer   # correct
from preprocess import MultiOmicsTransformer                  # forbidden
```

Order: standard library, third party, first party — separated by blank lines and
alphabetised within each group.

### Errors and logging

Raise a specific exception with a message naming the offending value. Never
`assert` for anything a user can trigger; `assert` is for invariants only.
Use the `logging` module, never bare `print`, outside `cli/`.

### Determinism

Anything stochastic takes an explicit `seed` argument or a `numpy.random.Generator`.
No module-level `np.random.seed` calls.

---

## Testing

| Layer | Location | Must prove |
| --- | --- | --- |
| Preprocessing | `tests/unit/test_preprocess.py` | Statistics are fit on train only; curated genes force-included; no test row influences any fitted parameter |
| Folds | `tests/unit/test_folds.py` | No patient in both train and test; folds round-trip through JSON unchanged; the same seed and cohort reproduce the same partition |
| Graph | `tests/unit/test_graph.py` | SPD clamping and the unreachable bucket; positional encodings are permutation-equivariant |
| Models | `tests/unit/test_models.py` | Every registry entry builds, fits and returns well-formed probabilities on a shared synthetic fold |
| Metrics | `tests/unit/test_metrics.py` | Metric values against hand-computed cases; Nadeau–Bengio interval wider than the naive one |
| End to end | `tests/integration/test_pipeline.py` | A synthetic cohort runs through the harness and emits every declared artifact |

Tests use synthetic data and never require the real cohort. Target: every public
function in `mogformer/` reachable from at least one test.

Run:

```bash
pytest -q
```

---

## Tooling

```bash
ruff check src utils tests      # lint
ruff format src utils tests     # format
mypy src                              # types
pytest -q                             # tests
python -m utils.build_registry        # refresh the registry below
python -m utils.build_registry --check  # CI: fail on drift or duplicate names
```

`--check` is the gate. It fails when the registry is stale and when any
first-party name is defined in more than one module.

---

## Contested symbols

Symbols implemented in both legacy trees, with the reconciliation decision.
Verified by diffing the extracted source of each pair.

| Symbol | Status | Decision |
| --- | --- | --- |
| `FoldSpec` | identical | Take either. Lands in `data/folds.py`. |
| `make_folds` | identical | Take either. The cohort, not the function, caused the divergence. |
| `save_folds` | identical | Take either. |
| `load_folds` | identical | Take either. |
| `OmicsData` | identical | Take either. Lands in `data/omics.py`. |
| `_load_omics_raw` | identical | Take either. |
| `load_curated_genes` | identical | Take either. |
| `MultiOmicsTransformer` | differs by one comment | Take the harness version; it keeps the note explaining why `__init__` must store parameters verbatim for `sklearn.clone`. |
| `load_omics` | **diverged** | Merge. Harness reads raw TCGA files with `label_col="SUBTYPE"` and hardcodes the Normal-like drop; the notebook reads the cleaned `*_common.csv` files with `label_col="subtype"` and no drop. The merged loader takes file names, label column and an explicit `exclude_classes` argument from config, defaulting to excluding nothing. |
| `_save` | differs | Figure-saving helper, duplicated four times in the notebook. One implementation in `evaluation/plots.py`, re-exported for `analysis/plots.py`. |

### `src/` against `Recent_Kaggle.py`

| Symbol | Similarity | Decision |
| --- | --- | --- |
| `ModalityLifting` | 31% | **Unified.** Union of both: the supervised branch's periodic tokenizer, pretrained-embedding adapter and identity table, plus the self-supervised branch's value-level mask tokens and shared-address accessors. Lives in `models/layers/modality_lifting.py`. |
| `StructuralGraphAttention` | 29% | **Unified.** The self-supervised version is a superset — it adds the signed regulatory bias and the `structural_bias` ablation switch — so it is taken wholesale, with the bias table sized `max_distance + 3`. |
| `GlobalGraphTransformer` | 89% | **Unified**, taking the self-supervised summary-token padding. See the defect above. |
| `MiniTransformer` | 96% | **Unified.** The self-supervised version adds `eval_mask`; otherwise identical. |
| `MOGFormerConfig` | 42% | **Replaced** by the typed sections in `mogformer/config.py`. |
| `select_folds` | 76% | **Replaced** by one implementation in `data/folds.py`. |
| `apply_class_exclusion` / `filter_omics_classes` | near-duplicates | **Split.** `omics.exclude_classes` returns the filtered cohort and a keep mask; `folds.remap_fold_indices` rewrites indices. Composing them preserves fold membership, which is what makes a reduced-class run paired. |
| `MODALITIES` / `MODALITY_ORDER` | same constant, two names | **One name**, `MODALITY_ORDER`, defined in `data/omics.py`. |
| Modality-dropout block | copied verbatim into both fusion layers | **Extracted** to `models/layers/modality_dropout.py`. |

### Shadowed definitions in `Recent_Kaggle.py`

Concatenating three standalone scripts into one file left nine names defined
more than once. Python keeps only the last definition, so the earlier ones are
dead code that never runs. Each must be assigned to its owning module during the
split rather than resolved by whichever copy happened to win.

| Name | Definitions | Lines |
| --- | --- | --- |
| `_save` | 4 | 1252, 3429, 4190, 4778 |
| `build_frozen_model` | 3 | 3176, 3783, 4622 |
| `_load_all` | 3 | 3287, 4390, 4854 |
| `_style` | 3 | 3424, 4183, 4771 |
| `main` | 3 | 3570, 4420, 4875 |
| `seed_everything` | 2 | 2139, 2569 |
| `_write_readme` | 2 | 3702, 4537 |
| `make_gene_batches` | 2 | 3869, 4651 |
| `parity_check` | 2 | 3965, 4739 |

---

## Symbol registry

Search this before writing a function. Regenerate with
`python -m utils.build_registry`.

<!-- BEGIN GENERATED REGISTRY -->

> Generated by `python -m utils.build_registry`. Do not edit by hand.

### Package symbols

_260 symbols across 35 modules._

#### `mogformer.analysis.clustering`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `ConsensusResult` | class | 47–67 | The outcome of consensus clustering at one number of clusters. |
| `ConsensusResult.core_mask(self)` | method | 65–67 | Return True for patients whose cluster membership is unambiguous. |
| `as_array(embedding)` | function | 70–84 | Coerce a torch tensor or array-like into a float array. |
| `top_principal_components(values, n_components)` | function | 87–103 | Project onto the leading principal components. |
| `consensus_matrix(values, n_clusters, n_resample, subsample_frac, seed)` | function | 106–155 | Build the co-assignment matrix over resampled clusterings. |
| `proportion_ambiguous_clustering(consensus, bounds)` | function | 158–175 | Summarise a consensus matrix as its fraction of ambiguous pairs. |
| `consensus_index(consensus, labels)` | function | 178–197 | Compute each patient's mean co-assignment with its own cluster. |
| `run_consensus_clustering(embedding, n_clusters_list, n_resample, subsample_frac, seed)` | function | 200–242 | Cluster by consensus across a range of cluster counts. |
| `order_by_linkage(consensus)` | function | 245–259 | Order patients so a consensus heatmap shows its block structure. |
| `silhouette_full_versus_pcs(embedding, n_clusters_list, n_components, seed)` | function | 262–308 | Compare silhouette in the full space against a leading-component space. |
| `per_cluster_silhouette(embedding, labels)` | function | 311–321 | Return each patient's silhouette against its assigned cluster. |
| `zca_whiten(embedding, ridge)` | function | 324–344 | Remove the dominant scale from an embedding, keeping its axes. |
| `covariate_association(scores, covariate)` | function | 347–394 | Measure how strongly one covariate tracks a continuous score. |
| `screen_covariates(scores, covariates, roles)` | function | 397–435 | Test every covariate against a score, with a multiplicity correction. |
| `_holm_correct(p_values)` | function | 438–459 | Apply the Holm step-down correction, ignoring missing values. |
| `RepresentationRung` | class | 463–476 | One rung of the representation ablation ladder. |
| `compare_representations(representations, reference_labels, n_clusters, n_resample, subsample_frac, seed)` | function | 479–534 | Ask whether a partition needs the model that produced it. |
| `partition_agreement(partitions)` | function | 537–554 | Compute pairwise agreement between partitions of the same patients. |

#### `mogformer.analysis.embedding`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `PretrainingBundle` | class | 54–77 | Everything a downstream phase needs to reuse a frozen encoder. |
| `split_train_monitor(data, monitor_frac, seed, stratify_by_subtype)` | function | 80–116 | Split patients into a training and a monitoring slice. |
| `identify_subtype_patients(data, label, expected_range)` | function | 119–159 | Return the indices of patients carrying one subtype label. |
| `extract_embeddings(model, loader, graph_pe, spd, grn, device)` | function | 163–194 | Read the patient summary token for every patient in a loader. |
| `build_frozen_axes(config, data, device)` | function | 197–261 | Fit the study-wide preprocessing and build its graph tensors. |
| `run_pretraining(config, data, monitor_frac, device)` | function | 264–390 | Pretrain the encoder outcome-blind and extract the patient embeddings. |
| `save_pretraining_bundle(bundle, model, preprocessor, config, data)` | function | 393–466 | Write everything needed to reproduce or reuse the frozen encoder. |
| `_normalisation_stats(preprocessor)` | function | 469–492 | Extract per-modality scaling and imputation statistics. |
| `_write_table(frame, path)` | function | 495–511 | Write a frame as parquet, falling back to CSV when unavailable. |
| `load_pretraining_bundle(root)` | function | 514–556 | Read a saved bundle back for a downstream phase. |
| `subtype_slice(bundle, indices)` | function | 559–576 | Take one subtype's rows out of an extracted embedding. |

#### `mogformer.analysis.plots`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `plot_consensus_heatmap(result, directory, name)` | function | 45–78 | Draw a consensus matrix with patients ordered by linkage. |
| `plot_stability_by_k(results, directory, name)` | function | 81–125 | Draw stability against the number of clusters. |
| `plot_representation_ladder(rungs, directory, name)` | function | 128–177 | Draw each representation's stability and its agreement with the reference. |
| `plot_confound_screen(screen, directory, name)` | function | 180–231 | Draw covariate effect sizes, coloured by role. |
| `plot_response_curves(curves, genes, directory, name)` | function | 234–281 | Draw predicted expression against injected value for selected genes. |
| `plot_sign_concordance(result, directory, name)` | function | 284–349 | Draw observed sign concordance against both bars it must clear. |
| `plot_trans_versus_coexpression(edges, directory, name)` | function | 352–393 | Draw learned trans slope against observed co-expression. |
| `plot_kaplan_meier(frame, duration_col, event_col, group_col, directory, name, p_value)` | function | 396–471 | Draw survival curves per group with a risk table beneath. |
| `plot_hazard_forest(ladder, directory, name)` | function | 474–526 | Draw the adjustment ladder as a forest plot. |

#### `mogformer.analysis.probe_cis`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `make_gene_batches(spd, gene_indices, batch_size, min_distance, grn)` | function | 53–94 | Group genes that can be probed together without interfering. |
| `probe_genes(model, inputs, gene_indices, inject_modality, graph_pe, spd, grn, device, grid, patient_chunk, cnv_diploid)` | function | 98–189 | Sweep an injected value and record the model's predicted expression. |
| `GridRankCorrelation` | class | 192–226 | Spearman correlation between the injected grid and the response. |
| `GridRankCorrelation.__init__(self, n_patients, grid)` | method | 203–212 | Precompute the constant grid ranks. |
| `GridRankCorrelation.__call__(self, response)` | method | 214–226 | Correlate one flattened response against the grid. |
| `patient_slopes(response, grid)` | function | 229–245 | Compute each patient's least-squares slope of response against the grid. |
| `ResponseStatistics` | class | 249–277 | Summary of one gene's response curve. |
| `ResponseStatistics.interval_excludes_zero(self)` | method | 275–277 | Return True when the slope interval lies wholly off zero. |
| `summarise_response(response, correlator, rng, grid, n_boot, delta_threshold)` | function | 280–331 | Summarise a response curve with bootstrap intervals over patients. |
| `observed_correlation(inputs, gene, inject_modality)` | function | 334–357 | Correlate a gene's observed modality against its observed expression. |
| `run_cis_map(model, inputs, gene_order, graph_pe, spd, grn, device, inject_modality, arrow, curated_genes, cnv_diploid, grid, batch_size, n_boot, seed)` | function | 360–467 | Probe every gene and summarise the learned coupling genome-wide. |
| `parity_check(model, inputs, gene_order, graph_pe, spd, grn, device, inject_modality, batch_size, n_genes, grid, seed)` | function | 470–574 | Quantify what batching costs, on a random sample of genes. |

#### `mogformer.analysis.probe_trans`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `SignConcordanceResult` | class | 57–94 | Outcome of the corrected sign-concordance test. |
| `SignConcordanceResult.is_informative(self)` | method | 87–94 | Return True only if the result clears both bars. |
| `probe_transcription_factor(model, inputs, factor, read_genes, graph_pe, spd, grn, device, inject_modality, mask_mode, grid, patient_chunk)` | function | 98–187 | Perturb one transcription factor and read the response at many genes. |
| `slope_with_interval(response, rng, grid, n_boot)` | function | 190–222 | Summarise a response by its slope and a bootstrap interval. |
| `observed_coexpression(inputs, factor, target)` | function | 225–245 | Correlate two genes' observed expression. |
| `run_trans_sweep(model, inputs, gene_order, grn, graph_pe, spd, device, mask_mode, null_label, inject_modality, n_controls_per_factor, grid, n_boot, seed)` | function | 248–354 | Probe every regulatory edge, alongside matched non-edge controls. |
| `_count_concordant(edge_sign, slope)` | function | 357–361 | Count edges whose slope sign matches the network's sign. |
| `sign_permutation_test(edges, n_perm, seed)` | function | 364–442 | Test sign concordance against a null that preserves both biases. |
| `magnitude_tests(sweep, mask_mode)` | function | 445–495 | Test whether effects are larger on real edges than on controls. |
| `permuted_graph_retention(real, permuted)` | function | 498–531 | Measure how much of the trans effect survives shuffling the graph. |

#### `mogformer.analysis.survival`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `CoxResult` | class | 41–70 | One rung of the adjustment ladder. |
| `CoxResult.interval_includes_null(self)` | method | 68–70 | Return True when the interval spans a hazard ratio of one. |
| `months_from_days(days)` | function | 73–82 | Convert follow-up times from days to months. |
| `check_events_per_variable(n_events, n_covariates)` | function | 85–107 | Warn when a model has more covariates than its event count supports. |
| `fit_cox(frame, duration_col, event_col, predictor, covariates, model_label, estimand)` | function | 110–172 | Fit one Cox proportional-hazards rung. |
| `adjustment_ladder(frame, duration_col, event_col, predictor, age_col, stage_col)` | function | 175–250 | Fit the pre-registered ladder of adjustments in order. |
| `proportional_hazards_check(frame, duration_col, event_col, predictor, covariates)` | function | 253–291 | Test the proportional-hazards assumption via scaled Schoenfeld residuals. |
| `restricted_mean_survival_difference(frame, duration_col, event_col, group_col, horizons, n_boot, seed)` | function | 294–375 | Compare restricted mean survival between two groups at fixed horizons. |
| `describe_followup(frame, duration_col, event_col, group_col)` | function | 378–410 | Summarise follow-up and event counts before any model is fitted. |
| `logrank_p_value(frame, duration_col, event_col, group_col)` | function | 413–451 | Return the log-rank p-value comparing two groups. |

#### `mogformer.cli.__main__`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `_configure_logging(verbose)` | function | 25–36 | Send logs to stderr at the requested level. |
| `_load(config_path)` | function | 39–50 | Load a configuration and seed the run from it. |
| `command_folds(args)` | function | 53–91 | Build and persist the shared cross-validation partition. |
| `command_train(args)` | function | 94–108 | Train and score the transformer across the shared folds. |
| `command_pretrain(args)` | function | 111–131 | Run outcome-blind self-supervised pretraining. |
| `build_parser()` | function | 134–167 | Construct the top-level argument parser. |
| `main(argv)` | function | 170–186 | Entry point for the ``mogformer`` command. |

#### `mogformer.config`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `DataConfig` | class | 30–53 | Where the cohort lives and which patients it contains. |
| `FoldConfig` | class | 57–77 | The shared cross-validation partition. |
| `PreprocessConfig` | class | 81–94 | Per-fold feature selection and scaling. |
| `GraphConfig` | class | 98–120 | The biological priors and their derived tensors. |
| `ModelConfig` | class | 124–169 | Architecture of the transformer. |
| `TrainConfig` | class | 173–199 | Optimisation and early stopping. |
| `ExperimentConfig` | class | 203–229 | A complete run: cohort, folds, preprocessing, priors, model, training. |
| `ExperimentConfig.output_dir(self)` | method | 227–229 | Return the directory this run writes into. |
| `_build_section(cls, values, path)` | function | 243–274 | Construct one configuration section from a mapping. |
| `load_config(path)` | function | 277–306 | Read an experiment configuration from a YAML file. |
| `save_config(config, path)` | function | 309–322 | Write a resolved configuration beside its results. |

#### `mogformer.data.folds`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `FoldSpec` | class | 28–41 | One outer cross-validation fold. |
| `make_folds(y, patient_ids, n_splits, n_repeats, seed)` | function | 44–85 | Build repeated stratified patient-level folds. |
| `assert_no_leakage(folds, patient_ids)` | function | 88–113 | Verify that no patient appears in both halves of any fold. |
| `save_folds(folds, patient_ids, path)` | function | 116–145 | Write folds to JSON, recording both indices and patient barcodes. |
| `load_folds(path)` | function | 148–167 | Read folds previously written by :func:`save_folds`. |
| `assert_folds_match(loaded_patient_order, current_patient_ids)` | function | 170–200 | Verify that loaded folds were built over the current cohort. |
| `remap_fold_indices(folds, keep_mask)` | function | 203–241 | Rewrite fold indices after patients have been dropped from a cohort. |
| `select_folds(folds, n_repeats, max_folds)` | function | 244–270 | Take a prefix of the fold list, for cheap partial runs. |

#### `mogformer.data.omics`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `OmicsData` | class | 34–90 | Aligned, raw multi-omics matrices with encoded labels. |
| `OmicsData.n_genes(self)` | method | 56–58 | Return the number of genes per modality block. |
| `OmicsData.n_patients(self)` | method | 61–63 | Return the number of patients. |
| `OmicsData.block_slice(self, modality)` | method | 65–83 | Return the column slice of ``X`` holding one modality. |
| `OmicsData.class_counts(self)` | method | 85–90 | Return the number of patients per class, keyed by class name. |
| `_load_omics_raw(filepath)` | function | 93–115 | Read one genes-by-patients matrix, collapsing duplicate gene rows. |
| `load_omics(raw_dir, rna_file, cnv_file, methy_file, clin_file, label_col, exclude)` | function | 118–228 | Load and align raw multi-omics matrices with clinical labels. |
| `load_curated_genes(path, gene_universe)` | function | 231–269 | Read a curated gene list and keep the genes present in the universe. |
| `exclude_classes(data, classes)` | function | 272–335 | Drop whole classes from a loaded dataset and re-encode labels. |

#### `mogformer.data.preprocess`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MultiOmicsTransformer` | class | 30–262 | Select variable genes, impute, log RNA and standardise, all per fold. |
| `MultiOmicsTransformer.__init__(self, n_genes, gene_names, top_k, curated_genes, active_modalities, mad_on_log_rna, impute)` | method | 47–83 | Store hyperparameters verbatim. |
| `MultiOmicsTransformer._active(self)` | method | 86–88 | Return the active modalities as a tuple, without touching the param. |
| `MultiOmicsTransformer._curated(self)` | method | 91–93 | Return the curated gene list, without touching the stored param. |
| `MultiOmicsTransformer._block(self, X, modality)` | method | 95–114 | Return the columns of ``X`` belonging to one modality. |
| `MultiOmicsTransformer._median_absolute_deviation(block)` | method | 117–127 | Return the per-gene median absolute deviation, ignoring missing values. |
| `MultiOmicsTransformer._apply_impute(block, medians)` | method | 130–144 | Fill missing values with the supplied per-gene medians. |
| `MultiOmicsTransformer.fit(self, X, y)` | method | 146–211 | Select genes and fit imputation medians and scalers on ``X``. |
| `MultiOmicsTransformer.transform(self, X)` | method | 213–233 | Apply the fitted selection, imputation, log and scaling. |
| `MultiOmicsTransformer.get_selected_gene_names(self)` | method | 235–244 | Return the genes selected by the last ``fit``. |
| `MultiOmicsTransformer.get_feature_names_out(self, input_features)` | method | 246–262 | Return output column names as ``modality:gene``. |

#### `mogformer.evaluation.estimator`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `split_modalities(features, n_selected)` | function | 40–66 | Split a transformed matrix back into its three modality blocks. |
| `MOGFormerClassifier` | class | 69–408 | Train and score the graph transformer through the scikit-learn API. |
| `MOGFormerClassifier.__init__(self, selected_genes, string_cache, grn_cache, num_classes, d, pe_dim, pe_method, mini_heads, global_heads, global_layers, dropout, rna_dropout_prob, cnv_dropout_prob, meth_dropout_prob, max_distance, attention_bias_mode, numerical_tokenizer, unimodal_dropout_fill, lambda_gate, use_grn, lr, min_lr, weight_decay, max_epochs, patience, batch_size, inner_val_frac, focal_gamma, seed, device)` | method | 80–178 | Store hyperparameters verbatim for ``sklearn.base.clone``. |
| `MOGFormerClassifier._resolve_device(self)` | method | 180–184 | Return the device to train on. |
| `MOGFormerClassifier._build_graph_tensors(self, device)` | method | 186–214 | Build the fold's graph tensors from the selected genes. |
| `MOGFormerClassifier._make_loader(self, features, labels, shuffle)` | method | 216–241 | Wrap one split's modality blocks in a data loader. |
| `MOGFormerClassifier.fit(self, X, y)` | method | 243–352 | Train on one fold, early-stopping on an inner split. |
| `MOGFormerClassifier._infer(self, loader, device)` | method | 354–383 | Run the fitted model over a loader without gradients. |
| `MOGFormerClassifier.predict_proba(self, X)` | method | 385–397 | Return class probabilities for held-out patients. |
| `MOGFormerClassifier.predict(self, X)` | method | 399–408 | Return the most likely class for each held-out patient. |

#### `mogformer.evaluation.metrics`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `compute_fold_metrics(y_true, y_pred, y_proba, class_labels, class_names)` | function | 35–122 | Score one fold's predictions. |
| `_safe_macro_average_precision(binarised, proba, present)` | function | 125–143 | Average precision over the classes present in this fold. |
| `fold_confusion(y_true, y_pred, class_labels)` | function | 146–160 | Return one fold's confusion matrix with a fixed class order. |
| `nadeau_bengio_correction(n_splits)` | function | 163–180 | Return the test-to-train ratio used by the corrected-variance formula. |
| `aggregate_across_folds(per_fold_values, n_splits, alpha)` | function | 183–225 | Summarise one metric over the estimates of a repeated partition. |

#### `mogformer.evaluation.plots`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `apply_house_style()` | function | 41–57 | Set the matplotlib defaults used by every figure in the project. |
| `save_figure(figure, directory, name)` | function | 60–81 | Write a figure as both PNG and SVG. |
| `plot_model_comparison(per_fold, directory, name, metric, references)` | function | 84–141 | Draw per-fold scores per model as a box plot with the individual folds. |
| `plot_interval_forest(summary, directory, name, metric)` | function | 144–195 | Draw each model's mean with both confidence intervals. |
| `plot_per_class_f1(summary, class_names, directory, name)` | function | 198–247 | Draw per-class F1 for every model. |
| `plot_critical_difference(average_ranks, critical_difference, directory, name)` | function | 250–303 | Draw a critical-difference diagram over average ranks. |
| `plot_confusion(matrix, class_names, directory, name, normalise)` | function | 306–359 | Draw a confusion matrix, row-normalised by default. |

#### `mogformer.evaluation.registry`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `HarnessContext` | class | 37–65 | Everything a model factory needs to build its pipeline for one run. |
| `build_preprocessor(context, modalities, top_k, curated)` | function | 68–92 | Construct a fold-local preprocessing stage for one model. |
| `BalancedXGB` | class | 95–118 | Gradient-boosted trees that rebalance classes at fit time. |
| `BalancedXGB.__new__(cls, **kwargs)` | method | 106–118 | Return a configured, class-balancing ``XGBClassifier``. |
| `PAM50NearestCentroid` | class | 121–209 | Classify by rank correlation to per-class expression centroids. |
| `PAM50NearestCentroid.__init__(self, temperature)` | method | 135–141 | Store the softmax temperature used to turn correlations into scores. |
| `PAM50NearestCentroid._rank_standardise(matrix)` | method | 144–161 | Rank each row and scale it to unit norm. |
| `PAM50NearestCentroid.fit(self, X, y)` | method | 163–178 | Compute one centroid per class from the training fold. |
| `PAM50NearestCentroid._correlate(self, X)` | method | 180–184 | Return each sample's rank correlation to every centroid. |
| `PAM50NearestCentroid.predict(self, X)` | method | 186–195 | Assign each sample to its best-correlating centroid. |
| `PAM50NearestCentroid.predict_proba(self, X)` | method | 197–209 | Return softmax-scaled correlations as class probabilities. |
| `LateFusionClassifier` | class | 212–279 | Average the predictions of one independent model per modality. |
| `LateFusionClassifier.__init__(self, base_pipeline_factory, modalities)` | method | 224–237 | Store the factory and the modalities to fuse. |
| `LateFusionClassifier.fit(self, X, y)` | method | 239–255 | Fit one pipeline per modality. |
| `LateFusionClassifier.predict_proba(self, X)` | method | 257–268 | Average the per-modality probability estimates. |
| `LateFusionClassifier.predict(self, X)` | method | 270–279 | Return the highest-probability class after fusion. |
| `ModelSpec` | class | 283–301 | One registered model and how to build, tune and describe it. |
| `build_registry()` | function | 304–457 | Construct the full model registry. |
| `late_fusion_spec(name, base)` | function | 460–486 | Wrap a registered model as its late-fusion counterpart. |

#### `mogformer.evaluation.runner`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `load_cohort(config)` | function | 48–65 | Load the cohort described by a configuration. |
| `build_context(config, data, pam50_genes)` | function | 68–92 | Assemble the context every model factory needs. |
| `score_fold(spec, context, data, fold, n_inner_splits, n_search_iter)` | function | 95–155 | Fit and score one model on one fold. |
| `run_cross_validation(config, models, pam50_genes)` | function | 158–244 | Score every requested model across the shared partition. |
| `summarise(per_fold, n_splits)` | function | 247–262 | Aggregate a long-format per-fold frame into one row per model and metric. |
| `render_leaderboard(summary, metric)` | function | 265–294 | Render the leaderboard for one metric as Markdown. |
| `load_per_fold_scores(path, metric)` | function | 297–315 | Read per-fold scores for one metric, ready for paired comparison. |

#### `mogformer.evaluation.stats`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `corrected_resampled_ttest(differences, n_splits)` | function | 41–86 | Run the Nadeau–Bengio corrected paired t-test on per-fold differences. |
| `rank_biserial(differences)` | function | 89–111 | Compute the matched-pairs rank-biserial effect size. |
| `pairwise_compare(per_fold, n_splits)` | function | 114–170 | Compare every pair of models on their shared folds. |
| `friedman_nemenyi(per_fold)` | function | 173–216 | Rank models across folds and compute the Nemenyi critical difference. |

#### `mogformer.graph.grn`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `GRNSignedCache` | class | 26–123 | Parse CollecTRI once over a gene universe, then serve induced subgraphs. |
| `GRNSignedCache.__init__(self, grn_path, universe_genes)` | method | 38–71 | Parse the regulatory network over ``universe_genes``. |
| `GRNSignedCache.induced_signed_adjacency(self, selected_genes, direction, use_sign)` | method | 73–123 | Build the signed adjacency of the subgraph induced on selected genes. |

#### `mogformer.graph.positional_encoding`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `GraphPositionalEncoding` | class | 19–140 | Compute a fixed structural encoding for every node of a gene graph. |
| `GraphPositionalEncoding.__init__(self, pe_dim, method)` | method | 30–51 | Initialise the encoder. |
| `GraphPositionalEncoding.forward(self, adjacency)` | method | 54–76 | Encode every node of ``adjacency``. |
| `GraphPositionalEncoding._random_walk(self, adjacency)` | method | 78–100 | Return per-node random-walk return probabilities. |
| `GraphPositionalEncoding._laplacian(self, adjacency)` | method | 102–140 | Return the low-frequency eigenvectors of the normalised Laplacian. |

#### `mogformer.graph.spd`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `compute_shortest_path_matrix(adjacency, max_distance)` | function | 14–60 | Compute the truncated all-pairs shortest-path matrix of a gene graph. |

#### `mogformer.graph.string_graph`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `parse_string_edges(ppi_path, alias_path, universe_genes, confidence_threshold)` | function | 26–74 | Read STRING interactions and return undirected gene-symbol edges. |
| `StringGraphCache` | class | 77–142 | Parse STRING once over a gene universe, then serve induced subgraphs. |
| `StringGraphCache.__init__(self, ppi_path, alias_path, universe_genes, confidence_threshold)` | method | 91–117 | Parse the interactome over ``universe_genes``. |
| `StringGraphCache.induced_adjacency(self, selected_genes)` | method | 119–142 | Build the binary adjacency of the subgraph induced on selected genes. |

#### `mogformer.models.classifier`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MultiOmicsGraphClassifier` | class | 24–186 | Predict a tumor subtype from three omics modalities and a gene graph. |
| `MultiOmicsGraphClassifier.__init__(self, num_classes, d, pe_dim, mini_heads, global_heads, global_layers, dropout, rna_dropout_prob, cnv_dropout_prob, meth_dropout_prob, max_distance, attention_bias_mode, gene_id_embedding, n_universe, gene_ids, pretrained_gene_emb, pretrained_emb_adapter_rank, numerical_tokenizer, plr_n_frequencies, plr_sigma, lambda_gate, unimodal_dropout_fill, use_grn)` | method | 36–136 | Assemble the four stages. |
| `MultiOmicsGraphClassifier.forward(self, rna, cnv, methy, graph_pe, spd_matrix, grn_matrix, eval_mask, structural_bias, return_attention)` | method | 138–186 | Predict subtype logits for a batch of patients. |

#### `mogformer.models.layers.decoder`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `DualHeadDecoder` | class | 29–105 | Reconstruct per-gene, per-modality values from a shared address grid. |
| `DualHeadDecoder.__init__(self, d)` | method | 37–52 | Build the two heads and the modulation projections. |
| `DualHeadDecoder._build_block(d)` | method | 55–59 | Return the two-layer block used throughout the decoder. |
| `DualHeadDecoder.forward(self, summary, hidden_last, gene_embedding, modality_embedding)` | method | 61–105 | Predict every gene-by-modality value from both heads. |

#### `mogformer.models.layers.gated_fusion`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `GatedFusion` | class | 29–108 | Fuse per-gene modality tokens by a competing softmax gate. |
| `GatedFusion.__init__(self, d, dropout, rna_dropout_prob, cnv_dropout_prob, meth_dropout_prob, unimodal_dropout_fill)` | method | 37–74 | Build the gate scorer and the output projection. |
| `GatedFusion.forward(self, z_rna, z_cnv, z_methy, eval_mask)` | method | 76–108 | Fuse the three modality streams into one token per gene. |

#### `mogformer.models.layers.global_transformer`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `GlobalGraphTransformer` | class | 21–137 | Run structurally biased attention across all genes of a patient. |
| `GlobalGraphTransformer.__init__(self, d, pe_dim, num_heads, num_layers, dim_feedforward, max_distance, attention_bias_mode, dropout, lambda_gate, use_grn)` | method | 37–93 | Build the projector, the summary token and the attention stack. |
| `GlobalGraphTransformer.forward(self, h, graph_pe, spd_matrix, grn_matrix, structural_bias)` | method | 95–137 | Contextualise gene tokens against one another. |

#### `mogformer.models.layers.masking`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MaskedMultiModalMasker` | class | 28–120 | Draw disjoint whole-gene and single-modality masks per patient. |
| `MaskedMultiModalMasker.__init__(self, mask_gene_frac, mask_modality_frac, mask_modality_weights)` | method | 37–75 | Store the masking rates and the per-modality weighting. |
| `MaskedMultiModalMasker.forward(self, x)` | method | 78–120 | Draw a mask for one batch of patients. |

#### `mogformer.models.layers.mini_transformer`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MiniTransformer` | class | 24–119 | Fuse per-gene modality tokens by attention over a four-token sequence. |
| `MiniTransformer.__init__(self, d, num_heads, dropout, rna_dropout_prob, cnv_dropout_prob, meth_dropout_prob, unimodal_dropout_fill)` | method | 33–75 | Build the summary token, attention and feed-forward sublayers. |
| `MiniTransformer.forward(self, z_rna, z_cnv, z_methy, eval_mask)` | method | 77–119 | Fuse the three modality streams into one token per gene. |

#### `mogformer.models.layers.modality_dropout`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `ModalityDropout` | class | 27–160 | Hide at most one modality per gene during training. |
| `ModalityDropout.__init__(self, d, rna_dropout_prob, cnv_dropout_prob, meth_dropout_prob, fill, mask_token_std)` | method | 37–84 | Store the probabilities and, if needed, build the mask tokens. |
| `ModalityDropout.forward(self, z_rna, z_cnv, z_methy, eval_mask)` | method | 86–160 | Apply stochastic dropout in training, or deterministic masking in eval. |

#### `mogformer.models.layers.modality_lifting`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `GeneEmbeddingAdapter` | class | 32–84 | Project frozen pretrained gene embeddings, optionally adapting them first. |
| `GeneEmbeddingAdapter.__init__(self, pretrained, d_out, adapter_rank)` | method | 44–68 | Register the frozen embeddings and build the projection. |
| `GeneEmbeddingAdapter.forward(self)` | method | 70–84 | Return the projected gene embeddings. |
| `PeriodicLinearTokenizer` | class | 87–139 | Tokenize a scalar through a learnable periodic basis. |
| `PeriodicLinearTokenizer.__init__(self, d, n_frequencies, sigma)` | method | 107–126 | Build the periodic basis and the two projections. |
| `PeriodicLinearTokenizer.forward(self, x)` | method | 128–139 | Tokenize a batch of per-gene scalars. |
| `ModalityLifting` | class | 142–354 | Turn three per-gene scalars into three modality-tagged token streams. |
| `ModalityLifting.__init__(self, d, gene_id_embedding, n_universe, gene_ids, gene_id_std, numerical_tokenizer, plr_n_frequencies, plr_sigma, mask_token_std, pretrained_gene_emb, pretrained_emb_adapter_rank)` | method | 152–245 | Build the tokenizers, embeddings and mask tokens. |
| `ModalityLifting._build_mlp_tokenizer(d)` | method | 248–252 | Return the default two-layer scalar tokenizer. |
| `ModalityLifting._init_weights(self)` | method | 254–271 | Initialise tokenizer weights, modality embeddings and mask tokens. |
| `ModalityLifting.get_gene_embedding(self)` | method | 273–289 | Return the per-gene identity embeddings currently in use. |
| `ModalityLifting.get_modality_embeddings(self)` | method | 291–297 | Return the three modality embeddings, shared with the decoder address. |
| `ModalityLifting.forward(self, rna, cnv, methy, mask_bool)` | method | 299–354 | Lift the three scalar streams into token streams. |

#### `mogformer.models.layers.structural_attention`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `StructuralGraphAttention` | class | 32–198 | Multi-head attention with a distance bias and an optional regulatory bias. |
| `StructuralGraphAttention.__init__(self, d_model, num_heads, max_distance, mode, dropout, lambda_gate)` | method | 52–112 | Build the projections and the bias tables. |
| `StructuralGraphAttention._structural_bias(self, spd_matrix, grn_matrix)` | method | 114–136 | Build the additive bias from distances and regulatory edges. |
| `StructuralGraphAttention.forward(self, h, spd_matrix, grn_matrix, structural_bias)` | method | 138–198 | Attend over the token sequence. |
| `StructuralAttentionBlock` | class | 201–272 | Pre-layer-norm transformer block wrapping :class:`StructuralGraphAttention`. |
| `StructuralAttentionBlock.__init__(self, d_model, num_heads, dim_feedforward, max_distance, mode, dropout, lambda_gate)` | method | 211–249 | Build the attention and feed-forward sublayers. |
| `StructuralAttentionBlock.forward(self, x, spd_matrix, grn_matrix, structural_bias)` | method | 251–272 | Apply structural attention then the feed-forward sublayer. |

#### `mogformer.models.ssl`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MOGFormerSSL` | class | 33–249 | Masked multi-modal encoder over the gene graph, with a dual-head decoder. |
| `MOGFormerSSL.__init__(self, d, pe_dim, mini_heads, global_heads, global_layers, dropout, max_distance, attention_bias_mode, use_grn, gene_id_embedding, n_universe, gene_ids, lambda_gate, fusion_type, mask_gene_frac, mask_modality_frac, mask_modality_weights, rna_only, cnv_off)` | method | 46–156 | Assemble encoder, masker and decoder. |
| `MOGFormerSSL.forward(self, rna, cnv, methy, graph_pe, spd_matrix, grn_matrix, mask, mask_bool, structural_bias)` | method | 158–249 | Encode a batch, and reconstruct masked values when masking is on. |

#### `mogformer.training.losses`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `MultiClassFocalLoss` | class | 25–99 | Focal cross-entropy with optional per-class weights. |
| `MultiClassFocalLoss.__init__(self, alpha, gamma, reduction)` | method | 34–66 | Store the weighting and focusing configuration. |
| `MultiClassFocalLoss.alpha_weights(self)` | method | 69–72 | Return the class weights, already on the module's device. |
| `MultiClassFocalLoss.forward(self, inputs, targets)` | method | 74–99 | Compute the focal loss. |
| `sqrt_dampened_weights(y, num_classes)` | function | 102–130 | Compute class weights as the square root of inverse frequency. |
| `masked_huber_dual(xhat_global, xhat_local, targets, mask_bool, lambda_global, lambda_local, delta)` | function | 133–174 | Score both reconstruction heads on the masked entries only. |
| `participation_ratio(embeddings)` | function | 178–197 | Estimate how many dimensions an embedding actually uses. |

#### `mogformer.training.seed`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `seed_everything(seed, deterministic)` | function | 21–40 | Seed Python, NumPy and PyTorch, including CUDA. |

#### `mogformer.training.trainer`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `ValidationScores` | class | 40–57 | One epoch's monitoring diagnostics. |
| `TrainingHistory` | class | 61–84 | Per-epoch diagnostics collected during pretraining. |
| `MaskedReconstructionTrainer` | class | 87–380 | Train the self-supervised encoder and monitor it for collapse. |
| `MaskedReconstructionTrainer.__init__(self, model, train_loader, val_loader, device, spd_matrix, graph_pe, grn_matrix, lr, weight_decay, lambda_global, lambda_local, delta)` | method | 96–158 | Set up the optimiser with a separate group for the bias gates. |
| `MaskedReconstructionTrainer._to_device(self, batch)` | method | 160–165 | Move one batch's three modality tensors onto the training device. |
| `MaskedReconstructionTrainer._forward(self, rna, cnv, methy, mask)` | method | 167–171 | Run the encoder with this trainer's graph tensors. |
| `MaskedReconstructionTrainer.train_epoch(self)` | method | 173–199 | Run one pass over the training batches. |
| `MaskedReconstructionTrainer.validate_epoch(self)` | method | 202–280 | Score the monitoring split and collect collapse diagnostics. |
| `MaskedReconstructionTrainer._record_gate_magnitudes(self)` | method | 282–295 | Append the regulatory bias magnitudes of the first block. |
| `MaskedReconstructionTrainer.fit_early(self, max_epochs, patience, log_every)` | method | 297–358 | Train until the monitoring loss stops improving. |
| `MaskedReconstructionTrainer.harvest_gates(self)` | method | 361–380 | Collect fusion gate weights and the methylation input that drove them. |

#### `utils.build_registry`

| Symbol | Kind | Lines | Purpose |
| --- | --- | --- | --- |
| `Symbol` | class | 67–97 | One class, function or method discovered in a source file. |
| `Symbol.qualified_name(self)` | method | 95–97 | Return ``Owner.name`` for methods and ``name`` for everything else. |
| `_module_path(path)` | function | 100–117 | Convert a file path into a dotted module path. |
| `_first_docstring_line(node)` | function | 120–131 | Return the first non-empty line of ``node``'s docstring, or ``""``. |
| `_render_signature(node)` | function | 134–143 | Render a function's parameter names as a compact signature string. |
| `_iter_source_files(roots)` | function | 146–165 | Yield every Python file under ``roots``, skipping caches and venvs. |
| `parse_symbols(path)` | function | 168–229 | Extract every top-level and nested definition from one source file. |
| `collect(roots)` | function | 232–237 | Parse every source file under ``roots`` and return the merged symbols. |
| `find_duplicates(symbols)` | function | 240–265 | Group symbols that share a name across different modules. |
| `_escape(text)` | function | 268–270 | Escape pipe characters so a cell cannot break the Markdown table. |
| `render_registry(symbols)` | function | 273–318 | Render the package symbol registry as Markdown. |
| `render_migration_table(symbols)` | function | 321–333 | Render a per-file count of symbols still awaiting migration. |
| `build_block()` | function | 336–375 | Assemble the full generated section of ``ARCHITECTURE.md``. |
| `write_block(check_only)` | function | 378–433 | Rewrite the generated block in ``ARCHITECTURE.md``. |
| `main(argv)` | function | 436–445 | Entry point for ``python -m utils.build_registry``. |

<sub>Largest modules: `mogformer.evaluation.registry` (19), `mogformer.analysis.clustering` (18), `utils.build_registry` (15), `mogformer.models.layers.modality_lifting` (13), `mogformer.analysis.probe_cis` (12)</sub>

### Duplicate names

None. Every first-party name resolves to one definition.

### Legacy sources still in the tree

| Legacy source | Symbols | Status |
| --- | --- | --- |
| `scripts/figures/clust.py` | 18 | not yet migrated |
| `scripts/GRN download.py` | 9 | not yet migrated |
| `scripts/figures/generate_metabric_plots.py` | 3 | not yet migrated |
| `scripts/figures/generate_surv_plots.py` | 3 | not yet migrated |
| `scripts/figures/ari.py` | 2 | not yet migrated |
| `scripts/figures/generate_event_counts_comparison.py` | 1 | not yet migrated |
| `scripts/figures/generate_luma_os_pfi_distributions.py` | 1 | not yet migrated |
| `scripts/figures/viz_networks.py` | 1 | not yet migrated |

<!-- END GENERATED REGISTRY -->
