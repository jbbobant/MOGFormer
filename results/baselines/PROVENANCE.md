# MOGFormer Phase 0 — leakage-free evaluation harness + baseline suite

Wraps your existing data around a rigorous, fold-refit harness so every model
(B0–B5 now; B6/B7/MOGFormer later) is compared through the **identical** protocol.

## Why this replaces the old flow
The original `preprocess.py` fit MAD selection + scaling **once** on a single
split and wrote CSVs; `train_baselines.py` then re-split those processed CSVs.
That leaks fold information and makes baselines vs. MOGFormer non-comparable.
Here, selection + impute + log1p(RNA) + StandardScaler are an sklearn
transformer (`MultiOmicsTransformer`) that **refits inside every fold**, driven
by one shared set of patient-level folds (`results/folds.json`).

## Layout
```
harness/
  data.py        raw (genes x patients) loader, patient-level align + dedup assert, label encode
  preprocess.py  MultiOmicsTransformer: per-fold MAD consensus selection, curated force-include,
                 median impute, log1p(RNA), StandardScaler. active_modalities switches blocks.
  cv.py          5x5 stratified patient-level folds, no-leakage assert, persist/load folds.json
  metrics.py     full metric suite + naive 95% CI + Nadeau-Bengio corrected CI
  models.py      registry: B0a/B0b/B1/B2/B3/B4/B5 active; B6/B7/M deferred stubs.
                 PAM50NearestCentroid, CV-safe BalancedXGB, LateFusionClassifier.
  stats.py       paired Wilcoxon, corrected-resampled t-test, effect size, Friedman+Nemenyi CD
  plots.py       V01–V14 (PNG+SVG @300dpi). UMAP optional -> PCA fallback.
  run_phase0.py  orchestrates the whole matrix + writes results/
tests/
  test_smoke.py        proves no-leakage / per-fold refit / curated force-include (synthetic)
  test_models.py       every active model fits + predict_proba through a shared fold
  test_run_phase0.py   full end-to-end synthetic run; checks all artifacts + V01–V14
config.yaml
```

## Run
```bash
pip install scikit-learn scipy pandas numpy xgboost pyyaml matplotlib umap-learn
# put CGenes.txt and PAL50.txt in data/raw/ alongside the four data_*.csv
python -m harness.run_phase0 --config config.yaml --data_dir data/raw --results_dir results
# sanity-check the harness itself first:
python tests/test_run_phase0.py
```

## Outputs (results/)
`config.yaml` (+ versions, resolved cohort), `folds.json`, `metrics_per_fold.csv`,
`metrics_summary.csv`, `metrics_summary.md` (leaderboard), `stats_pairwise.csv`,
`figures/V01..V14.{png,svg}`.

## Known caveats to revisit (flagged, not hidden)
- **Late fusion is untuned** in this pass (prob-averaging at default params) while
  early fusion is inner-tuned. The early-vs-late comparison carries this asymmetry
  until you flip it (would triple tuning cost).
- **Sweep reuse**: 6(a)/6(b)/6(c) reuse B5's modal-best hyperparameters from the
  main run rather than re-tuning at every grid point (budget). Main leaderboard
  uses proper per-fold nested tuning.
- **gene_count_grid includes 100** to anchor the reference operating point on V09.
- **V11 embeddings** are fit on all data (exploratory visualization only; never
  used for any performance estimate).
- **sklearn >= 1.8** deprecates `penalty='elasticnet'`; pin version or adjust B2 if
  you see the FutureWarning. The resolved version is logged in results/config.yaml.

## Next (MOGFormer)
The registry has an `M_mogformer` slot. Wrapping it = build a sklearn-compatible
estimator whose `fit` runs your existing training loop (architecture unchanged),
early-stopping on an inner-validation split, with `predict_proba`. It then flows
through the same folds/metrics/stats with zero protocol drift. Per-fold graph PE
regeneration (because gene selection is fold-local) is the open design decision
we deferred.
