# MOGFormer

A multi-omics graph transformer over RNA-seq, copy number and DNA methylation,
used to ask what such a model actually learns about breast cancer — and, in
particular, about Luminal A tumours.

> **Status: refactor complete.** The Kaggle notebooks and the two parallel
> module trees have been consolidated into `src/mogformer/`, covered by 155
> tests, and the superseded sources removed. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the layout, conventions and the
> generated symbol registry.

---

## The short version

PAM50 breast cancer subtypes are assigned by a 50-gene **RNA** centroid
classifier. Training a trimodal model to recover them therefore asks copy number
and methylation to contribute to a label that, by construction, holds no
information they uniquely carry. That framing turned out to be the project's
most useful result, and the work reorganised around it.

Three findings follow, each backed by a pre-registered protocol or an explicit
negative control.

### 1. On a circular label, fusion buys nothing

Scored through a shared, leakage-free harness, the transformer does not lead the
classical baselines — which is the expected outcome when the target is
effectively an RNA lookup.

| Model | Classes | n | macro-F1 |
| --- | --- | --- | --- |
| XGBoost (early fusion) | 5 | 25 | **0.852** |
| MOGFormer | 5 | 25 | 0.835 [0.816, 0.853] |
| PAM50 nearest-centroid (RNA) | 5 | 25 | 0.738 |
| XGBoost (early fusion) | 4 | 25 | **0.926** [0.917, 0.935] |
| MOGFormer | 4 | 10 | 0.878 [0.857, 0.900] |

### 2. The unsupervised split is stable, redundant, and not prognostic

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

The [Phase 5 protocol](results/SSL/SURV_CLEAN/TCGA/phase5/protocol/step1_protocol.md)
was locked before any survival data was touched.

### 3. The model routes effects along real regulatory edges — but not with the right sign

Intervening on a gene's methylation and reading its reconstructed expression
recovers the silencing direction genome-wide, and perturbations of a
transcription factor move its real targets more than matched non-targets.

| Quantity | Value |
| --- | --- |
| cis median ρ (methylation → expression) | −0.206, Wilcoxon p = 1.1e−72 |
| trans effect on edges vs matched non-edges | \|slope\| 0.0022 vs 0.0015, p = 3.4e−15 |
| edges whose interval excludes zero | 98.1%, Wilcoxon p = 9.7e−51 |
| correlation with observed co-expression | Spearman 0.68 |

**The direction of those effects is not recovered.** Sign concordance with
CollecTRI is 60.0% over 903 edges. Against a coin flip that is
overwhelmingly significant — and the coin flip is the wrong null. The edge set
is 77% activating and the model's slopes carry their own sign bias, so a
label-permutation null that preserves both sits at 58.6% and gives
**p = 0.16**. Worse, always guessing "activating" scores **77.2%**, so the
model is beaten by a constant predictor.

| Sign concordance | Value |
| --- | --- |
| observed | 0.600 |
| label-permutation null | 0.586 (p = 0.16, z = 1.09) |
| trivial majority baseline | **0.772** — not beaten |

Two further limits are load-bearing and reported rather than buried: a
permuted-graph null retains **63.9%** of the trans effect, so only about a third
is attributable to graph identity; and the gene-level specificity test does not
separate curated silenced genes from controls (p = 0.49), so the cis claim is a
correct *global* direction, not per-gene discrimination.

For context, in observed METABRIC expression the same CollecTRI edge signs reach
0.548 concordance against the same 0.760 trivial baseline — the raw data does not
recover them either, which is the honest frame for what the model can and cannot
be expected to learn.

---

## Data

Not distributed with the repository (≈5.6 GB).

| Source | Contents |
| --- | --- |
| TCGA-BRCA (cBioPortal) | RNA-seq, CNV, methylation, clinical — 949 patients |
| METABRIC (cBioPortal) | External validation cohort |
| STRING v12.0 | Human protein–protein interactions |
| CollecTRI | Signed transcription-factor to target edges |

Place raw downloads under `data/raw/` and cleaned matrices under `data/clean/`.
Paths are configured, never hardcoded.

---

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra baselines --extra analysis
```

```bash
uv run pytest
```

```bash
uv run ruff check src utils tests
```

```bash
uv run python -m utils.build_registry --check
```

Then run a phase:

```bash
uv run mogformer folds --config configs/tcga.yaml
```

```bash
uv run mogformer train --config configs/tcga.yaml --models B5_xgboost B1_pam50_centroid
```

The registry command above is the repository's guard against duplicate implementations. It
fails when the symbol registry in `ARCHITECTURE.md` is stale, or when any name
is defined in more than one module — the defect that previously caused the
baselines and the transformer to be scored on different patient cohorts.

---

## Repository map

| Path | Contents |
| --- | --- |
| `src/mogformer/` | The package |
| `tests/` | Unit and integration tests on synthetic data |
| `utils/` | Repository tooling, including the registry generator |
| `scripts/` | Data acquisition and figure code not yet folded into the package |
| `results/` | Cross-validation runs, SSL phases, survival analyses |
| `results/baselines/` | Classical baseline harness outputs |
| `P9/`, `PROBE/` | Interventional probe outputs |
| `figures/` | Network and cohort figures |

---

## License

MIT — see [LICENSE](LICENSE).
